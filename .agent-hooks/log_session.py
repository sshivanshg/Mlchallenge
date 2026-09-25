#!/usr/bin/env python3
"""Archive agent sessions (Cursor / Claude Code / Codex) under <project>/sessions/.

Usage: log_session.py <cursor|claude|codex>   (hook payload JSON on stdin)

Layout:
  sessions/<tool>/<session_id>/meta.json        first-seen info
  sessions/<tool>/<session_id>/events.jsonl     one line per hook event
  sessions/<tool>/<session_id>/transcript.jsonl latest copy of the tool's transcript
  sessions/index.jsonl                          one line per new session

Must never break the agent: every failure is swallowed.
"""
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SESSIONS_DIR = PROJECT_ROOT / "sessions"
MAX_FIELD_CHARS = 20_000
TRANSCRIPT_EVENTS = {
    "stop", "sessionEnd", "afterAgentResponse", "preCompact",
    "Stop", "SessionEnd", "PreCompact", "SubagentStop",
}


def _truncate(value):
    if isinstance(value, str) and len(value) > MAX_FIELD_CHARS:
        return value[:MAX_FIELD_CHARS] + f"... [truncated {len(value) - MAX_FIELD_CHARS} chars]"
    if isinstance(value, dict):
        return {k: _truncate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_truncate(v) for v in value]
    return value


def _session_id(payload):
    for key in ("conversation_id", "session_id", "sessionId", "thread_id"):
        if payload.get(key):
            return str(payload[key])
    return "unknown-" + datetime.now().strftime("%Y%m%d")


def main():
    tool = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        payload = {"raw": raw}

    now = datetime.now(timezone.utc).isoformat()
    event = payload.get("hook_event_name") or payload.get("event") or "unknown"
    session_id = _session_id(payload)
    session_dir = SESSIONS_DIR / tool / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    meta_path = session_dir / "meta.json"
    if not meta_path.exists():
        meta = {
            "tool": tool,
            "session_id": session_id,
            "started_at": now,
            "cwd": payload.get("cwd") or (payload.get("workspace_roots") or [None])[0],
            "model": payload.get("model"),
            "transcript_path": payload.get("transcript_path"),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        with open(SESSIONS_DIR / "index.jsonl", "a") as f:
            f.write(json.dumps(meta) + "\n")

    with open(session_dir / "events.jsonl", "a") as f:
        f.write(json.dumps({"ts": now, "event": event, "payload": _truncate(payload)}) + "\n")

    transcript = payload.get("transcript_path")
    if transcript and event in TRANSCRIPT_EVENTS:
        src = Path(os.path.expanduser(transcript))
        if src.is_file():
            shutil.copyfile(src, session_dir / ("transcript" + (src.suffix or ".jsonl")))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        try:
            SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
            with open(SESSIONS_DIR / "hook-errors.log", "a") as f:
                f.write(f"{datetime.now().isoformat()} {sys.argv[1:]} {exc!r}\n")
        except Exception:
            pass
    if len(sys.argv) > 1 and sys.argv[1] == "cursor":
        print("{}")
    sys.exit(0)
