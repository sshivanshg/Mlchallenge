"""Text normalization: pure string transforms, open-set country labels."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

# Legal / address expansions — applied as token rewrites after tokenization.
_SUFFIX_MAP = {
    "corp": "corporation",
    "incorp": "incorporated",
    "inc": "incorporated",
    "ltd": "limited",
    "llc": "limited",
    "pvt": "private",
    "co": "company",
    "plc": "public",
    "llp": "limited",
    "sa": "sa",
    "sas": "sas",
    "sarl": "sarl",
    "gmbh": "gmbh",
}

_ADDR_MAP = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "ln": "lane",
    "dr": "drive",
    "hwy": "highway",
    "pkwy": "parkway",
    "ste": "suite",
    "apt": "apartment",
    "bldg": "building",
    "fl": "floor",
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
}

_AND_RE = re.compile(r"\s*&\s*")
_NON_ALNUM = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")
_DIGIT_RUN = re.compile(r"\d+")


def nfkc_casefold(text: str) -> str:
    if not text:
        return ""
    return unicodedata.normalize("NFKC", text).casefold()


def normalize_text(text: str, kind: str = "name") -> str:
    """Normalize a business name or address into a comparable token string."""
    t = nfkc_casefold(text)
    t = _AND_RE.sub(" and ", t)
    t = _NON_ALNUM.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    if not t:
        return ""
    mapping = _SUFFIX_MAP if kind == "name" else {**_SUFFIX_MAP, **_ADDR_MAP}
    tokens = [mapping.get(tok, tok) for tok in t.split()]
    return " ".join(tokens)


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return text.split()


def significant_tokens(tokens: list[str], min_len: int = 2) -> list[str]:
    stop = {"the", "and", "of", "a", "an", "for", "to", "in", "on", "at", "by"}
    return [t for t in tokens if len(t) >= min_len and t not in stop]


def numeric_tokens(text: str) -> list[str]:
    return _DIGIT_RUN.findall(text or "")


@lru_cache(maxsize=1)
def _phonetic_table() -> dict[str, str]:
    # Lightweight soundex-ish first-letter buckets for blocking (no external deps).
    return {}


def name_prefix(norm_name: str, n: int = 3) -> str:
    compact = "".join(norm_name.split())
    return compact[:n] if compact else ""


def country_key(country: str) -> str:
    """Open-set country label: normalize whitespace/case only; never filter values."""
    return nfkc_casefold((country or "").strip())
