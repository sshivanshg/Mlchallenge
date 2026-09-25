# AWS Entity Resolution Agent Skill (vendored)

Source: https://github.com/sshivanshg/aws-entity-resolution-skill

Installed into this challenge repo at:

- `.agents/skills/aws-entity-resolution/` (canonical Agent Skills path)
- `.claude/skills/aws-entity-resolution/` (Claude Code mirror)
- `.cursor/skills/aws-entity-resolution/` (Cursor mirror)

This repository contains the challenge workspace **plus** the skill framework.
The skill itself does not include challenge data or a competition solution.

## Verify skill metric

```bash
python3 -m unittest discover -s .agents/skills/aws-entity-resolution/scripts -p 'test_*.py'
```

## End-to-end smoke (skill tests + synthetic pipeline + validator)

```bash
bash scripts/setup_and_smoke.sh
```

Agents should follow `.agents/skills/aws-entity-resolution/SKILL.md` and the
`references/` docs before changing the ER pipeline under
`code/business_entity_resolution/`.
