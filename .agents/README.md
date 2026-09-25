# AWS Entity Resolution Agent Skill (vendored)

Source: https://github.com/sshivanshg/aws-entity-resolution-skill

Installed into this challenge repo at:

- `.agents/skills/aws-entity-resolution/` (canonical Agent Skills path)
- `.claude/skills/aws-entity-resolution/` (Claude Code mirror)
- `.cursor/skills/aws-entity-resolution/` (Cursor mirror)

This repository contains the challenge workspace **plus** the skill framework.

## Official dataset

Use the **real** competition TSVs (not synthetic stubs):

```bash
bash scripts/download_dataset.sh
```

Release: https://github.com/sshivanshg/Mlchallenge/releases/tag/dataset-v1  
See `data/README.md`.

## Verify skill metric

```bash
python3 -m unittest discover -s .agents/skills/aws-entity-resolution/scripts -p 'test_*.py'
```

## End-to-end readiness (skill tests + official data)

```bash
bash scripts/setup_and_smoke.sh
```

Agents should follow `.agents/skills/aws-entity-resolution/SKILL.md` and the
`references/` docs before changing the ER pipeline under
`code/business_entity_resolution/`.
