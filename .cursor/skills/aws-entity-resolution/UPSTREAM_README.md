# AWS Entity Resolution Agent Skill

Reusable Agent Skill for the AWS Machine Learning Challenge on noisy business entity resolution.

The skill lives at .agents/skills/aws-entity-resolution/ and guides an autonomous coding/research agent through data exploration, validation design, blocking, feature engineering, hard-negative mining, model selection, threshold calibration, singleton handling, error analysis, final inference, and submission checks.

Run the bundled metric tests with:

    python -m unittest discover -s .agents/skills/aws-entity-resolution/scripts -p 'test_*.py'

This repository contains the skill framework only. It does not contain challenge data or a competition solution.
