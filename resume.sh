#!/usr/bin/env bash
# Resume the Conditions Dashboard build.
# Run from repo root: bash resume.sh

claude --resume "Continue building the Conditions Dashboard. \
Read finance/BACKLOG.md for the full backlog (3 epics, 10 stories). \
Start with E3-S1 (app skeleton + launcher registration), then E1-S1 + E1-S2 (SPY/QQQ trend + VIX indicator). \
Use /architect for design decisions, /developer for implementation. \
momentum_data.py fragmentation fix is already done (uncommitted)."
