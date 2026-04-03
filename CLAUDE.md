# CLAUDE.md

Personal quantitative finance research platform with three workstreams:

1. **Tradelog** (`tradelog/`) — Angular + .NET trade logging web app with IBKR Flex Query sync
2. **Research** (`investing_framework/`) — Trading and investing framework documents, research summaries
3. **Finance** (`finance/`) — Python quantitative analysis apps, PyQtGraph dashboards, IBKR data pipeline

## Environment

- **Platform:** Windows 11, bash shell
- **Python:** 3.12+ with `uv` package manager (`uv sync` to install)
- **Backend:** .NET 10, SQL Server, EF Core
- **Frontend:** Angular 21, Tailwind, PrimeNG

## Testing — TDD

All implementation must follow Test-Driven Development:

1. **Write tests first** — before writing any production code, write failing tests that define the expected behavior
2. **Red → Green → Refactor** — confirm the test fails, write minimal code to pass it, then refactor
3. **Test frameworks**: `xUnit` for .NET (`tradelog/backend.Tests/`), `pytest` for Python
4. **Test behavior, not implementation** — assert on observable outcomes, not internal state

## Context Rules

Path-specific context loads automatically from `.claude/rules/` when working with files in each workstream. No manual mode switching needed.

## Available Skills

| Skill | Use For |
|-------|---------|
| `/architect` | Solution design, trade-offs, planning |
| `/developer` | Implementation with plan-first workflow |
| `/reviewer` | Code review with structured checklist |
| `/reqeng` | Requirements engineering |
| `/trader` | Momentum swing trading analysis (5–50 days) |
| `/investor` | Long-term portfolio + DRIFT options income |
| `/writer` | Editing framework documents and presentations |
