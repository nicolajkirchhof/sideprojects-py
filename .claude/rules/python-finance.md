---
paths:
  - "finance/**"
  - "IBKR_CSV_to_PKL_Converter.py"
  - "SmokeTestWindow.py"
---

# Python Finance Applications Context

## Stack

Python 3.12+ with `uv` as package manager. PyQtGraph/PyQt6 for interactive visualizations. IBKR TWS API via `ib_async` for live market data.

## Key Modules

- **`finance/utils/`** — Core infrastructure: data loading, indicators, chart styles, IBKR wrapper, options math
- **`finance/apps/`** — Interactive Qt applications (swing_plot, momentum_dashboard)
- **`finance/swing_pm/`** — Strategy analysis scripts (`# %%` cell markers for IPython/Spyder)
- **`finance/ibkr/`** — IBKR data collection: historical data, options, GEX, scanner
- **`finance/intraday_pm/backtester/`** — Backtesting framework for bracket/stock/options strategies

## Data Pipeline

```
IBKR API / Yahoo Finance / DoltSQL
    → finance/utils/ (load + cache as .pkl)
    → finance/utils/indicators.py (compute)
    → finance/swing_pm/ or finance/apps/ (analyze + visualize)
```

## Conventions

- `.pkl` files are the primary cache format — stored in `finance/_data/` (git-ignored)
- Scripts with `# %%` are designed for interactive execution, not pytest
- `chart_styles.py` is the single source of truth for indicator colors/styles across all apps
- Windows-focused (PowerShell admin scripts, PyQt6 desktop UI)
- Multi-source fallback: IBKR live → DoltSQL remote → offline pickle cache

## Build & Run

```bash
uv sync                              # Install/sync dependencies
python -m finance.apps               # List all available apps
python -m finance.apps swing-plot    # Launch swing trading dashboard
python -m finance.apps momentum      # Launch momentum dashboard
```

## Testing

- `pytest` for Python code (TDD: write tests first)
- Interactive scripts (`# %%`) are validated manually in IPython/Spyder
