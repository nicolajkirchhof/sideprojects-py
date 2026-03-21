# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Personal quantitative finance research platform focused on swing trading analysis, options tracking, and interactive dashboard development. Uses Python 3.12+ with heavy reliance on PyQtGraph/PyQt6 for interactive visualizations and IBKR for live market data.

## Environment Setup

Uses `uv` as the package manager:

```bash
uv sync                  # Install/sync dependencies from uv.lock
uv pip install -e .      # Install in editable mode
```

Legacy conda approach (see `admin/install-environment.ps1` for PowerShell setup with conda env `ds311`).

## Running Key Applications

```bash
# List all available apps
python -m finance.apps

# Interactive swing trading visualization dashboard
python -m finance.apps swing-plot

# Momentum/earnings analysis Qt dashboard
python -m finance.apps momentum
python -m finance.apps momentum --start-year 2022

# Convert IBKR CSV exports to fast-loading PKL format
python IBKR_CSV_to_PKL_Converter.py --root finance/_data/ibkr [--force] [--delete-csv]

# Validate Qt/pyqtgraph environment
python SmokeTestWindow.py
```

## Code Architecture

### Data Flow

```
IBKR API / Yahoo Finance / DoltSQL
    ↓
finance/utils/ibkr.py, dolt_data.py, yf/
    ↓  (cached as .pkl in finance/_data/)
finance/utils/swing_trading_data.py  →  SwingTradingData class
    ↓
finance/utils/indicators.py          →  Technical indicator computation
    ↓
finance/swing_pm/                    →  Analysis & strategy logic
    ↓
finance/apps/                        →  All interactive applications
├── swing_plot/                      →  Swing trading dashboard (PyQtGraph)
└── momentum_dashboard/              →  Momentum/earnings dashboard (PySide6)
```

### Key Modules

- **`finance/utils/`** — Core infrastructure shared across all analyses:
  - `swing_trading_data.py` — `SwingTradingData` class: primary data container; loads from IBKR → DoltSQL → offline cache fallback chain
  - `indicators.py` — Vectorized technical indicators (MA, ATR, VWAP, TTM Squeeze, Hurst exponent, slopes)
  - `chart_styles.py` — Framework-agnostic indicator color/style definitions (single source of truth for all apps)
  - `momentum_data.py` — Data loading/preparation for momentum/earnings analysis
  - `plots.py` — Matplotlib/mplfinance plotting helpers
  - `ibkr.py` — IBKR API wrapper; `dolt_data.py` — DoltSQL integration; `tradelog_client.py` — REST client for trade log
  - `options.py`, `greeks.py` — Black-Scholes and options Greeks calculations

- **`finance/apps/`** — All interactive applications, launched via `python -m finance.apps <name>`:
  - `_qt_bootstrap.py` — Shared Qt infrastructure (singleton app, dark palette, IPython integration)
  - `swing_plot/` — PyQtGraph multi-tab dashboard (price/candlestick, volatility, drawdowns, probability trees)
  - `momentum_dashboard/` — PySide6/PyQtGraph momentum & earnings analysis dashboard

- **`finance/swing_pm/`** — Strategy analysis scripts. `underlying_stats.py` uses `# %%` cell markers (runs as Jupyter-style notebook in IPython/Spyder). Contains streak probability, MA slope statistics, and trend duration analysis.

- **`finance/ibkr/`** — IBKR data collection: historical data download, options data, GEX calculations, market scanner.

- **`finance/intraday_pm/backtester/`** — Backtesting framework for bracket strategies, stock strategies, and options (CSV input data, strategy evaluation scripts).

- **`tradelog/backend.net/`** — C# .NET Core REST API with SQLite (`tradelog.db`) for centralized trade recording. Python code connects via `finance/utils/tradelog_client.py`.

### Data Storage Patterns

- `.pkl` (pickle) files are the primary cache format for speed; stored in `finance/_data/` (git-ignored)
- `_data/`, `_analysis/`, `_backup/` directories are git-ignored (contain large cached datasets)
- Multi-source fallback: IBKR live → DoltSQL remote → offline pickle cache

### Development Style

- Scripts with `# %%` cell markers are designed for interactive execution in IPython/Spyder (not pytest-based)
- No formal test framework; `finance/ibkr/test.py` is a Jupyter-style connection test notebook
- Windows-focused (PowerShell admin scripts, .NET backend, PyQt6 desktop UI)