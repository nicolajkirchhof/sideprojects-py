"""
finance.apps.assistant._export
================================
Ticker export functions for TA-E6-S1 (Barchart) and TA-E6-S2 (TWS).

Both functions are Qt-free and accept an optional base_dir for testability.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

_DEFAULT_EXPORT_DIR = Path("finance/_data/assistant")


def export_barchart(
    symbols: list[str],
    trade_date: date,
    *,
    base_dir: Path | None = None,
) -> Path:
    """Write a comma-separated ticker list to a .txt file.

    File: <base_dir>/watchlist-YYYY-MM-DD.txt

    Returns the path that was written.
    """
    base = Path(base_dir) if base_dir is not None else _DEFAULT_EXPORT_DIR
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"watchlist-{trade_date.isoformat()}.txt"
    path.write_text(",".join(s.upper() for s in symbols), encoding="utf-8")
    return path


def export_tws(
    symbols: list[str],
    trade_date: date,
    *,
    base_dir: Path | None = None,
) -> Path:
    """Write a TWS-importable CSV file.

    Each line: DES,SYMBOL,STK,SMART,,,,
    Symbols are uppercased per TWS requirements.

    File: <base_dir>/tws-watchlist-YYYY-MM-DD.csv

    Returns the path that was written.
    """
    base = Path(base_dir) if base_dir is not None else _DEFAULT_EXPORT_DIR
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"tws-watchlist-{trade_date.isoformat()}.csv"
    lines = [f"DES,{s.upper()},STK,SMART,,,," for s in symbols]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
