"""
noon_butterfly_bt.py
====================
BT-6-S3: Noon Iron Butterfly — DAX / ESTX50 / SPX.

Data source: consolidated Parquet files at finance/_data/intraday/noon_butterfly/{SYM}.parquet.
Built from PKL results files + gap rows using HV20 proxy for IV (flagged iv_source='hv_proxy').
Run build_noon_butterfly_parquet.py first to populate the Parquet store.

Each row = one trading day. The iron butterfly is sold at noon (local time),
wings set at ±10% ATM for DAX, ±5% for ESTX50/SPX; closed at session close.
`pnl` column = realised P&L in index points.

BT-6-S3 analysis:
  1. Overall metrics: win rate, avg P&L, Sharpe, max loss.
  2. IV filter: segment by noon_iv percentile; higher IV = richer premium.
  3. Verdict: Go if avg_pnl > 0 AND Sharpe > 0.1 AND max_loss acceptable.

Run from repo root:
    python finance/intraday_pm/backtests/noon_butterfly_bt.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PARQUET_DIR = Path("finance/_data/intraday/noon_butterfly")

# PKL symbol name → Parquet file stem
INSTRUMENTS = {
    "DAX":    "DAX",
    "ESTX50": "ESTX50",
    "SPX":    "SPX",
}

START_DATE = "2022-01-01"
END_DATE = date.today().isoformat()

RESULTS_APPEND_PATH = "finance/intraday_pm/RESULTS.md"

IV_LABELS = ["low_iv (bot 33%)", "normal_iv (33-67%)", "high_iv (top 33%)"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load(sym: str, start: str, end: str) -> pd.DataFrame:
    parquet_path = PARQUET_DIR / f"{sym}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)
    df = pd.read_parquet(parquet_path)
    df["date_str"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
    df = df[(df["date_str"] >= start) & (df["date_str"] < end)].copy()
    df = df.reset_index(drop=True)
    return df


def _add_iv_bucket(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    p33 = df["noon_iv"].quantile(0.33)
    p67 = df["noon_iv"].quantile(0.67)
    def _cat(v: float) -> str:
        if v <= p33:
            return IV_LABELS[0]
        if v >= p67:
            return IV_LABELS[2]
        return IV_LABELS[1]
    df["iv_cat"] = df["noon_iv"].apply(_cat)
    return df


def _metrics(pnl: pd.Series, label: str) -> dict:
    n = len(pnl)
    if n == 0:
        return {}
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    win_rate = len(wins) / n
    avg_win = wins.mean() if not wins.empty else 0.0
    avg_loss = losses.mean() if not losses.empty else 0.0
    expectancy = pnl.mean()
    std = pnl.std()
    sharpe = expectancy / std if std > 0 else 0.0
    max_loss = pnl.min()
    return {
        "label": label,
        "n": n,
        "win_rate_pct": win_rate * 100,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "sharpe": sharpe,
        "max_loss": max_loss,
    }


def _fmt(val: float, decimals: int = 2) -> str:
    if pd.isna(val):
        return "—"
    return f"{val:+.{decimals}f}"


def _fmt_plain(val: float, decimals: int = 1) -> str:
    if pd.isna(val):
        return "—"
    return f"{val:.{decimals}f}"


# ---------------------------------------------------------------------------
# Section builder
# ---------------------------------------------------------------------------
def build_section(results: dict[str, pd.DataFrame]) -> str:
    lines: list[str] = [
        "",
        "---",
        "",
        "## Noon Iron Butterfly — DAX / ESTX50 / SPX (BT-6-S3)",
        "",
        f"Generated: {date.today().isoformat()}  ",
        f"Period: {START_DATE} -> {END_DATE}  ",
        "Data: consolidated Parquet (PKL iv_source=pkl + HV20 gap rows iv_source=hv_proxy)  ",
        "Entry: noon local time; wings = 10% ATM (DAX) / 5% ATM (ESTX50/SPX)  ",
        "Exit: session close  ",
        "P&L units: index points (not normalised to lot size)  ",
        "",
        "### Overall Results",
        "",
        "| Instrument | N days | Win% | Avg win | Avg loss | Expectancy | Sharpe | Max loss | Go/No-go |",
        "|------------|--------|------|---------|----------|------------|--------|----------|----------|",
    ]

    for instr, df in results.items():
        m = _metrics(df["pnl"], instr)
        if not m:
            continue
        verdict = "Go" if m["expectancy"] > 0 and m["sharpe"] > 0.1 else "No-go"
        lines.append(
            f"| {instr} | {m['n']} | {_fmt_plain(m['win_rate_pct'])}% "
            f"| {_fmt(m['avg_win'])} | {_fmt(m['avg_loss'])} "
            f"| {_fmt(m['expectancy'])} | {_fmt(m['sharpe'], 3)} "
            f"| {_fmt(m['max_loss'])} | {verdict} |"
        )

    lines += [""]

    # IV segmentation
    lines += [
        "### IV-Filtered Results (noon implied vol percentile)",
        "",
        "| Instrument | IV bucket | N | Win% | Expectancy | Sharpe |",
        "|------------|-----------|---|------|------------|--------|",
    ]
    for instr, df in results.items():
        df_iv = _add_iv_bucket(df)
        for iv_cat in IV_LABELS:
            sub = df_iv[df_iv["iv_cat"] == iv_cat]["pnl"]
            m = _metrics(sub, iv_cat)
            if not m:
                continue
            lines.append(
                f"| {instr} | {iv_cat} | {m['n']} | {_fmt_plain(m['win_rate_pct'])}% "
                f"| {_fmt(m['expectancy'])} | {_fmt(m['sharpe'], 3)} |"
            )

    lines += [""]

    # Tail risk profile
    lines += [
        "### Tail Risk Profile",
        "",
        "| Instrument | P5 (pts) | P25 (pts) | Median (pts) | P75 (pts) | P95 (pts) |",
        "|------------|----------|-----------|--------------|-----------|-----------|",
    ]
    for instr, df in results.items():
        pnl = df["pnl"]
        lines.append(
            f"| {instr} | {_fmt(pnl.quantile(0.05))} | {_fmt(pnl.quantile(0.25))} "
            f"| {_fmt(pnl.quantile(0.50))} | {_fmt(pnl.quantile(0.75))} "
            f"| {_fmt(pnl.quantile(0.95))} |"
        )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    results: dict[str, pd.DataFrame] = {}

    for instr, sym in INSTRUMENTS.items():
        print(f"{instr}...", end=" ")
        try:
            df = _load(sym, START_DATE, END_DATE)
        except FileNotFoundError:
            print("Parquet not found — run build_noon_butterfly_parquet.py first")
            continue
        n = len(df)
        avg = df["pnl"].mean()
        wr = (df["pnl"] > 0).mean() * 100
        print(f"{n} days, win={wr:.1f}%, avg={avg:+.2f} pts")
        results[instr] = df

    section = build_section(results)
    with open(RESULTS_APPEND_PATH, "a", encoding="utf-8") as fh:
        fh.write(section)
    print(f"\nAppended -> {RESULTS_APPEND_PATH}")


if __name__ == "__main__":
    main()
