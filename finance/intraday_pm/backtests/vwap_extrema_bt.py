"""
vwap_extrema_bt.py
==================
BT-6-S2: VWAP Extrema — Filtered Bracket Backtest.

Data source: consolidated Parquet files at finance/_data/intraday/vwap_extrema/{SYMBOL}.parquet.
Built from PKL files (swing_vwap_ad/) + gap rows generated from 5-min Parquet intraday data.
Run build_vwap_extrema_parquet.py first to populate the Parquet store.

Strategy:
  - At each 5-min bar, an OCO bracket is placed around the candle.
  - `bracket_entry_in_trend = True` when the bracket triggers in the VWAP-trend
    direction (candle_sentiment × in_trend == bracket_trigger).
  - Win: bracket triggers in trend direction → gain = pts_move (remaining trend pts).
  - Loss: bracket triggers against trend → loss = candle_atr_pts (bar ATR as SL proxy).
  - Cost: 2 pts round-trip spread per trade.

BT-6-S2 analysis:
  1. Per-time-slot success_rate and EV across all symbols.
  2. Filter to "high-edge" slots (success_rate ≥ 60 % and EV > 2 pts after costs).
  3. Report per-symbol metrics for filtered vs unfiltered brackets.

Run from repo root:
    python finance/intraday_pm/backtests/vwap_extrema_bt.py
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
PARQUET_DIR = Path("finance/_data/intraday/vwap_extrema")
SYMBOLS = [
    "IBDE40", "IBGB100", "IBUS500", "IBUST100",
    "IBEU50", "IBFR40", "IBCH20", "IBNL25",
    "IBUS30", "IBAU200", "IBJP225", "IBES35", "USGOLD",
]

START_DATE = "2020-01-01"
END_DATE = date.today().isoformat()

SPREAD_COST_PTS = 2.0
SUCCESS_RATE_THRESHOLD = 0.60   # minimum to qualify as high-edge slot
EV_THRESHOLD_PTS = 2.0          # minimum EV after costs

RESULTS_APPEND_PATH = "finance/intraday_pm/RESULTS.md"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_bracket_moves(symbol: str) -> pd.DataFrame:
    """
    Load bracket moves from the consolidated Parquet file for a symbol,
    filtered to the backtest date range.
    """
    parquet_path = PARQUET_DIR / f"{symbol}.parquet"
    if not parquet_path.exists():
        print(f"  Parquet not found: {parquet_path} — run build_vwap_extrema_parquet.py first")
        return pd.DataFrame()

    df = pd.read_parquet(parquet_path)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df = df[(df["trade_date"] >= date.fromisoformat(START_DATE)) &
            (df["trade_date"] < date.fromisoformat(END_DATE))]

    if df.empty:
        return pd.DataFrame()

    df = df.set_index("ts")
    df.index = pd.to_datetime(df.index)
    df["time"] = df.index.time
    return df


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------
def _slot_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-time-slot metrics for bracket entries.
    Only considers rows where bracket was triggered (bracket_trigger != 0).
    """
    triggered = df[df["bracket_trigger"] != 0].copy()
    if triggered.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for time_slot, group in triggered.groupby("time"):
        n = len(group)
        wins = group[group["bracket_entry_in_trend"]]
        losses = group[~group["bracket_entry_in_trend"]]

        p_win = len(wins) / n
        avg_win = wins["pts_move"].mean() if not wins.empty else 0.0
        avg_loss = losses["candle_atr_pts"].mean() if not losses.empty else 0.0

        ev = p_win * avg_win - (1 - p_win) * avg_loss - SPREAD_COST_PTS

        rows.append({
            "time": time_slot,
            "n_trades": n,
            "success_rate": p_win,
            "avg_win_pts": avg_win,
            "avg_loss_pts": avg_loss,
            "ev_pts": ev,
        })

    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


def _overall_metrics(df: pd.DataFrame, label: str, cost_pts: float = SPREAD_COST_PTS) -> dict:
    triggered = df[df["bracket_trigger"] != 0]
    if triggered.empty:
        return {}
    n = len(triggered)
    wins = triggered[triggered["bracket_entry_in_trend"]]
    losses = triggered[~triggered["bracket_entry_in_trend"]]
    p_win = len(wins) / n
    avg_win = wins["pts_move"].mean() if not wins.empty else 0.0
    avg_loss = losses["candle_atr_pts"].mean() if not losses.empty else 0.0
    ev = p_win * avg_win - (1 - p_win) * avg_loss - cost_pts
    std = triggered.apply(
        lambda r: r["pts_move"] - cost_pts if r["bracket_entry_in_trend"]
        else -r["candle_atr_pts"] - cost_pts, axis=1
    ).std()
    sharpe = ev / std if std > 0 else 0.0
    return {
        "label": label,
        "n_trades": n,
        "success_rate_pct": p_win * 100,
        "avg_win_pts": avg_win,
        "avg_loss_pts": avg_loss,
        "ev_pts": ev,
        "sharpe": sharpe,
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
def build_section(symbol_results: dict[str, tuple[pd.DataFrame, pd.DataFrame, list[str]]]) -> str:
    """
    symbol_results: {symbol: (df_all_triggered, df_filtered_triggered, high_edge_slots)}
    """
    lines: list[str] = [
        "",
        "---",
        "",
        "## VWAP Extrema — Filtered Bracket (BT-6-S2)",
        "",
        f"Generated: {date.today().isoformat()}  ",
        f"Symbols: {', '.join(SYMBOLS)}  ",
        f"Period: {START_DATE} -> {END_DATE}  ",
        "Data: consolidated Parquet (PKL + Parquet gap rows)  ",
        "Win: bracket triggers in VWAP-trend direction → pts_move gained  ",
        "Loss: bracket triggers counter-trend → candle ATR lost  ",
        "Cost: 2 pts round-trip spread  ",
        f"High-edge filter: success_rate >= {SUCCESS_RATE_THRESHOLD*100:.0f}% AND EV > {EV_THRESHOLD_PTS} pts  ",
        "",
        "### Per-symbol: Unfiltered vs Filtered Bracket",
        "",
        "| Symbol | Group | N trades | Success% | Avg win (pts) | Avg loss (pts) | EV (pts) | Sharpe | Go/No-go |",
        "|--------|-------|----------|----------|---------------|----------------|----------|--------|----------|",
    ]

    for symbol, (df_all, df_filt, _) in symbol_results.items():
        for label, df in [("All slots", df_all), ("High-edge slots", df_filt)]:
            m = _overall_metrics(df, label)
            if not m:
                continue
            verdict = "Go" if m["ev_pts"] > 0 else "No-go"
            lines.append(
                f"| {symbol} | {label} | {m['n_trades']} | {_fmt_plain(m['success_rate_pct'])}% "
                f"| {_fmt(m['avg_win_pts'])} | {_fmt(m['avg_loss_pts'])} "
                f"| {_fmt(m['ev_pts'])} | {_fmt(m['sharpe'], 3)} | {verdict} |"
            )

    lines += [""]

    # High-edge slots per symbol
    lines += ["### High-Edge Time Slots (≥ 60% success rate, EV > 2 pts)", ""]
    for symbol, (_, _, he_slots) in symbol_results.items():
        slot_str = ", ".join(str(s) for s in he_slots[:10]) if he_slots else "none"
        lines.append(f"**{symbol}:** {slot_str}  ")
    lines.append("")

    # Slot breakdown for first symbol as illustration
    first_sym = SYMBOLS[0]
    if first_sym in symbol_results:
        df_all_trig = symbol_results[first_sym][0]
        slot_df = _slot_metrics(df_all_trig)
        if not slot_df.empty:
            he = slot_df[
                (slot_df["success_rate"] >= SUCCESS_RATE_THRESHOLD) &
                (slot_df["ev_pts"] >= EV_THRESHOLD_PTS)
            ]
            lines += [
                f"### {first_sym} — High-Edge Slots Detail",
                "",
                "| Time (local) | N | Success% | Avg win | Avg loss | EV (pts) |",
                "|--------------|---|----------|---------|----------|----------|",
            ]
            for _, row in he.iterrows():
                lines.append(
                    f"| {row['time']} | {row['n_trades']} | {_fmt_plain(row['success_rate']*100)}% "
                    f"| {_fmt(row['avg_win_pts'])} | {_fmt(row['avg_loss_pts'])} | {_fmt(row['ev_pts'])} |"
                )
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    symbol_results: dict[str, tuple[pd.DataFrame, pd.DataFrame, list]] = {}

    for symbol in SYMBOLS:
        print(f"\n{symbol}")
        df = _load_bracket_moves(symbol)
        if df.empty:
            print("  No data — skipping")
            continue
        triggered = df[df["bracket_trigger"] != 0]
        print(f"  {len(df):,} bars | {len(triggered):,} triggered brackets")

        # Per-slot analysis
        slot_df = _slot_metrics(df)
        high_edge = slot_df[
            (slot_df["success_rate"] >= SUCCESS_RATE_THRESHOLD) &
            (slot_df["ev_pts"] >= EV_THRESHOLD_PTS)
        ]
        he_times = set(high_edge["time"].tolist())
        print(f"  High-edge slots: {len(he_times)} (success >= {SUCCESS_RATE_THRESHOLD*100:.0f}%, EV > {EV_THRESHOLD_PTS} pts)")

        # Filter original data to high-edge slots
        df_filt = df[df["time"].isin(he_times)]
        symbol_results[symbol] = (df, df_filt, sorted(he_times))

        m_all = _overall_metrics(triggered, "All")
        m_filt = _overall_metrics(df_filt[df_filt["bracket_trigger"] != 0], "Filtered")
        if m_all:
            print(f"  All EV: {m_all['ev_pts']:+.2f} pts ({m_all['success_rate_pct']:.1f}% success)")
        if m_filt:
            print(f"  Filtered EV: {m_filt['ev_pts']:+.2f} pts ({m_filt['success_rate_pct']:.1f}% success, {m_filt['n_trades']} trades)")

    section = build_section(symbol_results)
    with open(RESULTS_APPEND_PATH, "a", encoding="utf-8") as fh:
        fh.write(section)
    print(f"\nAppended -> {RESULTS_APPEND_PATH}")


if __name__ == "__main__":
    main()
