"""
currency_momentum_bt.py
=======================
Weekly cross-sectional currency momentum backtest.

Based on Menkhoff, Sarno, Schmeling & Schrimpf (2012) "Currency Momentum
Strategies", Journal of Financial Economics 106.

Universe
--------
6 pairs, all expressed as appreciation of the non-USD currency vs USD:
  EURUSD, GBPUSD, AUDUSD, CHFUSD — used directly
  USDJPY, USDCAD — sign-inverted so positive return = non-USD appreciation

Methodology
-----------
1. Weekly close = last bar on Friday at or before 21:00 UTC (NY session end).
2. Momentum signal = log return over [t-52w, t-4w] (skip 1 month to avoid
   short-term reversal contamination — Menkhoff et al Appendix A).
3. Rank all 6 non-USD currencies each week.
4. Long top 3, short bottom 3.
5. Portfolio return = equal-weight mean(long returns) - mean(short returns).
6. Transaction cost: SPREAD_COST_PIPS per pair that changes position.

Run from repo root:
    python finance/intraday_pm/forex/currency_momentum_bt.py
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np
import pandas as pd

from finance.utils.intraday import get_bars
from finance.intraday_pm.backtests.hougaard_dax import _fmt, _fmt_plain

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCHEMA = "forex"
START  = datetime(2013, 1, 1, tzinfo=timezone.utc)
END    = datetime(2026, 4, 1, tzinfo=timezone.utc)

# Pairs where positive return = non-USD appreciation (used directly)
DIRECT_PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "CHFUSD"]

# Pairs where positive return = USD appreciation; inverted to get non-USD vs USD
INVERTED_PAIRS = ["USDJPY", "USDCAD"]

# pip_size for transaction cost conversion (major pairs only)
PIP_SIZE_DIRECT   = 0.0001
PIP_SIZE_INVERTED = 0.0001  # USDJPY cost charged in USD, not JPY — use approx

SPREAD_COST_PIPS = 1.5   # round-trip cost per pair when position changes

MOMENTUM_LOOKBACK_WEEKS = 52  # total lookback window
MOMENTUM_SKIP_WEEKS     = 4   # skip most recent N weeks (reversal buffer)
N_LONG  = 3  # number of pairs to hold long
N_SHORT = 3  # number of pairs to hold short

RESULTS_APPEND_PATH = "finance/intraday_pm/forex/RESULTS.md"
WEEKLY_CLOSE_HOUR   = 21  # hour at-or-before which we take the weekly close (UTC)


# ---------------------------------------------------------------------------
# Public helpers (imported by tests)
# ---------------------------------------------------------------------------
def _invert_returns(returns: pd.Series) -> pd.Series:
    """Flip sign so that a rising USD-base pair means non-USD appreciation."""
    return -returns


def _momentum_score(
    weekly_prices: pd.Series,
    lookback: int = MOMENTUM_LOOKBACK_WEEKS,
    skip: int = MOMENTUM_SKIP_WEEKS,
) -> pd.Series:
    """
    Compute the momentum signal at each weekly bar.

    Signal at bar t = log(price[t - skip] / price[t - lookback])

    No look-ahead: signal at bar t uses only data up to t-skip.
    NaN is returned until at least `lookback` bars of history are available.
    """
    log_p = np.log(weekly_prices)
    return log_p.shift(skip) - log_p.shift(lookback)


def _rank_long_short(
    scores: pd.Series,
    n: int = N_LONG,
) -> tuple[list[str], list[str]]:
    """
    Return (long_list, short_list) of length n each.
    Ties are broken alphabetically to ensure determinism.
    """
    sorted_scores = scores.sort_values(ascending=False, kind="stable").sort_index(
        key=lambda idx: scores[idx],
        ascending=False,
    )
    # Sort descending by score, then ascending by name for ties
    ranked = scores.sort_values(ascending=False).pipe(
        lambda s: s.iloc[np.lexsort((s.index.map(list(scores.index).index), -s.values))]
        if len(s) > 0 else s
    )
    # Stable: sort by score desc, break ties alphabetically
    df = pd.DataFrame({"score": scores})
    df = df.sort_values("score", ascending=False, kind="stable")
    df["tiebreak"] = df.index  # alphabetical
    df = df.sort_values(["score", "tiebreak"], ascending=[False, True], kind="stable")

    long_list  = df.index[:n].tolist()
    short_list = df.index[-n:].tolist()
    return long_list, short_list


def _rebalance_cost(
    prev_positions: dict[str, int],
    new_positions: dict[str, int],
    cost_per_trade: float = SPREAD_COST_PIPS,
) -> float:
    """
    Total transaction cost in pips for moving from prev_positions to new_positions.

    A "trade" is any pair whose position changes (new entry, exit, or direction flip).
    Each such change costs cost_per_trade pips.
    """
    all_pairs = set(prev_positions) | set(new_positions)
    n_changes = sum(
        1 for p in all_pairs
        if prev_positions.get(p, 0) != new_positions.get(p, 0)
    )
    return n_changes * cost_per_trade


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_weekly_closes() -> pd.DataFrame:
    """
    Load weekly Friday closes for all pairs.
    Returns DataFrame with one column per non-USD currency (returns already sign-adjusted).
    Columns: EURUSD, GBPUSD, AUDUSD, CHFUSD, inv_USDJPY, inv_USDCAD
    """
    weekly: dict[str, pd.Series] = {}

    for pair in DIRECT_PAIRS + INVERTED_PAIRS:
        df = get_bars(pair, SCHEMA, START, END, period="1W")
        if df.empty:
            raise RuntimeError(f"No weekly data for {pair}")
        # Weekly resample ends on Sunday by default; we want Friday close.
        # Use 1-min data resampled to weekly with week ending Friday.
        df1m = get_bars(pair, SCHEMA, START, END, period=None)
        if df1m.empty:
            raise RuntimeError(f"No 1-min data for {pair}")

        # Weekly close: last bar on or before Friday 21:00 UTC
        # Resample to W-FRI (week ending Friday)
        wk = (
            df1m["c"]
            .resample("W-FRI")
            .last()
            .dropna()
        )
        label = f"inv_{pair}" if pair in INVERTED_PAIRS else pair
        weekly[label] = wk

    closes = pd.DataFrame(weekly).sort_index()

    # Invert USD-base pairs so positive return = non-USD appreciation
    for pair in INVERTED_PAIRS:
        col = f"inv_{pair}"
        closes[col] = 1.0 / closes[col]

    return closes


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
def _run_backtest(closes: pd.DataFrame) -> pd.DataFrame:
    """
    Run the weekly momentum strategy.

    Returns a DataFrame with columns:
        week_end, portfolio_return_gross, cost_pips, portfolio_return_net,
        long_pairs, short_pairs
    """
    pairs = list(closes.columns)
    log_prices = np.log(closes)

    # Momentum signal per pair per week
    signals: dict[str, pd.Series] = {
        p: _momentum_score(closes[p]) for p in pairs
    }
    signal_df = pd.DataFrame(signals)

    results: list[dict] = []
    prev_positions: dict[str, int] = {}

    for i in range(len(closes) - 1):
        week_end      = closes.index[i]
        next_week_end = closes.index[i + 1]

        row_signals = signal_df.iloc[i]
        if row_signals.isna().all():
            continue
        if row_signals.isna().any():
            # Fill NaN with 0 (neutral) to allow partial ranking
            row_signals = row_signals.fillna(0.0)

        long_pairs, short_pairs = _rank_long_short(row_signals, n=N_LONG)

        # Build new positions: +1 long, -1 short
        new_positions = {p: 1 for p in long_pairs} | {p: -1 for p in short_pairs}

        # Transaction cost for this rebalance (in pips)
        cost_pips = _rebalance_cost(prev_positions, new_positions)

        # Next week return for each pair
        next_returns = (log_prices.iloc[i + 1] - log_prices.iloc[i])

        long_ret  = next_returns[long_pairs].mean()  if long_pairs  else 0.0
        short_ret = next_returns[short_pairs].mean() if short_pairs else 0.0
        gross_ret = long_ret - short_ret

        # Convert cost from pips to approximate log-return equivalent
        # Using average pip_size of 0.0001 for a rough scaling
        cost_return = cost_pips * PIP_SIZE_DIRECT

        results.append({
            "week_end":              week_end,
            "portfolio_return_gross": gross_ret,
            "cost_pips":              cost_pips,
            "cost_return":            cost_return,
            "portfolio_return_net":   gross_ret - cost_return,
            "long_pairs":             ",".join(long_pairs),
            "short_pairs":            ",".join(short_pairs),
        })

        prev_positions = new_positions

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Section builder
# ---------------------------------------------------------------------------
def _build_section(results_df: pd.DataFrame, closes: pd.DataFrame) -> str:
    if results_df.empty:
        return "\n---\n\n## Currency Momentum\n\n_No results_\n"

    valid = results_df.dropna(subset=["portfolio_return_net"])
    n_weeks      = len(valid)
    gross_weekly = valid["portfolio_return_gross"]
    net_weekly   = valid["portfolio_return_net"]

    ann_gross_ret = gross_weekly.mean() * 52
    ann_net_ret   = net_weekly.mean() * 52
    ann_gross_vol = gross_weekly.std() * np.sqrt(52)
    ann_net_vol   = net_weekly.std() * np.sqrt(52)

    sharpe_gross = (gross_weekly.mean() / gross_weekly.std()) * np.sqrt(52) if gross_weekly.std() > 0 else 0.0
    sharpe_net   = (net_weekly.mean()   / net_weekly.std())   * np.sqrt(52) if net_weekly.std()   > 0 else 0.0
    win_rate     = (net_weekly > 0).mean() * 100
    avg_cost     = valid["cost_pips"].mean()

    cumulative_gross = (1 + gross_weekly).cumprod()
    cumulative_net   = (1 + net_weekly).cumprod()
    total_gross_ret  = cumulative_gross.iloc[-1] - 1
    total_net_ret    = cumulative_net.iloc[-1] - 1

    max_dd_gross = (cumulative_gross / cumulative_gross.cummax() - 1).min()
    max_dd_net   = (cumulative_net   / cumulative_net.cummax()   - 1).min()

    pairs_list = ", ".join(closes.columns.tolist())

    lines = [
        "",
        "---",
        "",
        "## Currency Momentum (BT-FX-2)",
        "",
        f"Generated: {date.today().isoformat()}  ",
        f"Universe: {pairs_list}  ",
        f"Period: {valid['week_end'].iloc[0].date().isoformat()} -> {valid['week_end'].iloc[-1].date().isoformat()}  ",
        f"Lookback: {MOMENTUM_LOOKBACK_WEEKS}w − {MOMENTUM_SKIP_WEEKS}w skip  ",
        f"Portfolio: long top {N_LONG}, short bottom {N_SHORT} pairs weekly  ",
        f"Cost: {SPREAD_COST_PIPS} pips/trade when position changes  ",
        "",
        "### Summary statistics",
        "",
        "| Metric | Gross | Net (after costs) |",
        "|--------|-------|-------------------|",
        f"| Weeks tested | {n_weeks} | {n_weeks} |",
        f"| Annualised return | {ann_gross_ret:+.1%} | {ann_net_ret:+.1%} |",
        f"| Annualised volatility | {ann_gross_vol:.1%} | {ann_net_vol:.1%} |",
        f"| Sharpe ratio | {sharpe_gross:+.3f} | {sharpe_net:+.3f} |",
        f"| Total return | {total_gross_ret:+.1%} | {total_net_ret:+.1%} |",
        f"| Max drawdown | {max_dd_gross:.1%} | {max_dd_net:.1%} |",
        f"| Win rate (weekly) | {(gross_weekly > 0).mean()*100:.1f}% | {win_rate:.1f}% |",
        f"| Avg rebalance cost | — | {avg_cost:.1f} pips/week |",
        "",
        "### Pair frequency in long / short portfolios",
        "",
        "| Pair | Long weeks | Short weeks | Long% |",
        "|------|-----------|-------------|-------|",
    ]

    for col in closes.columns:
        long_count  = valid["long_pairs"].str.contains(col, regex=False).sum()
        short_count = valid["short_pairs"].str.contains(col, regex=False).sum()
        total_count = long_count + short_count
        long_pct    = long_count / total_count * 100 if total_count > 0 else 0.0
        lines.append(f"| {col} | {long_count} | {short_count} | {long_pct:.0f}% |")

    lines += [
        "",
        "> EV is measured in log-return space. Costs converted from pips using",
        f"> pip_size = {PIP_SIZE_DIRECT} (EURUSD-equivalent). Sharpe is annualised",
        "> assuming 52 weekly observations per year, zero risk-free rate.",
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading weekly closes...")
    closes = _load_weekly_closes()
    print(f"  {len(closes)} weeks × {len(closes.columns)} pairs: {list(closes.columns)}")
    print(f"  Period: {closes.index[0].date()} -> {closes.index[-1].date()}")

    print("Running momentum backtest...")
    results_df = _run_backtest(closes)
    valid = results_df.dropna(subset=["portfolio_return_net"])
    print(f"  {len(valid)} valid weeks after signal warm-up")

    if not valid.empty:
        net = valid["portfolio_return_net"]
        sharpe = (net.mean() / net.std()) * np.sqrt(52) if net.std() > 0 else 0.0
        ann_ret = net.mean() * 52
        print(f"  Net annualised return: {ann_ret:+.1%}")
        print(f"  Net annualised Sharpe: {sharpe:+.3f}")

    section = _build_section(results_df, closes)
    with open(RESULTS_APPEND_PATH, "a", encoding="utf-8") as fh:
        fh.write(section)
    print(f"\nAppended -> {RESULTS_APPEND_PATH}")


if __name__ == "__main__":
    main()
