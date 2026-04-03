"""
Core PM Backtest Runner

Interactive script (# %% cells) for running VRP analysis, range tests,
and trade simulations on GLD, SLV, TLT, USO.

Usage:
    python -m finance.core_pm.run_backtest
    or run cells interactively in IPython/Spyder
"""

# %%
from finance.utils.swing_trading_data import SwingTradingData
from finance.core_pm.backtest import (
    compute_vrp, compute_range_test, simulate_trades,
    trades_to_dataframe, BacktestConfig,
)
from finance.core_pm.analysis import (
    vrp_summary, range_test_summary, trade_summary,
    print_vrp_report, print_range_report, print_trade_report,
)
import pandas as pd

SYMBOLS = ['GLD', 'SLV', 'TLT', 'USO']

# %% ── Load Data ─────────────────────────────────────────

print("Loading data...")
data = {}
for sym in SYMBOLS:
    std = SwingTradingData(sym, datasource='offline')
    if std.empty:
        print(f"  {sym}: NO DATA — skipping")
        continue
    df = std.df_day.dropna(subset=['ivc']).copy()
    df.attrs['symbol'] = sym
    data[sym] = df
    print(f"  {sym}: {len(df)} rows with IV, {df.index.min().date()} to {df.index.max().date()}")

# %% ── Phase 1: VRP Analysis ────────────────────────────

print("\n\nPhase 1: Computing VRP...")
all_vrp_summaries = []

for sym, df in data.items():
    vrp_df = compute_vrp(df, windows=[30, 45, 60])
    summary = vrp_summary(vrp_df, sym)
    all_vrp_summaries.append(summary)

vrp_results = pd.concat(all_vrp_summaries, ignore_index=True)
print_vrp_report(vrp_results)

# %% ── Phase 2: Range Test ──────────────────────────────

print("\n\nPhase 2: Computing range tests...")
all_range_summaries = []

for sym, df in data.items():
    range_df = compute_range_test(df, dte=45)
    summary = range_test_summary(range_df, sym)
    all_range_summaries.append(summary)

range_results = pd.concat(all_range_summaries, ignore_index=True)
print_range_report(range_results)

# %% ── Phase 3: Trade Simulation — Short Put ────────────

print("\n\nPhase 3: Simulating trades...")
all_trade_summaries = []

# Test configurations
configs = [
    # Short put — unfiltered
    BacktestConfig(structure='short_put', dte=45, entry_interval_days=14,
                   ivp_filter=None, sma_filter=False),
    # Short put — IVP > 50 filter
    BacktestConfig(structure='short_put', dte=45, entry_interval_days=14,
                   ivp_filter=50.0, sma_filter=False),
    # Strangle — unfiltered
    BacktestConfig(structure='strangle', dte=45, entry_interval_days=14,
                   ivp_filter=None, sma_filter=False, call_delta=0.25),
    # Strangle — IVP > 50 filter
    BacktestConfig(structure='strangle', dte=45, entry_interval_days=14,
                   ivp_filter=50.0, sma_filter=False, call_delta=0.25),
    # Iron condor — unfiltered
    BacktestConfig(structure='iron_condor', dte=45, entry_interval_days=14,
                   ivp_filter=None, sma_filter=False, ic_wing_width=0.05),
    # Iron condor — IVP > 50 filter
    BacktestConfig(structure='iron_condor', dte=45, entry_interval_days=14,
                   ivp_filter=50.0, sma_filter=False, ic_wing_width=0.05),
]

for sym, df in data.items():
    for cfg in configs:
        filter_label = f"ivp>{cfg.ivp_filter:.0f}" if cfg.ivp_filter else "unfiltered"
        trades = simulate_trades(df, cfg)
        if not trades:
            continue
        trades_df = trades_to_dataframe(trades)
        trades_df['structure'] = f"{cfg.structure}_{filter_label}"
        summary = trade_summary(trades_df, sym)
        all_trade_summaries.append(summary)

trade_results = pd.concat(all_trade_summaries, ignore_index=True)
print_trade_report(trade_results)

# %% ── Comparison: Filtered vs Unfiltered ────────────────

print("\n\n" + "=" * 70)
print("FILTER COMPARISON — IVP > 50 vs Unfiltered")
print("=" * 70)

for sym in SYMBOLS:
    sym_trades = trade_results[trade_results['symbol'] == sym]
    if sym_trades.empty:
        continue
    print(f"\n  {sym}:")
    print(f"  {'Structure':<30} {'Win%':>8} {'Avg P&L':>10} {'Total P&L':>12} {'PF':>6} {'Max DD':>10}")
    print(f"  {'─' * 76}")
    for _, row in sym_trades[sym_trades['regime'] == 'all'].iterrows():
        pf = f"{row['profit_factor']:.2f}" if row['profit_factor'] < 100 else "∞"
        print(f"  {row['structure']:<30} {row['win_rate']:>7.1f}% ${row['avg_pnl']:>9.2f} "
              f"${row['total_pnl']:>11.2f} {pf:>6} ${row['max_drawdown']:>9.2f}")

# %% ── Summary ──────────────────────────────────────────

print("\n\n" + "=" * 70)
print("KEY FINDINGS")
print("=" * 70)
print("""
Check:
1. Is VRP consistently positive across all underlyings? → Edge exists
2. Does IVP > 50 filter improve win rate? → Filter is valuable
3. Short put vs strangle vs IC — which has best risk-adjusted return?
4. Which underlying has the most reliable VRP?
""")
