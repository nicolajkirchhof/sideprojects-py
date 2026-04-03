"""
Core PM Analysis — Reporting and summary statistics for backtest results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def vrp_summary(vrp_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Summarize VRP statistics by window and IVP regime."""
    rows = []
    for col in [c for c in vrp_df.columns if c.startswith('vrp_')]:
        window = col.replace('vrp_', '')
        data = vrp_df[col].dropna()
        if data.empty:
            continue

        for regime, mask in _ivp_regimes(vrp_df):
            subset = data[mask].dropna()
            if len(subset) < 30:
                continue
            rows.append({
                'symbol': symbol,
                'window': window,
                'regime': regime,
                'n': len(subset),
                'mean_vrp': subset.mean(),
                'median_vrp': subset.median(),
                'pct_positive': (subset > 0).mean() * 100,
                'p5': subset.quantile(0.05),
                'p25': subset.quantile(0.25),
                'p75': subset.quantile(0.75),
                'p95': subset.quantile(0.95),
            })

    return pd.DataFrame(rows)


def range_test_summary(range_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Summarize range test win rates by IVP regime and SMA filter."""
    rows = []

    for regime_name, mask in _ivp_regimes_from_col(range_df, 'ivp'):
        for sma_label, sma_mask in [('all', pd.Series(True, index=range_df.index)),
                                     ('above_200sma', range_df['above_200sma'] == True),
                                     ('below_200sma', range_df['above_200sma'] == False)]:
            combined = mask & sma_mask
            subset = range_df[combined]
            if len(subset) < 30:
                continue
            rows.append({
                'symbol': symbol,
                'regime': regime_name,
                'sma_filter': sma_label,
                'n': len(subset),
                'win_rate': subset['stayed_in_range'].mean() * 100,
                'put_breach_rate': subset['put_breached'].mean() * 100,
                'call_breach_rate': subset['call_breached'].mean() * 100,
            })

    return pd.DataFrame(rows)


def trade_summary(trades_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Summarize trade results by structure and regime."""
    if trades_df.empty:
        return pd.DataFrame()

    rows = []
    for structure in trades_df['structure'].unique():
        struct_df = trades_df[trades_df['structure'] == structure]

        for regime_name, mask in _ivp_regimes_from_col(struct_df, 'ivp_at_entry'):
            subset = struct_df[mask]
            if len(subset) < 10:
                continue

            wins = subset[subset['win']]
            losses = subset[~subset['win']]
            cum_pnl = subset['pnl'].cumsum()

            rows.append({
                'symbol': symbol,
                'structure': structure,
                'regime': regime_name,
                'n_trades': len(subset),
                'win_rate': len(wins) / len(subset) * 100,
                'avg_pnl': subset['pnl'].mean(),
                'total_pnl': subset['pnl'].sum(),
                'avg_win': wins['pnl'].mean() if len(wins) > 0 else 0,
                'avg_loss': losses['pnl'].mean() if len(losses) > 0 else 0,
                'max_win': subset['pnl'].max(),
                'max_loss': subset['pnl'].min(),
                'profit_factor': abs(wins['pnl'].sum() / losses['pnl'].sum()) if losses['pnl'].sum() != 0 else np.inf,
                'avg_credit': subset['credit_received'].mean(),
                'avg_vrp': subset['vrp'].mean(),
                'stopped_out_pct': subset['stopped_out'].mean() * 100,
                'max_drawdown': _max_drawdown(cum_pnl),
            })

    return pd.DataFrame(rows)


def print_vrp_report(vrp_summaries: pd.DataFrame):
    """Print formatted VRP report."""
    print("\n" + "=" * 70)
    print("VARIANCE RISK PREMIUM ANALYSIS")
    print("=" * 70)

    for symbol in vrp_summaries['symbol'].unique():
        sym_data = vrp_summaries[vrp_summaries['symbol'] == symbol]
        print(f"\n{'─' * 50}")
        print(f"  {symbol}")
        print(f"{'─' * 50}")
        print(f"  {'Window':<8} {'Regime':<12} {'N':>6} {'Mean VRP':>10} {'% Positive':>12} {'P5':>8} {'P95':>8}")
        for _, row in sym_data.iterrows():
            print(f"  {row['window']:<8} {row['regime']:<12} {row['n']:>6.0f} "
                  f"{row['mean_vrp']:>10.4f} {row['pct_positive']:>11.1f}% "
                  f"{row['p5']:>8.4f} {row['p95']:>8.4f}")


def print_range_report(range_summaries: pd.DataFrame):
    """Print formatted range test report."""
    print("\n" + "=" * 70)
    print("RANGE TEST — STRANGLE WIN RATES (±1 SD, 45d)")
    print("=" * 70)

    for symbol in range_summaries['symbol'].unique():
        sym_data = range_summaries[range_summaries['symbol'] == symbol]
        print(f"\n{'─' * 50}")
        print(f"  {symbol}")
        print(f"{'─' * 50}")
        print(f"  {'Regime':<12} {'SMA Filter':<15} {'N':>6} {'Win%':>8} {'Put Breach':>12} {'Call Breach':>12}")
        for _, row in sym_data.iterrows():
            print(f"  {row['regime']:<12} {row['sma_filter']:<15} {row['n']:>6.0f} "
                  f"{row['win_rate']:>7.1f}% {row['put_breach_rate']:>11.1f}% "
                  f"{row['call_breach_rate']:>11.1f}%")


def print_trade_report(trade_summaries: pd.DataFrame):
    """Print formatted trade simulation report."""
    print("\n" + "=" * 70)
    print("TRADE SIMULATION RESULTS")
    print("=" * 70)

    for symbol in trade_summaries['symbol'].unique():
        sym_data = trade_summaries[trade_summaries['symbol'] == symbol]
        print(f"\n{'─' * 60}")
        print(f"  {symbol}")
        print(f"{'─' * 60}")
        for _, row in sym_data.iterrows():
            print(f"\n  {row['structure'].upper()} — {row['regime']}")
            print(f"    Trades: {row['n_trades']:.0f}  |  Win rate: {row['win_rate']:.1f}%  |  "
                  f"Stopped out: {row['stopped_out_pct']:.1f}%")
            print(f"    Avg P&L: ${row['avg_pnl']:.2f}  |  Total P&L: ${row['total_pnl']:.2f}")
            print(f"    Avg win: ${row['avg_win']:.2f}  |  Avg loss: ${row['avg_loss']:.2f}  |  "
                  f"Profit factor: {row['profit_factor']:.2f}")
            print(f"    Max DD: ${row['max_drawdown']:.2f}  |  Avg credit: ${row['avg_credit']:.2f}  |  "
                  f"Avg VRP: {row['avg_vrp']:.4f}")


# ── Helpers ──────────────────────────────────────────────

def _ivp_regimes(df: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    """Generate IVP regime masks from iv_pct column."""
    ivp = df['iv_pct'] if 'iv_pct' in df.columns else pd.Series(50.0, index=df.index)
    return [
        ('all', pd.Series(True, index=df.index)),
        ('ivp<30', ivp < 30),
        ('ivp_30_50', (ivp >= 30) & (ivp < 50)),
        ('ivp_50_70', (ivp >= 50) & (ivp < 70)),
        ('ivp>70', ivp >= 70),
    ]


def _ivp_regimes_from_col(df: pd.DataFrame, col: str) -> list[tuple[str, pd.Series]]:
    """Generate IVP regime masks from a named column."""
    ivp = df[col] if col in df.columns else pd.Series(50.0, index=df.index)
    return [
        ('all', pd.Series(True, index=df.index)),
        ('ivp<30', ivp < 30),
        ('ivp_30_50', (ivp >= 30) & (ivp < 50)),
        ('ivp_50_70', (ivp >= 50) & (ivp < 70)),
        ('ivp>70', ivp >= 70),
    ]


def _max_drawdown(cumulative_pnl: pd.Series) -> float:
    """Calculate max drawdown from a cumulative P&L series."""
    if cumulative_pnl.empty:
        return 0.0
    peak = cumulative_pnl.expanding().max()
    drawdown = cumulative_pnl - peak
    return drawdown.min()
