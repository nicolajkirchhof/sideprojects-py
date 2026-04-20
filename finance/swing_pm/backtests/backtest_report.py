"""
finance.swing_pm.backtests.backtest_report
==========================================
Shared reporting utility for swing trading backtests.

Takes a filtered DataFrame (from the momentum_earnings dataset) and forward
return columns, computes standard metrics, and outputs a markdown summary.

Usage
-----
    from finance.swing_pm.backtests.backtest_report import generate_report

    report = generate_report(
        df=df_beats,
        label="PEAD Beats (Q4 SUE)",
        horizons=[1, 5, 10, 20, 40, 60],
        return_col_template="cpct{horizon}",
        segment_cols=["market_cap_class", "spy_class"],
    )
    print(report.markdown)
    report.save("finance/_data/backtest_results/swing/pead_beats.md")
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from finance.utils.backtest import InstrumentClass, cost_per_trade


# ---------------------------------------------------------------------------
# Cost constants
# ---------------------------------------------------------------------------
_DEFAULT_NOTIONAL = 5_000  # assumed average trade notional for cost estimation


@dataclass
class HorizonMetrics:
    """Metrics for a single forward-return horizon."""
    horizon: int
    n_events: int
    win_rate: float          # % of events with positive return
    mean_return: float       # mean forward return (%)
    median_return: float     # median forward return (%)
    std_return: float        # std dev of forward returns (%)
    sharpe_proxy: float      # mean / std (annualized proxy not applied — raw ratio)
    q05: float
    q25: float
    q75: float
    q95: float
    cost_adjusted_mean: float  # mean return minus round-trip cost as %


@dataclass
class SegmentBreakdown:
    """Metrics broken down by a segment column."""
    segment_col: str
    rows: list[dict] = field(default_factory=list)


@dataclass
class BacktestReport:
    """Complete backtest report with metrics, segments, and markdown output."""
    label: str
    total_events: int
    date_range: str
    horizons: list[HorizonMetrics]
    segments: list[SegmentBreakdown]
    markdown: str

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.markdown)


def _compute_horizon_metrics(
    series: pd.Series,
    horizon: int,
    cost_pct: float,
) -> HorizonMetrics:
    """Compute metrics for a single forward return series."""
    vals = series.dropna().to_numpy(dtype=float)
    n = len(vals)
    if n == 0:
        return HorizonMetrics(
            horizon=horizon, n_events=0, win_rate=0, mean_return=0,
            median_return=0, std_return=0, sharpe_proxy=0,
            q05=0, q25=0, q75=0, q95=0, cost_adjusted_mean=0,
        )

    mean_ret = float(np.nanmean(vals))
    std_ret = float(np.nanstd(vals))
    return HorizonMetrics(
        horizon=horizon,
        n_events=n,
        win_rate=float(np.mean(vals > 0) * 100),
        mean_return=mean_ret,
        median_return=float(np.nanmedian(vals)),
        std_return=std_ret,
        sharpe_proxy=mean_ret / std_ret if std_ret > 0 else 0,
        q05=float(np.nanquantile(vals, 0.05)),
        q25=float(np.nanquantile(vals, 0.25)),
        q75=float(np.nanquantile(vals, 0.75)),
        q95=float(np.nanquantile(vals, 0.95)),
        cost_adjusted_mean=mean_ret - cost_pct,
    )


def _format_horizon_table(horizons: list[HorizonMetrics]) -> str:
    """Format horizon metrics as a markdown table."""
    lines = [
        "| Horizon | N | Win% | Mean% | Median% | Std% | Sharpe | Q5% | Q25% | Q75% | Q95% | Net Mean% |",
        "|---------|---|------|-------|---------|------|--------|-----|------|------|------|-----------|",
    ]
    for h in horizons:
        lines.append(
            f"| {h.horizon:>3}d | {h.n_events:,} | {h.win_rate:.1f} | "
            f"{h.mean_return:+.2f} | {h.median_return:+.2f} | {h.std_return:.2f} | "
            f"{h.sharpe_proxy:+.3f} | {h.q05:+.1f} | {h.q25:+.1f} | "
            f"{h.q75:+.1f} | {h.q95:+.1f} | {h.cost_adjusted_mean:+.2f} |"
        )
    return "\n".join(lines)


def _format_segment_table(segment: SegmentBreakdown, horizons: list[int]) -> str:
    """Format segment breakdown as a markdown table."""
    if not segment.rows:
        return f"*No data for segment `{segment.segment_col}`*"

    horizon_headers = " | ".join(f"{h}d Mean%" for h in horizons)
    horizon_sep = " | ".join("-------" for _ in horizons)

    lines = [
        f"| {segment.segment_col} | N | {horizon_headers} |",
        f"|---|---|{horizon_sep}|",
    ]
    for row in segment.rows:
        vals = " | ".join(f"{row.get(f'mean_{h}', 0):+.2f}" for h in horizons)
        lines.append(f"| {row['value']} | {row['n']:,} | {vals} |")
    return "\n".join(lines)


def generate_report(
    df: pd.DataFrame,
    label: str,
    horizons: list[int] | None = None,
    return_col_template: str = "cpct{horizon}",
    segment_cols: list[str] | None = None,
    instrument_class: InstrumentClass = InstrumentClass.STOCK,
    notional: float = _DEFAULT_NOTIONAL,
) -> BacktestReport:
    """
    Generate a standardized backtest report from the momentum_earnings dataset.

    Parameters
    ----------
    df:
        Filtered DataFrame with forward return columns (cpct1..cpct60).
    label:
        Human-readable label for the report (e.g. "PEAD Beats Q4 SUE").
    horizons:
        List of forward-day horizons to measure (default: [1, 5, 10, 20, 40, 60]).
    return_col_template:
        Template for column names. Use {horizon} placeholder.
    segment_cols:
        Columns to segment by (e.g. ["market_cap_class", "spy_class"]).
    instrument_class:
        For cost estimation.
    notional:
        Average trade notional for cost estimation.
    """
    if horizons is None:
        horizons = [1, 5, 10, 20, 40, 60]
    if segment_cols is None:
        segment_cols = []

    # Cost as percentage of notional (round-trip)
    cost_dollars = cost_per_trade(instrument_class, notional)
    cost_pct = (cost_dollars / notional) * 100

    # Compute horizon metrics
    horizon_metrics = []
    for h in horizons:
        col = return_col_template.format(horizon=h)
        if col in df.columns:
            horizon_metrics.append(_compute_horizon_metrics(df[col], h, cost_pct))
        else:
            horizon_metrics.append(HorizonMetrics(
                horizon=h, n_events=0, win_rate=0, mean_return=0,
                median_return=0, std_return=0, sharpe_proxy=0,
                q05=0, q25=0, q75=0, q95=0, cost_adjusted_mean=0,
            ))

    # Compute segment breakdowns
    segments = []
    for seg_col in segment_cols:
        if seg_col not in df.columns:
            continue
        breakdown = SegmentBreakdown(segment_col=seg_col)
        for val, group in df.groupby(seg_col, dropna=False):
            row: dict = {"value": str(val), "n": len(group)}
            for h in horizons:
                col = return_col_template.format(horizon=h)
                if col in group.columns:
                    row[f"mean_{h}"] = float(group[col].mean())
                else:
                    row[f"mean_{h}"] = 0.0
            breakdown.rows.append(row)
        segments.append(breakdown)

    # Date range
    date_range = "unknown"
    if "date" in df.columns and not df["date"].isna().all():
        d_min = df["date"].min()
        d_max = df["date"].max()
        date_range = f"{d_min.date()} to {d_max.date()}"

    # Build markdown
    md_lines = [
        f"# Backtest Report: {label}",
        f"",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Events:** {len(df):,}",
        f"**Date range:** {date_range}",
        f"**Cost model:** {instrument_class.value} — ${cost_dollars:.2f}/trade ({cost_pct:.3f}% of ${notional:,.0f})",
        f"",
        f"---",
        f"",
        f"## Forward Returns by Horizon",
        f"",
        _format_horizon_table(horizon_metrics),
        f"",
    ]

    for seg in segments:
        md_lines.extend([
            f"---",
            f"",
            f"## By {seg.segment_col}",
            f"",
            _format_segment_table(seg, horizons),
            f"",
        ])

    markdown = "\n".join(md_lines)

    return BacktestReport(
        label=label,
        total_events=len(df),
        date_range=date_range,
        horizons=horizon_metrics,
        segments=segments,
        markdown=markdown,
    )
