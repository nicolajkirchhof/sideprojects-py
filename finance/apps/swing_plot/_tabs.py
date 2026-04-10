"""
finance.apps.swing_plot._tabs
===============================
Matplotlib tab renderers for the swing plot dashboard.
Each render_* function accepts a matplotlib Figure and populates it.
No Qt, no PyQtGraph — pure matplotlib.
"""
import calendar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.stats import norm as _scipy_norm

from finance.utils.plots import violinplot_columns_with_labels, plot_probability_tree, annotate_violin
from finance.utils.momentum_data import load_ticker_earnings_events
from finance.utils.move_character import (
    calculate_regime_filter_stats,
    calculate_move_magnitude_stats,
    calculate_intratrend_retracement,
    calculate_hv_regime_stats,
    calculate_impulse_forward_returns,
    calculate_ivp,
    calculate_vrp,
    calculate_time_underwater,
    calculate_otm_viability,
    calculate_rs_vs_spy,
    calculate_vcp_tightness,
    calculate_overnight_reversal,
)


# ---------------------------------------------------------------------------
# Shared styling helper
# ---------------------------------------------------------------------------

def _style_ax(ax):
    """Apply dark-background styling to a matplotlib Axes."""
    ax.set_facecolor('#111111')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    ax.grid(True, alpha=0.2, color='white')


# ---------------------------------------------------------------------------
# Tab: Daily / Monthly Statistics (seasonality)
# ---------------------------------------------------------------------------

def render_stats_violins(fig: Figure, df: pd.DataFrame, start_year: int, end_year: int):
    fig.clear()
    fig.patch.set_facecolor('#111111')

    df_range = df[(df.index.year >= start_year) & (df.index.year <= end_year)]
    if df_range.empty or 'pct' not in df_range.columns:
        ax = fig.add_subplot(111)
        _style_ax(ax)
        ax.set_title(f'No data for {start_year}–{end_year}')
        return

    day_names = list(calendar.day_abbr)
    df_day_plot = pd.DataFrame()
    for i in range(5):
        d_data = df_range[df_range.index.dayofweek == i]['pct']
        if not d_data.empty:
            df_day_plot[day_names[i]] = d_data.reset_index(drop=True)

    df_month_plot = pd.DataFrame()
    for i in range(1, 13):
        m_data = df_range[df_range.index.month == i]['pct']
        if not m_data.empty:
            df_month_plot[calendar.month_name[i]] = m_data.reset_index(drop=True)

    axs = fig.subplots(nrows=2, ncols=1)
    for ax in axs:
        _style_ax(ax)

    if not df_day_plot.empty:
        violinplot_columns_with_labels(df_day_plot, ax=axs[0],
                                        title=f'Daily Returns by Weekday ({start_year}–{end_year})')
    else:
        axs[0].set_title(f'No daily data for {start_year}–{end_year}')

    if not df_month_plot.empty:
        violinplot_columns_with_labels(df_month_plot, ax=axs[1],
                                        title=f'Daily Returns by Month ({start_year}–{end_year})')
    else:
        axs[1].set_title(f'No monthly data for {start_year}–{end_year}')

    fig.tight_layout()


# ---------------------------------------------------------------------------
# Tab: Volatility Analysis
# ---------------------------------------------------------------------------

def render_volatility_analysis(fig: Figure, df: pd.DataFrame, start_year: int, end_year: int,
                                spy_df: pd.DataFrame | None = None):
    fig.clear()
    fig.patch.set_facecolor('#111111')

    if 'iv' not in df.columns:
        ax = fig.add_subplot(111)
        _style_ax(ax)
        ax.set_title('No IV data available for this instrument')
        return

    df_vol = df[(df.index.year >= start_year) & (df.index.year <= end_year)].copy()
    df_vol = df_vol.dropna(subset=['iv'])
    if df_vol.empty:
        ax = fig.add_subplot(111)
        _style_ax(ax)
        ax.set_title(f'No IV data for {start_year}–{end_year}')
        return

    have_spy = spy_df is not None and not spy_df.empty and 'c' in spy_df.columns
    n_rows = 6 if have_spy else 5

    gs = fig.add_gridspec(
        n_rows, 2, hspace=0.55, wspace=0.25,
        left=0.04, right=0.995, top=0.975, bottom=0.03,
    )
    ax_ivp     = fig.add_subplot(gs[0, :])
    ax_vrp     = fig.add_subplot(gs[1, :])
    ax_premium = fig.add_subplot(gs[2, :])
    ax_hv_time = fig.add_subplot(gs[3, :])
    ax_hv14    = fig.add_subplot(gs[4, 0])
    ax_hv20    = fig.add_subplot(gs[4, 1])
    axes = [ax_ivp, ax_vrp, ax_premium, ax_hv_time, ax_hv14, ax_hv20]
    if have_spy:
        ax_corr = fig.add_subplot(gs[5, :])
        axes.append(ax_corr)
    for ax in axes:
        _style_ax(ax)

    # ---- IVP timeline (1y lookback) ----
    ivp = calculate_ivp(df_vol['iv'], window=252).dropna()
    if not ivp.empty:
        ax_ivp.plot(ivp.index, ivp.values, color='#48dbfb', linewidth=1.0)
        ax_ivp.fill_between(ivp.index, 50, ivp.values, where=(ivp.values >= 50),
                             color='#24ad54', alpha=0.2, label='IVP ≥ 50 (DRIFT-eligible)')
        ax_ivp.fill_between(ivp.index, 50, ivp.values, where=(ivp.values < 50),
                             color='#ec4533', alpha=0.15)
        ax_ivp.axhline(50, color='#f5a623', linestyle='--', linewidth=0.8)
        ax_ivp.set_ylim(0, 100)
        ax_ivp.set_title(f'IV Percentile (1y)   current: {float(ivp.iloc[-1]):.0f}')
        ax_ivp.set_ylabel('IVP (%)')
        leg = ax_ivp.legend(fontsize='x-small', loc='upper left')
        plt.setp(leg.get_texts(), color='white')
    else:
        ax_ivp.set_title('IVP — insufficient history')

    # ---- VRP rolling spread (IV − HV20) with percentile ----
    hv_for_vrp = 'hv20' if 'hv20' in df_vol.columns else ('hv14' if 'hv14' in df_vol.columns else None)
    if hv_for_vrp:
        vrp_df = calculate_vrp(df_vol['iv'], df_vol[hv_for_vrp], window=252).dropna()
        if not vrp_df.empty:
            ax_vrp.plot(vrp_df.index, vrp_df['vrp'], color='#48dbfb', linewidth=1.0, label=f'IV − {hv_for_vrp.upper()}')
            ax_vrp.fill_between(vrp_df.index, 0, vrp_df['vrp'],
                                 where=(vrp_df['vrp'] >= 0), color='#24ad54', alpha=0.2)
            ax_vrp.fill_between(vrp_df.index, 0, vrp_df['vrp'],
                                 where=(vrp_df['vrp'] < 0), color='#ec4533', alpha=0.2)
            ax_vrp.axhline(0, color='#f5a623', linestyle='--', linewidth=0.8)
            cur_vrp = float(vrp_df['vrp'].iloc[-1])
            cur_pct = float(vrp_df['vrp_pct'].iloc[-1]) if pd.notna(vrp_df['vrp_pct'].iloc[-1]) else float('nan')
            ax_vrp.set_title(f'Variance Risk Premium   current: {cur_vrp:+.2f}  ({cur_pct:.0f}th pctile)')
            ax_vrp.set_ylabel('IV − HV')
            leg = ax_vrp.legend(fontsize='x-small', loc='upper left')
            plt.setp(leg.get_texts(), color='white')
    else:
        ax_vrp.set_title('VRP — no HV data')

    # ---- IV–HV premium distributions ----
    hv_cols = [c for c in ['hvc', 'hv14', 'hv20', 'hv50'] if c in df_vol.columns]
    if hv_cols:
        df_prem = pd.DataFrame({f'IV-{col}': df_vol['iv'] - df_vol[col] for col in hv_cols})
        violinplot_columns_with_labels(df_prem, ax=ax_premium,
                                        title='IV − HV Premium Distributions')

    # ---- IV vs HV time series ----
    ax_hv_time.plot(df_vol.index, df_vol['iv'], label='IV', color='cyan', lw=2)
    for col in hv_cols:
        ax_hv_time.plot(df_vol.index, df_vol[col], label=col, alpha=0.6, lw=1)
    ax_hv_time.set_title('IV vs Realized Volatilities')
    leg = ax_hv_time.legend(ncol=len(hv_cols) + 1, loc='upper left', framealpha=0.3)
    plt.setp(leg.get_texts(), color='white')

    # ---- IV vs HV14 scatter ----
    if 'hv14' in df_vol.columns:
        max_v = max(df_vol['iv'].max(), df_vol['hv14'].max())
        ax_hv14.scatter(df_vol['hv14'], df_vol['iv'], alpha=0.3, color='tab:blue')
        ax_hv14.plot([0, max_v], [0, max_v], 'r--', alpha=0.6, label='IV=HV14')
        ax_hv14.set_title('IV vs HV14 (Short-term)')
        ax_hv14.set_xlabel('HV14')
        ax_hv14.set_ylabel('IV')
        leg = ax_hv14.legend(framealpha=0.3)
        plt.setp(leg.get_texts(), color='white')

    # ---- IV vs HV20 scatter ----
    if 'hv20' in df_vol.columns:
        max_v = max(df_vol['iv'].max(), df_vol['hv20'].max())
        ax_hv20.scatter(df_vol['hv20'], df_vol['iv'], alpha=0.3, color='tab:green')
        ax_hv20.plot([0, max_v], [0, max_v], 'r--', alpha=0.6, label='IV=HV20')
        ax_hv20.set_title('IV vs HV20 (Standard)')
        ax_hv20.set_xlabel('HV20')
        ax_hv20.set_ylabel('IV')
        leg = ax_hv20.legend(framealpha=0.3)
        plt.setp(leg.get_texts(), color='white')

    # ---- Rolling 60d correlation to SPY ----
    if have_spy:
        merged = pd.concat([df_vol['c'].pct_change().rename('r'),
                            spy_df['c'].pct_change().rename('spy_r')], axis=1).dropna()
        if not merged.empty:
            corr = merged['r'].rolling(60).corr(merged['spy_r']).dropna()
            ax_corr.plot(corr.index, corr.values, color='#48dbfb', linewidth=1.0)
            ax_corr.axhline(0, color='#f5a623', linestyle='--', linewidth=0.8)
            ax_corr.set_ylim(-1, 1)
            cur = float(corr.iloc[-1]) if not corr.empty else float('nan')
            ax_corr.set_title(f'Rolling 60d Correlation to SPY   current: {cur:+.2f}')
            ax_corr.set_ylabel('ρ')
        else:
            ax_corr.set_title('SPY correlation — no overlap')


# ---------------------------------------------------------------------------
# Tab: Drawdown Analysis
# ---------------------------------------------------------------------------

def render_drawdown_analysis(fig: Figure, df: pd.DataFrame, start_year: int, end_year: int):
    fig.clear()
    fig.patch.set_facecolor('#111111')

    if 'c' not in df.columns:
        return

    df_dd = df[(df.index.year >= start_year) & (df.index.year <= end_year)].copy()
    if df_dd.empty:
        ax = fig.add_subplot(111)
        _style_ax(ax)
        ax.set_title(f'No data for {start_year}–{end_year}')
        return

    df_dd['ath']          = df_dd['c'].cummax()
    df_dd['drawdown_pct'] = (df_dd['c'] - df_dd['ath']) / df_dd['ath'] * 100
    df_dd['is_dd']        = df_dd['c'] < df_dd['ath']
    df_dd['dd_group']     = (df_dd['is_dd'] != df_dd['is_dd'].shift()).cumsum()

    drawdown_groups = df_dd[df_dd['is_dd']].groupby('dd_group')

    gs = fig.add_gridspec(5, 1, height_ratios=[1, 1, 1.5, 1.5, 1.5])
    ax_scatter = fig.add_subplot(gs[0])
    ax_tuw     = fig.add_subplot(gs[1])
    ax_short   = fig.add_subplot(gs[2])
    ax_med     = fig.add_subplot(gs[3])
    ax_long    = fig.add_subplot(gs[4])

    for ax in [ax_scatter, ax_tuw, ax_short, ax_med, ax_long]:
        _style_ax(ax)

    # Time-underwater histogram (days between ATH and next ATH)
    tuw = calculate_time_underwater(df_dd['c'])
    if not tuw.empty:
        bins = np.logspace(0, np.log10(max(10, tuw.max())), 30)
        ax_tuw.hist(tuw.values, bins=bins, color='#48dbfb', alpha=0.75)
        ax_tuw.set_xscale('log')
        ax_tuw.axvline(float(tuw.median()), color='#f5a623', linestyle='--', linewidth=1.0,
                       label=f'Median: {tuw.median():.0f}d')
        ax_tuw.axvline(float(tuw.quantile(0.9)), color='#ec4533', linestyle='--', linewidth=1.0,
                       label=f'90th: {tuw.quantile(0.9):.0f}d')
        ax_tuw.set_title(f'Time Underwater — {len(tuw)} episodes')
        ax_tuw.set_xlabel('Days from ATH to next ATH (log scale)')
        ax_tuw.set_ylabel('# Episodes')
        leg = ax_tuw.legend(fontsize='x-small')
        plt.setp(leg.get_texts(), color='white')
    else:
        ax_tuw.set_title('Time Underwater — no episodes')

    dd_summary = []
    counts = {'short': 0, 'med': 0, 'long': 0}

    for _, group in drawdown_groups:
        duration = len(group)
        if duration < 2:
            continue
        severity = group['drawdown_pct'].min()
        pre_dd_slice = df_dd.loc[df_dd.index < group.index[0], 'iv'].tail(1).values \
            if 'iv' in df_dd.columns else []
        iv_bottom = group.loc[group['drawdown_pct'].idxmin(), 'iv'] \
            if 'iv' in group.columns else 0
        iv_exp = ((iv_bottom - pre_dd_slice[0]) / pre_dd_slice[0] * 100) \
            if (len(pre_dd_slice) > 0 and pre_dd_slice[0] > 0) else 0

        dd_summary.append({'dur': duration, 'sev': abs(severity), 'iv': iv_exp})

        pre_dd     = df_dd.loc[df_dd.index < group.index[0]].tail(1)
        base_price = pre_dd['c'].values[0] if not pre_dd.empty else group['c'].iloc[0]
        path = (group['c'].to_numpy() / base_price - 1) * 100
        days = np.arange(len(path))

        if duration < 30:
            ax_short.plot(days, path, alpha=0.3, linewidth=1, color='tab:blue')
            counts['short'] += 1
        elif duration <= 65:
            ax_med.plot(days, path, alpha=0.5, linewidth=1.2,
                        label=f"{group.index[0].year} ({duration}d)")
            counts['med'] += 1
        else:
            ax_long.plot(days, path, alpha=0.7, linewidth=1.5,
                         label=f"{group.index[0].year} ({duration}d)")
            counts['long'] += 1

    if dd_summary:
        df_summ = pd.DataFrame(dd_summary)
        sc = ax_scatter.scatter(df_summ['dur'], df_summ['sev'],
                                c=df_summ['iv'], cmap='YlOrRd', s=100, alpha=0.6, edgecolors='white')
        ax_scatter.set_title('Drawdown Severity vs Duration (Color: IV Expansion %)')
        ax_scatter.set_xlabel('Duration (Days)')
        ax_scatter.set_ylabel('Max Severity (%)')
        cb = fig.colorbar(sc, ax=ax_scatter)
        cb.set_label('IV Expansion %', color='white')
        cb.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')

    for ax, key, title in [
        (ax_short, 'short', f"Short-Term (< 30 Days, n={counts['short']})"),
        (ax_med,   'med',   f"Medium-Term (30-65 Days, n={counts['med']})"),
        (ax_long,  'long',  f"Long-Term (> 65 Days, n={counts['long']})"),
    ]:
        ax.axhline(0, color='white', linewidth=1, alpha=0.5)
        ax.set_title(title)
        ax.set_ylabel('Close % Change from ATH')
        ax.grid(True, alpha=0.2, color='white')

    if counts['med'] > 0:
        ncol = 5 if counts['med'] > 15 else 3
        leg = ax_med.legend(loc='lower left', fontsize=8, ncol=ncol, framealpha=0.3)
        plt.setp(leg.get_texts(), color='white')

    ax_long.set_xlabel('Days since ATH')
    if counts['long'] > 0:
        leg = ax_long.legend(loc='lower left', fontsize=9, ncol=4, framealpha=0.3)
        plt.setp(leg.get_texts(), color='white')

    fig.tight_layout()


# ---------------------------------------------------------------------------
# Tab: Move Character (Blocks 1–5 + Regime)
# ---------------------------------------------------------------------------

_STATE_COLORS = {'Uptrend': '#24ad54', 'Pullback': '#f5a623', 'Breakdown': '#ec4533'}
_REGIME_COLORS = {'Low': '#24ad54', 'Medium': '#f5a623', 'High': '#ec4533'}


def render_trend_regime(fig: Figure, df: pd.DataFrame, symbol: str = ''):
    """
    Trader-primary tab: MA20 regime stats (PM-01 trend confirmation).

      Rows 0-2 — Regime Filter (episodes, recovery, durations, depth, fwd returns)
    """
    fig.clear()
    fig.patch.set_facecolor('#111111')
    try:
        fig.set_layout_engine('none')
    except Exception:
        pass

    master = fig.add_gridspec(
        3, 1,
        hspace=0.55,
        left=0.035, right=0.995, top=0.975, bottom=0.03,
    )

    try:
        episodes_df, regime_summary = calculate_regime_filter_stats(df)
        _render_regime_section(fig, master[0:3, 0], episodes_df, regime_summary, symbol)
    except Exception as e:
        ax = fig.add_subplot(master[0:3, 0])
        _style_ax(ax)
        ax.set_title(f'Regime Filter: {e}')


def render_pullback_vcp(fig: Figure, df: pd.DataFrame, symbol: str = '',
                         spy_df: pd.DataFrame | None = None):
    """
    Trader-primary tab: pullback entries + VCP tightness (PM-01 / PM-09).

      Row 0 — Intra-Trend Retracement Depth
      Row 1 — RS line vs SPY + 63d RS percentile (if spy_df available)
      Row 2 — VCP Tightness (range-contraction detector)
    """
    fig.clear()
    fig.patch.set_facecolor('#111111')
    try:
        fig.set_layout_engine('none')
    except Exception:
        pass

    have_spy = spy_df is not None and not spy_df.empty
    n_rows = 3 if have_spy else 2
    master = fig.add_gridspec(
        n_rows, 1,
        hspace=0.55,
        left=0.035, right=0.995, top=0.975, bottom=0.03,
    )

    try:
        long_retr, short_retr = calculate_intratrend_retracement(df)
        med_atr20 = df['atrp20'].median() if 'atrp20' in df.columns else 1.0
        _render_retracement_section(fig, master[0, 0], long_retr, short_retr, med_atr20, symbol)
    except Exception as e:
        ax = fig.add_subplot(master[0, 0])
        _style_ax(ax)
        ax.set_title(f'Retracement: {e}')

    if have_spy:
        try:
            _render_rs_section(fig, master[1, 0], df, spy_df, symbol)
        except Exception as e:
            ax = fig.add_subplot(master[1, 0])
            _style_ax(ax)
            ax.set_title(f'RS vs SPY: {e}')

    vcp_row = 2 if have_spy else 1
    try:
        _render_vcp_section(fig, master[vcp_row, 0], df, symbol)
    except Exception as e:
        ax = fig.add_subplot(master[vcp_row, 0])
        _style_ax(ax)
        ax.set_title(f'VCP Tightness: {e}')


def render_move_otm(fig: Figure, df: pd.DataFrame, symbol: str = ''):
    """
    Trader-primary tab: decide whether a swing can be expressed via long OTM
    options (fat tails required) and size stops/targets.

      Row  0   — Move Magnitude (hist, tail freq, OTM viability QQ)
      Row  1   — IVP timeline (1y window) with 50% band
      Rows 2-4 — HV Regime + Impulse forward returns
      Row  5   — Overnight Reversal (PM-08)
    """
    fig.clear()
    fig.patch.set_facecolor('#111111')
    try:
        fig.set_layout_engine('none')
    except Exception:
        pass

    master = fig.add_gridspec(
        6, 1,
        hspace=0.55,
        left=0.035, right=0.995, top=0.975, bottom=0.03,
    )

    try:
        norm_moves, tail_freq, excess_kurtosis, skewness = calculate_move_magnitude_stats(df)
        flag_move = 'EXPLOSIVE' if tail_freq[2.0]['total'] > 5.0 else 'GRADUAL'
        _render_move_magnitude_section(fig, master[0, 0], df, norm_moves, tail_freq,
                                        excess_kurtosis, skewness, flag_move, symbol)
    except Exception as e:
        ax = fig.add_subplot(master[0, 0])
        _style_ax(ax)
        ax.set_title(f'Move Magnitude: {e}')

    try:
        _render_ivp_section(fig, master[1, 0], df, symbol)
    except Exception as e:
        ax = fig.add_subplot(master[1, 0])
        _style_ax(ax)
        ax.set_title(f'IVP: {e}')

    try:
        hv_series, hv_regime, hv_episodes, hv_transitions, hv_col_used = \
            calculate_hv_regime_stats(df)
        impulse_df = calculate_impulse_forward_returns(df)
        _render_hv_impulse_section(fig, master[2:5, 0], hv_series, hv_regime, hv_episodes,
                                    hv_transitions, hv_col_used, impulse_df, symbol)
    except Exception as e:
        ax = fig.add_subplot(master[2:5, 0])
        _style_ax(ax)
        ax.set_title(f'HV Regime / Impulse: {e}')

    try:
        _render_overnight_reversal_section(fig, master[5, 0], df, symbol)
    except Exception as e:
        ax = fig.add_subplot(master[5, 0])
        _style_ax(ax)
        ax.set_title(f'Overnight Reversal: {e}')


# ---- sub-renderers (one per section) ----------------------------------------

def _render_regime_section(fig, spec, episodes_df, regime_summary, symbol):
    """3 rows x 6 cols: summary bars + duration/depth violins + forward-return violins."""
    gs = spec.subgridspec(3, 6, hspace=0.55, wspace=0.35)
    states  = ['Uptrend', 'Pullback', 'Breakdown']
    colors  = [_STATE_COLORS[s] for s in states]

    pullback_eps  = episodes_df[episodes_df['state'] == 'Pullback']
    breakdown_eps = episodes_df[episodes_df['state'] == 'Breakdown']
    uptrend_eps   = episodes_df[episodes_df['state'] == 'Uptrend']

    # Episode count
    ax_cnt = fig.add_subplot(gs[0, 0:2])
    _style_ax(ax_cnt)
    counts = [regime_summary[s].get('n_episodes', 0) for s in states]
    bars = ax_cnt.bar(states, counts, color=colors, alpha=0.8)
    for bar, val in zip(bars, counts):
        ax_cnt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(int(val)), ha='center', va='bottom', color='white', fontsize='small')
    ax_cnt.set_title(f'Regime Episode Count — {symbol}')
    ax_cnt.set_ylabel('# Episodes')

    # Pullback recovery rate
    ax_rec = fig.add_subplot(gs[0, 2:4])
    _style_ax(ax_rec)
    rr = regime_summary['Pullback'].get('recovery_rate_pct', 0)
    ax_rec.bar(['Recovery\n(≤10d)', 'No Recovery'], [rr, 100 - rr],
               color=['#24ad54', '#ec4533'], alpha=0.8)
    ax_rec.set_ylim(0, 100)
    ax_rec.set_title('Pullback → MA20 Recovery Rate')
    ax_rec.set_ylabel('% of Episodes')
    ax_rec.text(0, rr + 1, f'{rr:.1f}%', ha='center', color='white', fontsize='small')

    # Median duration
    ax_dur = fig.add_subplot(gs[0, 4:6])
    _style_ax(ax_dur)
    med_durs = [regime_summary[s].get('med_duration', 0) for s in states]
    bars2 = ax_dur.bar(states, med_durs, color=colors, alpha=0.8)
    for bar, val in zip(bars2, med_durs):
        ax_dur.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f'{val:.1f}d', ha='center', va='bottom', color='white', fontsize='small')
    ax_dur.set_title('Median Episode Duration (Days)')
    ax_dur.set_ylabel('Days')

    # Duration violin (Pullback vs Breakdown)
    ax_durv = fig.add_subplot(gs[1, 0:3])
    _style_ax(ax_durv)
    dur_data   = [g['duration'].dropna().values
                  for g in [pullback_eps, breakdown_eps] if len(g) > 1]
    dur_labels = [l for g, l in [(pullback_eps, 'Pullback'), (breakdown_eps, 'Breakdown')] if len(g) > 1]
    if dur_data:
        parts = ax_durv.violinplot(dur_data, showmedians=False, showextrema=True,
                                    quantiles=[[0.25, 0.5, 0.75]] * len(dur_data))
        for pc, lbl in zip(parts['bodies'], dur_labels):
            pc.set_facecolor(_STATE_COLORS[lbl]); pc.set_alpha(0.6)
        ax_durv.set_xticks(np.arange(1, len(dur_labels) + 1))
        ax_durv.set_xticklabels(dur_labels)
        annotate_violin(ax_durv, dur_data, np.arange(1, len(dur_labels) + 1), dur_labels)
    ax_durv.set_title('Episode Duration Distribution')
    ax_durv.set_ylabel('Days')

    # Max depth violin
    ax_dep = fig.add_subplot(gs[1, 3:6])
    _style_ax(ax_dep)
    dep_data   = [g['max_depth_pct'].dropna().values
                  for g in [pullback_eps, breakdown_eps] if len(g) > 1]
    dep_labels = dur_labels
    if dep_data:
        parts2 = ax_dep.violinplot(dep_data, showmedians=False, showextrema=True,
                                    quantiles=[[0.25, 0.5, 0.75]] * len(dep_data))
        for pc, lbl in zip(parts2['bodies'], dep_labels):
            pc.set_facecolor(_STATE_COLORS[lbl]); pc.set_alpha(0.6)
        ax_dep.set_xticks(np.arange(1, len(dep_labels) + 1))
        ax_dep.set_xticklabels(dep_labels)
        annotate_violin(ax_dep, dep_data, np.arange(1, len(dep_labels) + 1), dep_labels)
    ax_dep.axhline(0, color='#666', linewidth=0.8, linestyle='--')
    ax_dep.set_title('Max Depth Below MA20 (%)')
    ax_dep.set_ylabel('% Below MA20')

    # Forward returns
    for col_idx, (fwd_col, fwd_label) in enumerate([('fwd_5', '5d'), ('fwd_10', '10d'), ('fwd_20', '20d')]):
        ax_fwd = fig.add_subplot(gs[2, col_idx * 2: col_idx * 2 + 2])
        _style_ax(ax_fwd)
        fwd_vdata, fwd_vlabels = [], []
        for state, ep_df in [('Uptrend', uptrend_eps), ('Pullback', pullback_eps), ('Breakdown', breakdown_eps)]:
            if fwd_col in ep_df.columns:
                vals = ep_df[fwd_col].dropna().values
                if len(vals) > 1:
                    fwd_vdata.append(vals)
                    fwd_vlabels.append(state)
        if fwd_vdata:
            parts3 = ax_fwd.violinplot(fwd_vdata, showmedians=False, showextrema=True,
                                        quantiles=[[0.25, 0.5, 0.75]] * len(fwd_vdata))
            for pc, lbl in zip(parts3['bodies'], fwd_vlabels):
                pc.set_facecolor(_STATE_COLORS[lbl]); pc.set_alpha(0.6)
            ax_fwd.set_xticks(np.arange(1, len(fwd_vlabels) + 1))
            ax_fwd.set_xticklabels(fwd_vlabels, fontsize='x-small')
            annotate_violin(ax_fwd, fwd_vdata, np.arange(1, len(fwd_vlabels) + 1), fwd_vlabels)
        ax_fwd.axhline(0, color='#666', linewidth=0.8, linestyle='--')
        ax_fwd.set_title(f'Forward Return — {fwd_label}')
        ax_fwd.set_ylabel('Return %')


def _render_move_magnitude_section(fig, spec, df, norm_moves, tail_freq,
                                    excess_kurtosis, skewness, flag_move, symbol):
    gs = spec.subgridspec(1, 6, wspace=0.35)
    # Histogram — cols 0-1
    ax_hist = fig.add_subplot(gs[0, 0:2])
    _style_ax(ax_hist)
    vals = norm_moves.dropna().values
    lo, hi = np.percentile(vals, 0.5), np.percentile(vals, 99.5)
    ax_hist.hist(vals, bins=np.linspace(lo, hi, 60), color='#48dbfb', alpha=0.7, density=True)
    ax_hist.set_title(f'ATR20-Normalised Moves — {symbol}  [{flag_move}]')
    ax_hist.set_xlabel('Move / ATR20')
    ax_hist.set_ylabel('Density')
    for t, color in [(1.0, '#f5a623'), (1.5, '#ff8c00'), (2.0, '#ec4533'), (3.0, '#9b59b6')]:
        freq = tail_freq[t]['total']
        ax_hist.axvline( t, color=color, linestyle='--', linewidth=1.2, label=f'>{t:.1f}× ({freq:.1f}%)')
        ax_hist.axvline(-t, color=color, linestyle='--', linewidth=1.2)
    ax_hist.legend(fontsize='x-small')
    ax_hist.text(0.02, 0.97, f'Kurt: {excess_kurtosis:.1f}  Skew: {skewness:.2f}',
                 transform=ax_hist.transAxes, va='top', fontsize='x-small', color='#aaa')

    # Tail frequency bars — cols 2-3
    ax_tail = fig.add_subplot(gs[0, 2:4])
    _style_ax(ax_tail)
    thresh_labels = [f'>{t:.1f}×' for t in tail_freq]
    up_vals   = [tail_freq[t]['up']   for t in tail_freq]
    down_vals = [tail_freq[t]['down'] for t in tail_freq]
    x3, w3 = np.arange(len(thresh_labels)), 0.35
    ax_tail.bar(x3 - w3 / 2, up_vals,   w3, label='Up',   color='#24ad54', alpha=0.8)
    ax_tail.bar(x3 + w3 / 2, down_vals, w3, label='Down', color='#ec4533', alpha=0.8)
    for i, (u, d) in enumerate(zip(up_vals, down_vals)):
        ax_tail.text(i - w3 / 2, u + 0.05, f'{u:.1f}', ha='center', fontsize='xx-small', color='white')
        ax_tail.text(i + w3 / 2, d + 0.05, f'{d:.1f}', ha='center', fontsize='xx-small', color='white')
    ax_tail.set_xticks(x3)
    ax_tail.set_xticklabels(thresh_labels)
    ax_tail.set_title('Tail Frequencies by Direction (%)')
    ax_tail.set_ylabel('% of Days')
    ax_tail.legend(fontsize='x-small')
    ax_tail.axhline(5.0, color='#f5a623', linestyle=':', linewidth=1, alpha=0.7)

    # QQ plot reframed as OTM Long-Option Viability — cols 4-5
    ax_qq = fig.add_subplot(gs[0, 4:6])
    _style_ax(ax_qq)
    pct_raw = df['pct'].dropna().values
    mean_v, std_v = pct_raw.mean(), pct_raw.std(ddof=1)
    std_sorted = np.sort((pct_raw - mean_v) / std_v)
    n_qq = len(std_sorted)
    theoretical_q = _scipy_norm.ppf((np.arange(1, n_qq + 1) - 0.5) / n_qq)
    ax_qq.scatter(theoretical_q, std_sorted, s=3, alpha=0.4, color='#48dbfb')
    ref = np.array([theoretical_q[0], theoretical_q[-1]])
    ax_qq.plot(ref, ref, color='#f5a623', linewidth=1.5, linestyle='--', label='Normal')
    viability = calculate_otm_viability(df['pct'])
    flag = viability['flag']
    flag_color = {'VIABLE': '#24ad54', 'MARGINAL': '#f5a623', 'POOR': '#ec4533'}[flag]
    ax_qq.set_title(f'OTM Long-Option Viability — [{flag}]')
    ax_qq.set_xlabel('Theoretical Quantiles')
    ax_qq.set_ylabel('Empirical Quantiles')
    ax_qq.legend(fontsize='x-small')
    info = (f'Kurt: {viability["kurtosis"]:.2f}   Skew: {viability["skew"]:.2f}\n'
            f'>2σ: {viability["upper_tail_2"]:.2f}% (Gauss {viability["gauss_tail_2"]:.2f}%)\n'
            f'>3σ: {viability["upper_tail_3"]:.2f}% (Gauss {viability["gauss_tail_3"]:.2f}%)')
    ax_qq.text(0.02, 0.97, info,
               transform=ax_qq.transAxes, va='top', fontsize='x-small', color=flag_color)


def _render_retracement_section(fig, spec, long_retr, short_retr, med_atr20, symbol):
    gs = spec.subgridspec(1, 6, wspace=0.35)
    # Violin — cols 0-2
    ax_v = fig.add_subplot(gs[0, 0:3])
    _style_ax(ax_v)
    dir_colors = {'Long': '#24ad54', 'Short': '#ec4533'}
    vdata, vlabels = [], []
    for lbl, df_r in [('Long', long_retr), ('Short', short_retr)]:
        if not df_r.empty:
            vals = df_r['max_retracement_pct'].abs().dropna().values
            if len(vals) > 1:
                vdata.append(vals); vlabels.append(lbl)
    if vdata:
        parts = ax_v.violinplot(vdata, showmedians=False, showextrema=True,
                                 quantiles=[[0.25, 0.5, 0.75]] * len(vdata))
        for pc, lbl in zip(parts['bodies'], vlabels):
            pc.set_facecolor(dir_colors[lbl]); pc.set_alpha(0.6)
        ax_v.set_xticks(np.arange(1, len(vlabels) + 1))
        ax_v.set_xticklabels(vlabels)
        annotate_violin(ax_v, vdata, np.arange(1, len(vlabels) + 1), vlabels)
    for mult, color, ls in [(0.5, '#f5a623', ':'), (1.0, '#ff6b6b', '--'), (1.5, '#ec4533', '--')]:
        ax_v.axhline(med_atr20 * mult, color=color, linestyle=ls, linewidth=1.2,
                     label=f'{mult:.1f}× ATR20 ({med_atr20 * mult:.1f}%)')
    ax_v.set_title(f'Intra-Trend Retracement Depth — {symbol}')
    ax_v.set_ylabel('Retracement %')
    ax_v.legend(fontsize='x-small')

    # Scatter — cols 3-5
    ax_s = fig.add_subplot(gs[0, 3:6])
    _style_ax(ax_s)
    for lbl, df_r, color in [('Long', long_retr, '#24ad54'), ('Short', short_retr, '#ec4533')]:
        if not df_r.empty:
            ax_s.scatter(df_r['duration'], df_r['max_retracement_pct'].abs(),
                         alpha=0.4, s=15, color=color, label=lbl)
    for mult, color in [(0.5, '#f5a623'), (1.0, '#ff6b6b'), (1.5, '#ec4533')]:
        ax_s.axhline(med_atr20 * mult, color=color, linestyle=':', linewidth=0.8)
    ax_s.set_title('Trend Duration vs Max Retracement')
    ax_s.set_xlabel('Duration (Days)')
    ax_s.set_ylabel('Max Retracement %')
    ax_s.legend(fontsize='x-small')


def _render_hv_impulse_section(fig, spec, hv_series, hv_regime, hv_episodes,
                                hv_transitions, hv_col_used, impulse_df, symbol):
    gs = spec.subgridspec(3, 6, hspace=0.55, wspace=0.35)
    if hv_series is None:
        ax = fig.add_subplot(gs[:, :])
        _style_ax(ax)
        ax.set_title('HV data unavailable — cannot render regime analysis')
        return

    direct_spike = float(hv_transitions.loc['Low', 'High']) \
        if 'Low' in hv_transitions.index else 0.0
    flag_hv = 'SPIKE-PRONE' if direct_spike > 15 else 'MEAN-REVERTING'

    # HV time series with regime shading (full width) — rows 0-1
    ax_ts = fig.add_subplot(gs[0:2, :])
    _style_ax(ax_ts)
    ax_ts.plot(hv_series.index, hv_series.values, color='white', linewidth=0.8, alpha=0.9)
    for reg, color in _REGIME_COLORS.items():
        mask = hv_regime.reindex(hv_series.index, fill_value='Medium') == reg
        ax_ts.fill_between(hv_series.index, 0, hv_series.values,
                            where=mask, color=color, alpha=0.25, label=reg)
    ax_ts.set_title(f'{hv_col_used.upper()} Regime — {symbol}  [{flag_hv}]')
    ax_ts.set_ylabel('Realized Volatility (%)')
    leg = ax_ts.legend(fontsize='x-small', loc='upper left')
    plt.setp(leg.get_texts(), color='white')

    # Episode duration violin — row 2, cols 0-1
    ax_dur = fig.add_subplot(gs[2, 0:2])
    _style_ax(ax_dur)
    vdata, vlabels = [], []
    for reg in ['Low', 'Medium', 'High']:
        vals = hv_episodes[hv_episodes['regime'] == reg]['duration'].dropna().values
        if len(vals) > 1:
            vdata.append(vals); vlabels.append(reg)
    if vdata:
        parts = ax_dur.violinplot(vdata, showmedians=False, showextrema=True,
                                   quantiles=[[0.25, 0.5, 0.75]] * len(vdata))
        for pc, lbl in zip(parts['bodies'], vlabels):
            pc.set_facecolor(_REGIME_COLORS[lbl]); pc.set_alpha(0.6)
        ax_dur.set_xticks(np.arange(1, len(vlabels) + 1))
        ax_dur.set_xticklabels(vlabels)
        annotate_violin(ax_dur, vdata, np.arange(1, len(vlabels) + 1), vlabels)
    ax_dur.set_title('Regime Episode Duration (Days)')
    ax_dur.set_ylabel('Days')

    # Transition heatmap — row 2, cols 2-3
    ax_hm = fig.add_subplot(gs[2, 2:4])
    _style_ax(ax_hm)
    mat = hv_transitions.values.astype(float)
    im  = ax_hm.imshow(mat, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
    ax_hm.set_xticks([0, 1, 2]); ax_hm.set_yticks([0, 1, 2])
    ax_hm.set_xticklabels(['Low', 'Medium', 'High'])
    ax_hm.set_yticklabels(['Low', 'Medium', 'High'])
    ax_hm.set_xlabel('To'); ax_hm.set_ylabel('From')
    ax_hm.set_title('Transition Probabilities (%)')
    for i in range(3):
        for j in range(3):
            val = mat[i, j] if i < mat.shape[0] and j < mat.shape[1] else 0.0
            ax_hm.text(j, i, f'{val:.1f}%', ha='center', va='center',
                       fontsize='small', fontweight='bold',
                       color='black' if val > 50 else 'white')
    fig.colorbar(im, ax=ax_hm, label='Probability (%)')

    # Impulse forward returns (Block 5) — row 2, cols 4-5
    ax_imp = fig.add_subplot(gs[2, 4:6])
    _style_ax(ax_imp)
    if impulse_df is not None and not impulse_df.empty:
        fwd_cols = [c for c in ['fwd_5', 'fwd_10', 'fwd_20'] if c in impulse_df.columns]
        for direction, color in [('up', '#24ad54'), ('down', '#ec4533')]:
            sub = impulse_df[impulse_df['direction'] == direction]
            if sub.empty or not fwd_cols:
                continue
            medians = [sub[c].median() for c in fwd_cols]
            labels  = [c.replace('fwd_', '') + 'd' for c in fwd_cols]
            x = np.arange(len(labels))
            ax_imp.bar(x + (0.2 if direction == 'up' else -0.2), medians, 0.35,
                       color=color, alpha=0.7, label=f'{direction} impulse (n={len(sub)})')
        ax_imp.set_xticks(np.arange(len(fwd_cols)))
        ax_imp.set_xticklabels([c.replace('fwd_', '') + 'd' for c in fwd_cols])
        ax_imp.axhline(0, color='#666', linewidth=0.8, linestyle='--')
        ax_imp.set_title(f'Median Fwd Returns after ±1.75× ATR Impulse')
        ax_imp.set_ylabel('Return %')
        leg = ax_imp.legend(fontsize='x-small')
        plt.setp(leg.get_texts(), color='white')
    else:
        ax_imp.set_title('No impulse sessions found at 1.75× ATR threshold')


def _render_rs_section(fig, spec, df, spy_df, symbol):
    """RS line vs SPY (top) + 63d RS percentile (bottom)."""
    gs = spec.subgridspec(2, 1, hspace=0.4)
    ax_rs = fig.add_subplot(gs[0, 0])
    ax_pct = fig.add_subplot(gs[1, 0], sharex=ax_rs)
    _style_ax(ax_rs); _style_ax(ax_pct)

    if 'c' not in df.columns or spy_df is None or 'c' not in spy_df.columns:
        ax_rs.set_title('RS vs SPY — data unavailable')
        return

    rs_df = calculate_rs_vs_spy(df['c'], spy_df['c'], window=63)
    if rs_df.empty:
        ax_rs.set_title('RS vs SPY — no overlap')
        return

    ax_rs.plot(rs_df.index, rs_df['rs'], color='#48dbfb', linewidth=1.2)
    ax_rs.axhline(1.0, color='#f5a623', linestyle='--', linewidth=0.8, alpha=0.7, label='Baseline')
    ax_rs.set_title(f'Relative Strength vs SPY — {symbol}')
    ax_rs.set_ylabel('RS (norm.)')
    leg = ax_rs.legend(fontsize='x-small')
    plt.setp(leg.get_texts(), color='white')

    ax_pct.plot(rs_df.index, rs_df['rs_pct'], color='#24ad54', linewidth=1.0)
    ax_pct.axhline(50, color='#f5a623', linestyle='--', linewidth=0.8, alpha=0.7)
    ax_pct.axhline(80, color='#24ad54', linestyle=':', linewidth=0.8, alpha=0.5)
    ax_pct.axhline(20, color='#ec4533', linestyle=':', linewidth=0.8, alpha=0.5)
    ax_pct.set_ylim(0, 100)
    ax_pct.set_title('63d RS Percentile')
    ax_pct.set_ylabel('Percentile')


def _render_overnight_reversal_section(fig, spec, df, symbol):
    """PM-08 overnight reversal — distributions, hit rates, rolling edge."""
    gs = spec.subgridspec(1, 6, wspace=0.35)
    ax_dist = fig.add_subplot(gs[0, 0:2])
    ax_hit  = fig.add_subplot(gs[0, 2:4])
    ax_roll = fig.add_subplot(gs[0, 4:6])
    _style_ax(ax_dist); _style_ax(ax_hit); _style_ax(ax_roll)

    required = {'o', 'c', 'atrp20'}
    if not required.issubset(df.columns):
        ax_dist.set_title('Overnight reversal — missing o/c/atrp20')
        return

    res = calculate_overnight_reversal(df, atr_col='atrp20')
    stats = res['stats']
    rolling = res['rolling'].dropna()
    flag = res['flag']
    flag_color = {'REVERSAL': '#24ad54', 'NEUTRAL': '#f5a623', 'CONTINUATION': '#ec4533'}[flag]

    # Overnight return distributions per bucket (violin)
    buckets = list(stats.index)
    overnight = (df['o'].shift(-1) / df['c'] - 1) * 100
    intraday = df['pct'] if 'pct' in df.columns else df['c'].pct_change() * 100
    drop_atr = np.where(intraday < 0, (-intraday) / df['atrp20'], 0.0)
    work = pd.DataFrame({'overnight': overnight, 'drop_atr': drop_atr}).dropna()

    vdata, vlabels = [], []
    for bucket in buckets:
        if bucket == 'Baseline':
            vals = work.loc[work['drop_atr'] < 1.0, 'overnight'].values
        else:
            thresh = float(bucket.replace('Drop>', ''))
            vals = work.loc[work['drop_atr'] >= thresh, 'overnight'].values
        # Clip outliers for display
        if len(vals) > 5:
            lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
            vals = vals[(vals >= lo) & (vals <= hi)]
        if len(vals) > 5:
            vdata.append(vals)
            vlabels.append(bucket)

    if vdata:
        parts = ax_dist.violinplot(vdata, showmedians=False, showextrema=True,
                                    quantiles=[[0.25, 0.5, 0.75]] * len(vdata))
        for pc in parts['bodies']:
            pc.set_facecolor('#48dbfb'); pc.set_alpha(0.6)
        ax_dist.set_xticks(np.arange(1, len(vlabels) + 1))
        ax_dist.set_xticklabels(vlabels, fontsize='x-small')
        annotate_violin(ax_dist, vdata, np.arange(1, len(vlabels) + 1), vlabels)
    ax_dist.axhline(0, color='#666', linewidth=0.6)
    ax_dist.set_title(f'Overnight Returns by Intraday Drop — {symbol}  [{flag}]')
    ax_dist.set_ylabel('Overnight return %')

    # Hit rate bars by bucket
    x = np.arange(len(buckets))
    hit_vals = [stats.loc[b, 'hit_rate'] for b in buckets]
    n_vals   = [stats.loc[b, 'n'] for b in buckets]
    colors   = ['#48dbfb'] + ['#24ad54'] * (len(buckets) - 1)
    ax_hit.bar(x, hit_vals, color=colors, alpha=0.85)
    for xi, hv, nn in zip(x, hit_vals, n_vals):
        if not np.isnan(hv):
            ax_hit.text(xi, hv + 1, f'{hv:.0f}% (n={int(nn)})',
                         ha='center', fontsize='xx-small', color='white')
    ax_hit.set_xticks(x)
    ax_hit.set_xticklabels(buckets, fontsize='x-small')
    ax_hit.axhline(50, color='#f5a623', linestyle=':', linewidth=0.8, alpha=0.7)
    ax_hit.set_ylim(0, 100)
    ax_hit.set_title('Hit rate (% overnight > 0)')
    ax_hit.set_ylabel('%')

    # Rolling edge: drop>2 median − baseline median
    if not rolling.empty:
        ax_roll.plot(rolling.index, rolling.values, color='#48dbfb', linewidth=1.0)
        ax_roll.axhline(0, color='#f5a623', linestyle='--', linewidth=0.8)
        ax_roll.fill_between(rolling.index, 0, rolling.values,
                              where=(rolling.values > 0), color='#24ad54', alpha=0.2)
        ax_roll.fill_between(rolling.index, 0, rolling.values,
                              where=(rolling.values < 0), color='#ec4533', alpha=0.2)
        cur = float(rolling.iloc[-1])
        ax_roll.set_title(f'Rolling 60d edge (Drop>2 − Baseline)  current: {cur:+.2f}%')
        ax_roll.set_ylabel('Median Δ %')
    else:
        ax_roll.set_title('Rolling edge — insufficient data')

    # Colour the main title bar based on flag
    ax_dist.title.set_color(flag_color)


def _render_vcp_section(fig, spec, df, symbol):
    """VCP tightness timeline + histogram + conditional forward returns."""
    gs = spec.subgridspec(1, 6, wspace=0.35)
    ax_ts = fig.add_subplot(gs[0, 0:3])
    ax_hist = fig.add_subplot(gs[0, 3:4])
    ax_cond = fig.add_subplot(gs[0, 4:6])
    _style_ax(ax_ts); _style_ax(ax_hist); _style_ax(ax_cond)

    required = {'h', 'l', 'c', 'v'}
    if not required.issubset(df.columns):
        ax_ts.set_title('VCP — missing OHLCV columns')
        return

    vcp = calculate_vcp_tightness(df, window=10)
    timeline = vcp['timeline'].dropna()
    if timeline.empty:
        ax_ts.set_title('VCP — insufficient history')
        return

    tight = timeline['tightness']
    q33, q66 = tight.quantile([1/3, 2/3]).values
    # Robust display range — clip outliers so scale isn't ruined
    y_lo = max(0.0, float(tight.quantile(0.01)))
    y_hi = float(tight.quantile(0.99))
    cur = vcp['current']

    # Timeline
    ax_ts.plot(tight.index, tight.values, color='#48dbfb', linewidth=0.9)
    ax_ts.axhline(q33, color='#24ad54', linestyle='--', linewidth=0.8, alpha=0.7, label=f'Tight ≤ {q33:.1f}%')
    ax_ts.axhline(q66, color='#ec4533', linestyle='--', linewidth=0.8, alpha=0.7, label=f'Loose > {q66:.1f}%')
    ax_ts.fill_between(tight.index, 0, tight.values, where=(tight <= q33),
                        color='#24ad54', alpha=0.2)
    ax_ts.set_ylim(y_lo, y_hi)
    ax_ts.set_title(
        f'VCP Tightness (10d range/close) — {symbol}   '
        f'current: {cur["tightness"]:.1f}% ({cur["tight_pct_rank"]:.0f}th pct)   '
        f'vol_ratio: {cur["vol_ratio"]:.2f}'
    )
    ax_ts.set_ylabel('Range %')
    leg = ax_ts.legend(fontsize='x-small', loc='upper left')
    plt.setp(leg.get_texts(), color='white')

    # Histogram — clipped to [y_lo, y_hi] so outliers don't flatten the distribution
    tight_clip = tight[(tight >= y_lo) & (tight <= y_hi)]
    ax_hist.hist(tight_clip.values, bins=40, range=(y_lo, y_hi), color='#48dbfb', alpha=0.75)
    ax_hist.set_xlim(y_lo, y_hi)
    ax_hist.axvline(cur['tightness'], color='#f5a623', linestyle='--', linewidth=1.0,
                     label=f'Current: {cur["tightness"]:.1f}%')
    ax_hist.axvline(q33, color='#24ad54', linestyle=':', linewidth=0.8)
    ax_hist.axvline(q66, color='#ec4533', linestyle=':', linewidth=0.8)
    ax_hist.set_title('Tightness distribution')
    ax_hist.set_xlabel('Range %')
    leg = ax_hist.legend(fontsize='x-small')
    plt.setp(leg.get_texts(), color='white')

    # Conditional forward returns by tightness tercile
    cond = vcp['conditional']
    if not cond.empty:
        buckets = ['Tight', 'Mid', 'Loose']
        colors  = {'Tight': '#24ad54', 'Mid': '#f5a623', 'Loose': '#ec4533'}
        width = 0.25
        x = np.arange(len(cond.index))
        for i, bucket in enumerate(buckets):
            vals = cond[bucket].values
            ax_cond.bar(x + (i - 1) * width, vals, width,
                         color=colors[bucket], alpha=0.85, label=bucket)
            for xi, vv in zip(x + (i - 1) * width, vals):
                if not np.isnan(vv):
                    ax_cond.text(xi, vv + 0.05, f'{vv:.1f}',
                                  ha='center', fontsize='xx-small', color='white')
        ax_cond.set_xticks(x)
        ax_cond.set_xticklabels([h.replace('fwd_', '+') + 'd' for h in cond.index])
        ax_cond.axhline(0, color='#666', linewidth=0.6)
        ax_cond.set_title('Median fwd return by tightness tercile')
        ax_cond.set_ylabel('Return %')
        leg = ax_cond.legend(fontsize='x-small')
        plt.setp(leg.get_texts(), color='white')
    else:
        ax_cond.set_title('VCP conditional — insufficient data')


def _render_ivp_section(fig, spec, df, symbol):
    """1y IVP timeline with 50% band."""
    gs = spec.subgridspec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    _style_ax(ax)

    if 'iv' not in df.columns or df['iv'].dropna().empty:
        ax.set_title('IVP — no IV data')
        return

    ivp = calculate_ivp(df['iv'], window=252).dropna()
    if ivp.empty:
        ax.set_title('IVP — insufficient history')
        return

    ax.plot(ivp.index, ivp.values, color='#48dbfb', linewidth=1.0)
    ax.fill_between(ivp.index, 50, ivp.values, where=(ivp.values >= 50),
                     color='#24ad54', alpha=0.2, label='IVP ≥ 50 (DRIFT-eligible)')
    ax.fill_between(ivp.index, 50, ivp.values, where=(ivp.values < 50),
                     color='#ec4533', alpha=0.15)
    ax.axhline(50, color='#f5a623', linestyle='--', linewidth=0.8)
    ax.set_ylim(0, 100)
    current = float(ivp.iloc[-1])
    ax.set_title(f'IV Percentile (1y) — {symbol}   current: {current:.0f}')
    ax.set_ylabel('IVP (%)')
    leg = ax.legend(fontsize='x-small', loc='upper left')
    plt.setp(leg.get_texts(), color='white')


# ---------------------------------------------------------------------------
# Tab: PEAD Clustering (Post-Earnings Announcement Drift)
# ---------------------------------------------------------------------------

def render_pead(fig: Figure, symbol: str = ''):
    """
    Earnings clustering view — PM-02 (PEAD) + PM-03 (Pre-Earnings Anticipation).

    Row 0: All events, cpct[-20..+24], normalised to 0% at t=-1
    Row 1: Beats cluster | Misses cluster
    Row 2: Median drift by surprise direction | Hit rate at +5/+10/+20d
    Row 3: Pre-earnings drift predictive power — does drift from t=-20..-1
           predict a beat? Scatter + conditional hit rate.
    """
    fig.clear()
    fig.patch.set_facecolor('#111111')
    try:
        fig.set_layout_engine('none')
    except Exception:
        pass

    events = load_ticker_earnings_events(symbol) if symbol else pd.DataFrame()

    gs = fig.add_gridspec(
        4, 2, hspace=0.55, wspace=0.25,
        left=0.04, right=0.995, top=0.965, bottom=0.04,
    )
    ax_all    = fig.add_subplot(gs[0, :])
    ax_beat   = fig.add_subplot(gs[1, 0])
    ax_miss   = fig.add_subplot(gs[1, 1])
    ax_med    = fig.add_subplot(gs[2, 0])
    ax_hit    = fig.add_subplot(gs[2, 1])
    ax_pre_sc = fig.add_subplot(gs[3, 0])
    ax_pre_hr = fig.add_subplot(gs[3, 1])
    for ax in [ax_all, ax_beat, ax_miss, ax_med, ax_hit, ax_pre_sc, ax_pre_hr]:
        _style_ax(ax)

    if events.empty:
        ax_all.set_title(f'PEAD — no earnings events for {symbol}')
        return

    t_range = list(range(-20, 25))
    cols = [f'cpct{t}' for t in t_range if f'cpct{t}' in events.columns]
    if not cols:
        ax_all.set_title(f'PEAD — no cpct columns for {symbol}')
        return
    ts = [int(c.replace('cpct', '')) for c in cols]

    paths = events[cols].to_numpy(dtype=float)  # (n_events, n_t)
    # Normalise each row to 0 at t=-1 (the bar before the reaction day)
    if -1 in ts:
        base_idx = ts.index(-1)
        paths = paths - paths[:, [base_idx]]

    beat_mask = (events['surprise_dir'] == 'beat').to_numpy()
    miss_mask = (events['surprise_dir'] == 'miss').to_numpy()

    def _plot_cluster(ax, rows, title, line_color, band_color):
        if rows.shape[0] == 0:
            ax.set_title(f'{title} — no events')
            return
        for row in rows:
            ax.plot(ts, row, color=line_color, alpha=0.15, linewidth=0.8)
        med = np.nanmedian(rows, axis=0)
        q25 = np.nanpercentile(rows, 25, axis=0)
        q75 = np.nanpercentile(rows, 75, axis=0)
        ax.plot(ts, med, color=band_color, linewidth=2.0, label='Median')
        ax.fill_between(ts, q25, q75, color=band_color, alpha=0.25, label='IQR')
        ax.axvline(0, color='#f5a623', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.axhline(0, color='#666', linewidth=0.6)
        ax.set_title(f'{title}  (n={rows.shape[0]})')
        ax.set_xlabel('Trading days from announcement')
        ax.set_ylabel('Return %')
        leg = ax.legend(fontsize='x-small', loc='upper left')
        plt.setp(leg.get_texts(), color='white')

    # Top panel — all events
    _plot_cluster(ax_all, paths, f'All earnings events — {symbol}', '#48dbfb', '#48dbfb')

    # Beat / Miss split
    _plot_cluster(ax_beat, paths[beat_mask], 'Beats',  '#24ad54', '#24ad54')
    _plot_cluster(ax_miss, paths[miss_mask], 'Misses', '#ec4533', '#ec4533')

    # Median comparison
    if paths.shape[0] > 0:
        med_all  = np.nanmedian(paths, axis=0)
        ax_med.plot(ts, med_all, color='#48dbfb', linewidth=1.6, label=f'All (n={paths.shape[0]})')
        if beat_mask.any():
            ax_med.plot(ts, np.nanmedian(paths[beat_mask], axis=0),
                         color='#24ad54', linewidth=1.6, label=f'Beats (n={beat_mask.sum()})')
        if miss_mask.any():
            ax_med.plot(ts, np.nanmedian(paths[miss_mask], axis=0),
                         color='#ec4533', linewidth=1.6, label=f'Misses (n={miss_mask.sum()})')
        ax_med.axvline(0, color='#f5a623', linestyle='--', linewidth=0.8, alpha=0.7)
        ax_med.axhline(0, color='#666', linewidth=0.6)
        ax_med.set_title('Median drift by surprise direction')
        ax_med.set_xlabel('Trading days from announcement')
        ax_med.set_ylabel('Median return %')
        leg = ax_med.legend(fontsize='x-small', loc='upper left')
        plt.setp(leg.get_texts(), color='white')

    # Hit rate at t=+5, +10, +20 (% of events with return > 0)
    horizons = [5, 10, 20]
    horizon_idxs = [ts.index(h) for h in horizons if h in ts]
    if horizon_idxs:
        groups = [('All', paths), ('Beats', paths[beat_mask]), ('Misses', paths[miss_mask])]
        width = 0.25
        x = np.arange(len(horizons))
        colors = {'All': '#48dbfb', 'Beats': '#24ad54', 'Misses': '#ec4533'}
        for i, (label, rows) in enumerate(groups):
            if rows.shape[0] == 0:
                continue
            hit = [(rows[:, idx] > 0).mean() * 100 for idx in horizon_idxs]
            ax_hit.bar(x + (i - 1) * width, hit, width,
                        color=colors[label], alpha=0.8, label=f'{label} (n={rows.shape[0]})')
            for xi, hv in zip(x + (i - 1) * width, hit):
                ax_hit.text(xi, hv + 1, f'{hv:.0f}', ha='center', fontsize='xx-small', color='white')
        ax_hit.set_xticks(x)
        ax_hit.set_xticklabels([f'+{h}d' for h in horizons])
        ax_hit.axhline(50, color='#f5a623', linestyle=':', linewidth=0.8, alpha=0.7)
        ax_hit.set_ylim(0, 100)
        ax_hit.set_title('Hit Rate (% positive) by horizon')
        ax_hit.set_ylabel('% events > 0')
        leg = ax_hit.legend(fontsize='x-small', loc='upper right')
        plt.setp(leg.get_texts(), color='white')

    # -------- Pre-earnings anticipation (PM-03) --------
    # Pre-drift = return from t=-20 to t=-1 (end of the day before announcement).
    # Post-drift = return at t=+5 (typical PEAD window).
    if all(t in ts for t in [-20, -1, 5]):
        i_pre0  = ts.index(-20)
        i_pre1  = ts.index(-1)
        i_post  = ts.index(5)
        pre  = paths[:, i_pre1] - paths[:, i_pre0]
        post = paths[:, i_post] - paths[:, i_pre1]

        # Scatter: pre-drift (x) vs post-drift (y), colour by beat/miss
        if beat_mask.any():
            ax_pre_sc.scatter(pre[beat_mask], post[beat_mask], s=25, alpha=0.7,
                               color='#24ad54', label=f'Beat (n={beat_mask.sum()})')
        if miss_mask.any():
            ax_pre_sc.scatter(pre[miss_mask], post[miss_mask], s=25, alpha=0.7,
                               color='#ec4533', label=f'Miss (n={miss_mask.sum()})')
        unknown_mask = ~(beat_mask | miss_mask)
        if unknown_mask.any():
            ax_pre_sc.scatter(pre[unknown_mask], post[unknown_mask], s=15, alpha=0.4,
                               color='#aaaaaa', label=f'Unknown (n={unknown_mask.sum()})')
        ax_pre_sc.axvline(0, color='#666', linewidth=0.6)
        ax_pre_sc.axhline(0, color='#666', linewidth=0.6)
        ax_pre_sc.set_title('Pre-earnings drift (t-20→t-1) vs Post (t-1→t+5)')
        ax_pre_sc.set_xlabel('Pre-earnings return %')
        ax_pre_sc.set_ylabel('Post (+5d) return %')
        leg = ax_pre_sc.legend(fontsize='x-small', loc='upper left')
        plt.setp(leg.get_texts(), color='white')

        # Conditional: does positive pre-drift predict a beat and/or positive +5d?
        # Bars for: P(beat | pre>0), P(beat | pre<0), P(+5d>0 | pre>0), P(+5d>0 | pre<0)
        bars = []
        labels = []
        colors = []
        pre_pos = pre > 0
        pre_neg = pre < 0
        known   = beat_mask | miss_mask
        def _rate(m, v):
            m = m & v
            return (beat_mask[m].mean() * 100) if m.any() else np.nan

        if (pre_pos & known).any():
            labels.append('Beat | Pre>0'); bars.append(_rate(known, pre_pos)); colors.append('#24ad54')
        if (pre_neg & known).any():
            labels.append('Beat | Pre<0'); bars.append(_rate(known, pre_neg)); colors.append('#24ad54')
        if pre_pos.any():
            labels.append('+5d>0 | Pre>0')
            bars.append((post[pre_pos] > 0).mean() * 100)
            colors.append('#48dbfb')
        if pre_neg.any():
            labels.append('+5d>0 | Pre<0')
            bars.append((post[pre_neg] > 0).mean() * 100)
            colors.append('#48dbfb')

        if bars:
            x = np.arange(len(bars))
            ax_pre_hr.bar(x, bars, color=colors, alpha=0.85)
            for xi, bv in zip(x, bars):
                if not np.isnan(bv):
                    ax_pre_hr.text(xi, bv + 1, f'{bv:.0f}', ha='center', fontsize='xx-small', color='white')
            ax_pre_hr.set_xticks(x)
            ax_pre_hr.set_xticklabels(labels, fontsize='x-small', rotation=15)
            ax_pre_hr.axhline(50, color='#f5a623', linestyle=':', linewidth=0.8, alpha=0.7)
            ax_pre_hr.set_ylim(0, 100)
            ax_pre_hr.set_title('Pre-earnings drift → outcome (%)')
            ax_pre_hr.set_ylabel('% probability')
