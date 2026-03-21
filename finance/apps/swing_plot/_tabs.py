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
from finance.utils.move_character import (
    calculate_regime_filter_stats,
    calculate_move_magnitude_stats,
    calculate_intratrend_retracement,
    calculate_gap_intraday_decomposition,
    calculate_hv_regime_stats,
    calculate_impulse_forward_returns,
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

def render_volatility_analysis(fig: Figure, df: pd.DataFrame, start_year: int, end_year: int):
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

    df_vol['iv_change_pct'] = df_vol['iv'].pct_change() * 100

    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])
    ax_iv_dyn  = fig.add_subplot(gs[0, 0])
    ax_premium = fig.add_subplot(gs[0, 1])
    ax_hv_time = fig.add_subplot(gs[1, :])
    ax_hv14    = fig.add_subplot(gs[2, 0])
    ax_hv20    = fig.add_subplot(gs[2, 1])

    for ax in [ax_iv_dyn, ax_premium, ax_hv_time, ax_hv14, ax_hv20]:
        _style_ax(ax)

    # IV dynamics scatter
    ax_iv_dyn.scatter(df_vol['pct'], df_vol['iv_change_pct'],
                      c=np.abs(df_vol['pct']), cmap='Oranges', alpha=0.5)
    ax_iv_dyn.axhline(0, color='white', lw=1, alpha=0.5)
    ax_iv_dyn.axvline(0, color='white', lw=1, alpha=0.5)
    ax_iv_dyn.set_title('IV Change % vs Price Move %')
    ax_iv_dyn.set_xlabel('Underlying Pct Change')
    ax_iv_dyn.set_ylabel('IV Pct Change')

    # IV–HV premium distributions
    hv_cols = [c for c in ['hvc', 'hv14', 'hv20', 'hv50'] if c in df_vol.columns]
    if hv_cols:
        df_prem = pd.DataFrame({f'IV-{col}': df_vol['iv'] - df_vol[col] for col in hv_cols})
        violinplot_columns_with_labels(df_prem, ax=ax_premium,
                                        title='IV − HV Premium Distributions')

    # IV vs HV time series
    ax_hv_time.plot(df_vol.index, df_vol['iv'], label='IV', color='cyan', lw=2)
    for col in hv_cols:
        ax_hv_time.plot(df_vol.index, df_vol[col], label=col, alpha=0.6, lw=1)
    ax_hv_time.set_title('IV vs Realized Volatilities')
    leg = ax_hv_time.legend(ncol=len(hv_cols) + 1, loc='upper left', framealpha=0.3)
    plt.setp(leg.get_texts(), color='white')

    # IV vs HV14 scatter
    if 'hv14' in df_vol.columns:
        max_v = max(df_vol['iv'].max(), df_vol['hv14'].max())
        ax_hv14.scatter(df_vol['hv14'], df_vol['iv'], alpha=0.3, color='tab:blue')
        ax_hv14.plot([0, max_v], [0, max_v], 'r--', alpha=0.6, label='IV=HV14')
        ax_hv14.set_title('IV vs HV14 (Short-term)')
        ax_hv14.set_xlabel('HV14')
        ax_hv14.set_ylabel('IV')
        leg = ax_hv14.legend(framealpha=0.3)
        plt.setp(leg.get_texts(), color='white')

    # IV vs HV20 scatter
    if 'hv20' in df_vol.columns:
        max_v = max(df_vol['iv'].max(), df_vol['hv20'].max())
        ax_hv20.scatter(df_vol['hv20'], df_vol['iv'], alpha=0.3, color='tab:green')
        ax_hv20.plot([0, max_v], [0, max_v], 'r--', alpha=0.6, label='IV=HV20')
        ax_hv20.set_title('IV vs HV20 (Standard)')
        ax_hv20.set_xlabel('HV20')
        ax_hv20.set_ylabel('IV')
        leg = ax_hv20.legend(framealpha=0.3)
        plt.setp(leg.get_texts(), color='white')

    fig.tight_layout()


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

    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1.5, 1.5, 1.5])
    ax_scatter = fig.add_subplot(gs[0])
    ax_short   = fig.add_subplot(gs[1])
    ax_med     = fig.add_subplot(gs[2])
    ax_long    = fig.add_subplot(gs[3])

    for ax in [ax_scatter, ax_short, ax_med, ax_long]:
        _style_ax(ax)

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


def render_move_character(fig: Figure, df: pd.DataFrame, symbol: str = ''):
    """
    Render the Move Character tab.  5-section layout:
      Section 0 — MA20 Regime Filter stats
      Section 1 — Move Magnitude Distribution (Block 1)
      Section 2 — Intra-Trend Retracement Depth (Block 2)
      Section 3 — Gap vs Intraday Range Decomposition (Block 3)
      Section 4 — HV Regime + Impulse Forward Returns (Blocks 4 + 5)
    """
    fig.clear()
    fig.patch.set_facecolor('#111111')

    # ---- Section 0: Regime Filter ----------------------------------------
    try:
        episodes_df, regime_summary = calculate_regime_filter_stats(df)
        _render_regime_section(fig, episodes_df, regime_summary, symbol)
    except Exception as e:
        ax = fig.add_subplot(10, 1, 1)
        _style_ax(ax)
        ax.set_title(f'Regime Filter: {e}')

    # ---- Section 1: Move Magnitude ----------------------------------------
    try:
        norm_moves, tail_freq, excess_kurtosis, skewness = calculate_move_magnitude_stats(df)
        flag_move = 'EXPLOSIVE' if tail_freq[2.0]['total'] > 5.0 else 'GRADUAL'
        _render_move_magnitude_section(fig, df, norm_moves, tail_freq,
                                        excess_kurtosis, skewness, flag_move, symbol)
    except Exception as e:
        ax = fig.add_subplot(10, 1, 4)
        _style_ax(ax)
        ax.set_title(f'Move Magnitude: {e}')

    # ---- Section 2: Intra-Trend Retracement -------------------------------
    try:
        long_retr, short_retr = calculate_intratrend_retracement(df)
        med_atr20 = df['atrp20'].median() if 'atrp20' in df.columns else 1.0
        _render_retracement_section(fig, long_retr, short_retr, med_atr20, symbol)
    except Exception as e:
        ax = fig.add_subplot(10, 1, 6)
        _style_ax(ax)
        ax.set_title(f'Retracement: {e}')

    # ---- Section 3: Gap Decomposition -------------------------------------
    try:
        gap_stats, rolling_gap = calculate_gap_intraday_decomposition(df)
        mean_gap = gap_stats['gap_contrib'].mean() * 100
        flag_gap = 'GAP-HEAVY' if mean_gap > 40 else 'INTRADAY-DRIVEN'
        fill_rate = gap_stats.loc[gap_stats['gap_dir'] != 0, 'gap_filled'].mean() * 100
        _render_gap_section(fig, gap_stats, rolling_gap, mean_gap, fill_rate, flag_gap, symbol)
    except Exception as e:
        ax = fig.add_subplot(10, 1, 8)
        _style_ax(ax)
        ax.set_title(f'Gap Decomposition: {e}')

    # ---- Section 4: HV Regime + Impulse Forward Returns ------------------
    try:
        hv_series, hv_regime, hv_episodes, hv_transitions, hv_col_used = \
            calculate_hv_regime_stats(df)
        impulse_df = calculate_impulse_forward_returns(df)
        _render_hv_impulse_section(fig, hv_series, hv_regime, hv_episodes,
                                    hv_transitions, hv_col_used, impulse_df, symbol)
    except Exception as e:
        ax = fig.add_subplot(10, 1, 10)
        _style_ax(ax)
        ax.set_title(f'HV Regime / Impulse: {e}')

    fig.tight_layout(rect=[0, 0, 1, 1])


# ---- sub-renderers (one per section) ----------------------------------------

def _render_regime_section(fig, episodes_df, regime_summary, symbol):
    """Row 0: 3 summary bars + 2 duration/depth violins + 3 forward-return violins."""
    gs = fig.add_gridspec(10, 6, hspace=0.55, wspace=0.35)
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


def _render_move_magnitude_section(fig, df, norm_moves, tail_freq,
                                    excess_kurtosis, skewness, flag_move, symbol):
    gs = fig.add_gridspec(10, 3, hspace=0.55, wspace=0.35)

    # Histogram
    ax_hist = fig.add_subplot(gs[3, 0])
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

    # Tail frequency bars
    ax_tail = fig.add_subplot(gs[3, 1])
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

    # QQ plot
    ax_qq = fig.add_subplot(gs[3, 2])
    _style_ax(ax_qq)
    pct_raw = df['pct'].dropna().values
    mean_v, std_v = pct_raw.mean(), pct_raw.std(ddof=1)
    std_sorted = np.sort((pct_raw - mean_v) / std_v)
    n_qq = len(std_sorted)
    theoretical_q = _scipy_norm.ppf((np.arange(1, n_qq + 1) - 0.5) / n_qq)
    ax_qq.scatter(theoretical_q, std_sorted, s=3, alpha=0.4, color='#48dbfb')
    ref = np.array([theoretical_q[0], theoretical_q[-1]])
    ax_qq.plot(ref, ref, color='#f5a623', linewidth=1.5, linestyle='--', label='Normal')
    ax_qq.set_title('QQ Plot — Returns vs Normal')
    ax_qq.set_xlabel('Theoretical Quantiles')
    ax_qq.set_ylabel('Empirical Quantiles')
    ax_qq.legend(fontsize='x-small')
    ax_qq.text(0.02, 0.97, f'Excess kurtosis: {excess_kurtosis:.2f}',
               transform=ax_qq.transAxes, va='top', fontsize='x-small', color='#aaa')


def _render_retracement_section(fig, long_retr, short_retr, med_atr20, symbol):
    gs = fig.add_gridspec(10, 2, hspace=0.55, wspace=0.35)

    # Violin
    ax_v = fig.add_subplot(gs[4, 0])
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

    # Scatter
    ax_s = fig.add_subplot(gs[4, 1])
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


def _render_gap_section(fig, gap_stats, rolling_gap, mean_gap, fill_rate, flag_gap, symbol):
    gs = fig.add_gridspec(10, 2, hspace=0.55, wspace=0.35)

    # Gap contribution histogram
    ax_hist = fig.add_subplot(gs[6, 0])
    _style_ax(ax_hist)
    ax_hist.hist(gap_stats['gap_contrib'].dropna(), bins=40, color='#48dbfb', alpha=0.7, density=True)
    ax_hist.axvline(0.4, color='#f5a623', linestyle='--', linewidth=1.5, label='Gap-heavy (40%)')
    ax_hist.axvline(gap_stats['gap_contrib'].mean(), color='white', linestyle=':', linewidth=1.2,
                    label=f'Mean: {gap_stats["gap_contrib"].mean():.2f}')
    ax_hist.set_title(f'Gap Contribution — {symbol}  [{flag_gap}]')
    ax_hist.set_xlabel('Gap / True Range')
    ax_hist.set_ylabel('Density')
    ax_hist.legend(fontsize='x-small')
    pct_dom = (gap_stats['gap_contrib'] > 0.5).mean() * 100
    ax_hist.text(0.97, 0.97, f'{pct_dom:.1f}% gap-dominated',
                 transform=ax_hist.transAxes, ha='right', va='top', fontsize='x-small', color='#f5a623')

    # Overnight gap vs intraday range scatter
    ax_sc = fig.add_subplot(gs[6, 1])
    _style_ax(ax_sc)
    for mask, color, label in [
        (gap_stats['gap_dir'] > 0,  '#24ad54', 'Gap up'),
        (gap_stats['gap_dir'] < 0,  '#ec4533', 'Gap down'),
        (gap_stats['gap_dir'] == 0, '#aaaaaa', 'Flat open'),
    ]:
        ax_sc.scatter(gap_stats.loc[mask, 'overnight_gap_pct'],
                      gap_stats.loc[mask, 'intraday_range_pct'],
                      s=5, alpha=0.3, color=color, label=label)
    ax_sc.set_title('Overnight Gap vs Intraday Range')
    ax_sc.set_xlabel('Overnight Gap %')
    ax_sc.set_ylabel('Intraday Range %')
    ax_sc.legend(fontsize='x-small', markerscale=3)
    ax_sc.text(0.02, 0.97, f'Fill rate: {fill_rate:.1f}%',
               transform=ax_sc.transAxes, va='top', fontsize='x-small', color='#aaa')

    # Rolling gap timeline (full width)
    ax_roll = fig.add_subplot(gs[7, :])
    _style_ax(ax_roll)
    valid_roll = rolling_gap.dropna()
    ax_roll.plot(valid_roll.index, valid_roll.values * 100, color='#48dbfb', linewidth=1.2)
    ax_roll.axhline(40, color='#f5a623', linestyle='--', linewidth=1, label='40% threshold')
    ax_roll.fill_between(valid_roll.index, valid_roll.values * 100, 40,
                          where=(valid_roll.values * 100 > 40),
                          color='#f5a623', alpha=0.2, label='Gap-heavy periods')
    ax_roll.set_title('Rolling 63-Day Gap Contribution (%)')
    ax_roll.set_ylabel('Gap Contribution %')
    ax_roll.legend(fontsize='x-small')


def _render_hv_impulse_section(fig, hv_series, hv_regime, hv_episodes,
                                hv_transitions, hv_col_used, impulse_df, symbol):
    gs = fig.add_gridspec(10, 3, hspace=0.55, wspace=0.35)

    if hv_series is None:
        ax = fig.add_subplot(gs[8:10, :])
        _style_ax(ax)
        ax.set_title('HV data unavailable — cannot render regime analysis')
        return

    direct_spike = float(hv_transitions.loc['Low', 'High']) \
        if 'Low' in hv_transitions.index else 0.0
    flag_hv = 'SPIKE-PRONE' if direct_spike > 15 else 'MEAN-REVERTING'

    # HV time series with regime shading (full width)
    ax_ts = fig.add_subplot(gs[8, :])
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

    # Episode duration violin
    ax_dur = fig.add_subplot(gs[9, 0])
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

    # Transition heatmap
    ax_hm = fig.add_subplot(gs[9, 1])
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

    # Impulse forward returns (Block 5)
    ax_imp = fig.add_subplot(gs[9, 2])
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
