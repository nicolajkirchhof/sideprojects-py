"""
finance.utils._dormant.mpl_plots
==================================
Legacy matplotlib plotting functions.
Preserved for future migration to native Qt or reactivation with InfluxDB.
"""
import os
import re
from datetime import datetime

import mplfinance as mpf
import numpy as np
import pandas as pd
from matplotlib import gridspec, pyplot as plt
import matplotlib.ticker as mticker

from finance.utils._dormant.trading_day_data import TradingDayData
from finance.utils._dormant import influx
from finance.utils.chart_styles import (
    MPL_EMA_CONFIGS as EMA_CONFIGS,
    MPL_ATR_CONFIGS as ATR_CONFIGS,
    MPL_AC_CONFIGS as AC_CONFIGS,
    MPL_AC_REGIME_CONFIGS as AC_REGIME_CONFIGS,
    MPL_SLOPE_CONFIGS as SLOPE_CONFIGS,
    MPL_HURST_CONFIGS as HURST_CONFIGS,
    MPL_HV_CONFIGS as HV_CONFIGS,
)


def plot_multi_pane_mpl(df, symbol, ref_df=None, vlines=dict(vlines=[], colors=[]), fig=None):
    panes = 7 if ref_df is None else 8

    # 1. Setup Figure and Grid (8 Panes matching pyqtgraph)
    if fig is None:
        fig = mpf.figure(figsize=(24, 24))
        gs = gridspec.GridSpec(panes, 1, height_ratios=[3]+[1]*(panes-1))
        axes = [fig.add_subplot(gs[i]) for i in range(8)]
    else:
        axes = fig.get_axes()
        for ax in axes: ax.clear()

    # 2. Pane 1: Price + EMAs (using mplfinance)
    ap = []
    for col, color in EMA_CONFIGS.items():
        if col in df.columns:
            ap.append(mpf.make_addplot(df[col], ax=axes[0], color=color, width=1.0))

    mpf.plot(df, type='ohlc', columns=influx.MPF_COLUMN_MAPPING, volume=axes[1], style='yahoo', ax=axes[0], addplot=ap, vlines=vlines, datetime_format='%y-%m-%d')
    if ref_df is not None and len(ref_df) == len(df):
        mpf.plot(ref_df, type='ohlc', columns=influx.MPF_COLUMN_MAPPING, style='yahoo', ax=axes[7], addplot=ap, vlines=vlines, datetime_format='%y-%m-%d')
    axes[0].set_title(f"{symbol} Multi-Pane Analysis", fontsize=16)

    # 4. Indicator Panes (Panes 3-8)
    config_map = [
        (axes[2], ATR_CONFIGS, 'ATR %'),
        (axes[3], AC_CONFIGS, 'AutoCorr'),
        (axes[4], HURST_CONFIGS, 'Hurst'),
        (axes[5], AC_REGIME_CONFIGS, 'AC Regimes'),
        (axes[6], HV_CONFIGS, 'IV/HV')
    ]

    for ax, config, label in config_map:
        for col, cfg in config.items():
            if col in df.columns:
                ax.plot(df.index, df[col], color=cfg if isinstance(cfg, str) else cfg['color'],
                        lw=cfg.get('width', 1.0) if isinstance(cfg, dict) else 1.0, label=col)

        # Zero/Baseline levels
        if label in ['AutoCorr', 'EMA Slope', 'AC Regimes']: ax.axhline(0, color='#666', lw=0.8, ls='--')
        if label == 'ATR %': ax.axhline(1.2, color='#666', lw=0.8, ls='--')
        if label == 'Hurst': ax.axhline(0.5, color='#666', lw=0.8, ls='--')

        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def daily_change_plot(day_data: TradingDayData, alines=None, title_add='', atr_vlines=dict(vlines=[], colors=[]),
                      ad=True, basetime='5m'):
    if basetime == '5m':
        df_base = day_data.df_5m_ad if ad else day_data.df_5m
    elif basetime == '10m':
        df_base = day_data.df_10m_ad if ad else day_data.df_10m
    elif basetime == '15m':
        df_base = day_data.df_15m_ad if ad else day_data.df_15m
    df_30m = day_data.df_30m_ad if ad else day_data.df_30m

    fig = mpf.figure(style='yahoo', figsize=(19, 11), tight_layout=True)
    gs = gridspec.GridSpec(3, 2, height_ratios=[2, 0.25, 1], width_ratios=[1, 2])

    date_str = day_data.day_start.strftime('%a, %Y-%m-%d')
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[2, 0])
    ax3 = fig.add_subplot(gs[2, 1])
    ax4 = fig.add_subplot(gs[1, :])

    indicator_hlines = [day_data.cdc, day_data.pdc, day_data.pdh, day_data.pdl,
                        day_data.onh, day_data.onl, day_data.cwh, day_data.cwl, day_data.pwh, day_data.pwl]
    fig.suptitle(
        f'{day_data.symbol} {date_str} || O {day_data.cdo:.2f} H {day_data.cdh:.2f} C {day_data.cdc:.2f} L {day_data.cdl:.2f} || On: H {day_data.onh:.2f} L {day_data.onl:.2f} \n' +
        f'PD: H {day_data.pdh:.2f} C {day_data.pdc:.2f} L {day_data.pdl:.2f} || ' +
        f'CW: H {day_data.cwh:.2f} L {day_data.cwl:.2f} || PW: H {day_data.pwh:.2f} L {day_data.pwl:.2f} || {title_add}')

    hlines = dict(hlines=indicator_hlines, colors=['deeppink'] + ['#bf42f5'] * 5 + ['#3179f5'] * 4,
                  linewidths=[0.4] * 1 + [0.6] * 3 + [0.4] * 6,
                  linestyle=['--'] * 2 + ['-'] * (len(indicator_hlines) - 1))
    hlines_day = dict(hlines=indicator_hlines[1:], colors=['#bf42f5'] * 5 + ['#3179f5'] * 4,
                      linewidths=[0.6] * 3 + [0.4] * 6, linestyle=['--'] + ['-'] * (len(indicator_hlines) - 1))
    vlines = dict(vlines=[day_data.day_open, day_data.day_close], alpha=[0.2], colors=['deeppink'] * 2, linewidths=[1],
                  linestyle=['--'])

    ind_5m_ema20_plot = mpf.make_addplot(df_base['20EMA'], ax=ax1, width=0.6, color="#FF9900", linestyle='--')
    ind_5m_ema240_plot = mpf.make_addplot(df_base['200EMA'], ax=ax1, width=0.6, color='#0099FF', linestyle='--')
    ind_vwap3_plot = mpf.make_addplot(df_base['VWAP3'], ax=ax1, width=2, color='turquoise')

    ind_30m_ema20_plot = mpf.make_addplot(df_30m['20EMA'], ax=ax3, width=0.6, color="#FF9900", linestyle='--')

    ind_day_ema20_plot = mpf.make_addplot(day_data.df_day['20EMA'], ax=ax2, width=0.6, color="#FF9900", linestyle='--')

    mpf.plot(df_base, type='candle', ax=ax1, columns=influx.MPF_COLUMN_MAPPING, xrotation=0,
             datetime_format='%H:%M', tight_layout=True,
             scale_width_adjustment=dict(candle=1.35), hlines=hlines, alines=alines, vlines=vlines,
             addplot=[ind_5m_ema20_plot, ind_5m_ema240_plot, ind_vwap3_plot])
    mpf.plot(df_base, type='line', ax=ax4, columns=['lh'] * 5, xrotation=0, datetime_format='%H:%M', vlines=atr_vlines,
             tight_layout=True)

    mpf.plot(day_data.df_day, type='candle', ax=ax2, columns=influx.MPF_COLUMN_MAPPING, xrotation=0,
             datetime_format='%m-%d', tight_layout=True,
             hlines=hlines_day, warn_too_much_data=700, addplot=[ind_day_ema20_plot])
    mpf.plot(df_30m, type='candle', ax=ax3, columns=influx.MPF_COLUMN_MAPPING, xrotation=0, datetime_format='%H:%M',
             tight_layout=True,
             scale_width_adjustment=dict(candle=1.35), hlines=hlines, addplot=[ind_30m_ema20_plot])

    # Use MaxNLocator to increase the number of ticks
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(nbins=20))
    ax4.xaxis.set_major_locator(mticker.MaxNLocator(nbins=20))
    ax1.yaxis.set_major_locator(mticker.MaxNLocator(nbins=10))
    plt.tight_layout(h_pad=0.1)


def last_date_from_files(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    files_sorted = sorted(files, reverse=True)
    first_file = files_sorted[0] if files_sorted else None
    if first_file is not None:
        date_pattern = r'\d{4}-\d{2}-\d{2}'
        match = re.search(date_pattern, first_file)
        if match:
            date_str = match.group()
            parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
            print(f"Date string: {date_str}")
            print(f"Parsed date: {parsed_date}")
            return parsed_date
        else:
            print("No date found in filename.")
    return None


def heatmap(df_corr, mask=None, name='Correlation Matrix', ax=None):
    fig, ax = plt.subplots(figsize=(24, 14)) if ax is None else (ax.figure, ax)
    im = ax.imshow(df_corr, cmap='RdYlGn', vmin=-1, vmax=1)

    cols = df_corr.columns
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticklabels(cols)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(cols)):
        for j in range(len(cols)):
            if not mask[i, j]:
                val = df_corr.iloc[i, j]
                ax.text(j, i, f"{val:.2f}",
                        ha="center", va="center", color="black", fontsize=9)

    ax.set_title(name)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.colorbar(im, ax=ax, label='Correlation Coefficient')
    fig.tight_layout()


def boxplot_columns_with_labels(
    df,
    whis=1.5,
    showfliers=False,
    figsize=(23, 12),
    rotate=45,
    fmt="{:.2f}",
    text_kwargs=None,
    title='',
    ax=None
):
    text_kwargs = text_kwargs or {}
    cols = df.columns.tolist()
    data = [df[c].dropna().to_numpy(dtype=float) for c in cols]

    fig, ax = plt.subplots(figsize=figsize) if ax is None else (ax.figure, ax)
    ax.boxplot(data, tick_labels=cols, showfliers=showfliers, whis=whis)

    ax.axhline(0, color="gray", linewidth=1)
    ax.set_title("Distribution of values")
    ax.grid(True, axis="y", alpha=0.25)
    plt.setp(ax.get_xticklabels(), rotation=rotate, ha="right")

    for i, y in enumerate(data, start=1):
        y = y[np.isfinite(y)]
        if y.size == 0:
            continue

        q1, med, q3 = np.percentile(y, [25, 50, 75])
        iqr = q3 - q1
        lo_fence = q1 - whis * iqr
        hi_fence = q3 + whis * iqr

        in_lo = y[y >= lo_fence]
        in_hi = y[y <= hi_fence]
        wlo = np.min(in_lo) if in_lo.size else np.min(y)
        whi = np.max(in_hi) if in_hi.size else np.max(y)

        x = i
        bbox = dict(facecolor="white", edgecolor="none", alpha=0.75)

        ax.text(x, whi, f"whi={fmt.format(whi)}", va="bottom", ha="left", fontsize=8, bbox=bbox, **text_kwargs)
        ax.text(x, q3, f"q3={fmt.format(q3)}", va="bottom", ha="left", fontsize=8, bbox=bbox, **text_kwargs)
        ax.text(x, med, f"m ={fmt.format(med)}", va="center", ha="left", fontsize=8, bbox=bbox, **text_kwargs)
        ax.text(x, q1, f"q1={fmt.format(q1)}", va="top", ha="left", fontsize=8, bbox=bbox, **text_kwargs)
        ax.text(x, wlo, f"wlo={fmt.format(wlo)}", va="top", ha="left", fontsize=8, bbox=bbox, **text_kwargs)

    plt.tight_layout()
    return fig, ax
