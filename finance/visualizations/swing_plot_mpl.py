import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mplfinance as mpf
from networkx.algorithms.cuts import volume

from finance import utils
#%%
# Reusing your configurations for visual consistency
EMA_CONFIGS = {'ema10': '#f5deb3', 'ema20': '#e2b46d', 'ema50': '#c68e17', 'ema100': '#8b5a2b', 'ema200': '#4b3621', 'vwap3': '#00bfff'}
ATR_CONFIGS = {'atrp9': '#b0c4de', 'atrp14': '#4682b4', 'atrp20': '#000080'}
AC_CONFIGS = {'ac100_lag_1': '#e0ffff', 'ac100_lag_5': '#00ced1', 'ac100_lag_10': '#00bfff', 'ac100_lag_20': '#008080'}
AC_REGIME_CONFIGS = {'ac_mom': '#e1bee7', 'ac_mr': '#ba68c8', 'ac_comp': '#4a148c'}
SLOPE_CONFIGS = {'ema10_slope': '#ffccbc', 'ema20_slope': '#ff8a65', 'ema50_slope': '#ff5722', 'ema100_slope': '#e64a19', 'ema200_slope': '#bf360c'}
HURST_CONFIGS = {'hurst50': '#fff59d', 'hurst100': '#fbc02d'}
HV_CONFIGS = {'hv9': '#a5d6a7', 'hv14': '#66bb6a', 'hv20': '#2e7d32', 'iv': '#ff00ff'}

def plot_multi_pane_mpl(df, symbol, ref_df):
  panes = 7 if ref_df is None else 8

  # 1. Setup Figure and Grid (8 Panes matching pyqtgraph)
  fig = mpf.figure(figsize=(24, 24))
  gs = gridspec.GridSpec(panes, 1, height_ratios=[3]+[1]*(panes-1))

  axes = [fig.add_subplot(gs[i]) for i in range(8)]
  # for i in range(7): axes[i].sharex(axes[7]) # Link X-axes

  # 2. Pane 1: Price + EMAs (using mplfinance)
  ap = []
  for col, color in EMA_CONFIGS.items():
    if col in df.columns:
      ap.append(mpf.make_addplot(df[col], ax=axes[0], color=color, width=1.0))

  mpf.plot(df, type='ohlc', columns=utils.influx.MPF_COLUMN_MAPPING, volume=axes[1], style='yahoo', ax=axes[0], addplot=ap, datetime_format='%y-%m-%d')
  if ref_df:
    mpf.plot(ref_df, type='ohlc', columns=utils.influx.MPF_COLUMN_MAPPING, volume=axes[7], style='yahoo', ax=axes[0], addplot=ap, datetime_format='%y-%m-%d')
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

    # Zero/Baseline levels matching @thisFile
    if label in ['AutoCorr', 'EMA Slope', 'AC Regimes']: ax.axhline(0, color='#666', lw=0.8, ls='--')
    if label == 'ATR %': ax.axhline(1.2, color='#666', lw=0.8, ls='--')
    if label == 'Hurst': ax.axhline(0.5, color='#666', lw=0.8, ls='--')

    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)
    # Optional: ax.legend(loc='upper left', fontsize=8)

  plt.tight_layout()
  plt.show()

#%%
# Example usage:
# df = utils.indicators.swing_indicators(utils.barchart_data.daily_w_volatility('SPY'))
# data_spy = utils.swing_trading_data.SwingTradingData('SPY', offline=True)
plot_multi_pane_mpl(data_spy.df_day[-200:-1], 'SPY')
# plot_multi_pane_mpl(data_spy.df_week[-200:-1], 'SPY')
