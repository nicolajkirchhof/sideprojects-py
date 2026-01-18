import os
import re
from datetime import timedelta, datetime

import mplfinance as mpf
import numpy as np
import networkx as nx
import pandas as pd
from matplotlib import gridspec, pyplot as plt

from finance import utils
import matplotlib.ticker as mticker

import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui
from pyqtgraph.Qt import QtWidgets
import pyqtgraph.exporters


# Reusing your configurations for visual consistency
EMA_CONFIGS = {'ema10': '#f5deb3', 'ema20': '#e2b46d', 'ema50': '#c68e17', 'vwap3': '#00bfff'} #'ema100': '#8b5a2b', 'ema200': '#4b3621',
ATR_CONFIGS = {'atrp9': '#b0c4de', 'atrp14': '#4682b4', 'atrp20': '#000080'}
AC_CONFIGS = {'ac100_lag_1': '#e0ffff', 'ac100_lag_5': '#00ced1', 'ac100_lag_10': '#00bfff', 'ac100_lag_20': '#008080'}
AC_REGIME_CONFIGS = {'ac_mom': '#e1bee7', 'ac_mr': '#ba68c8', 'ac_comp': '#4a148c'}
SLOPE_CONFIGS = {'ema10_slope': '#ffccbc', 'ema20_slope': '#ff8a65', 'ema50_slope': '#ff5722', 'ema100_slope': '#e64a19', 'ema200_slope': '#bf360c'}
HURST_CONFIGS = {'hurst50': '#fff59d', 'hurst100': '#fbc02d'}
HV_CONFIGS = {'hv9': '#a5d6a7', 'hv14': '#66bb6a', 'hv30': '#2e7d32', 'iv': '#ff00ff'}




def plot_multi_pane_mpl(df, symbol, ref_df = None, vlines=dict(vlines=[], colors=[]), fig=None):
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

  mpf.plot(df, type='ohlc', columns=utils.influx.MPF_COLUMN_MAPPING, volume=axes[1], style='yahoo', ax=axes[0], addplot=ap, vlines=vlines, datetime_format='%y-%m-%d')
  if ref_df is not None and len(ref_df) == len(df):
    mpf.plot(ref_df, type='ohlc', columns=utils.influx.MPF_COLUMN_MAPPING, style='yahoo', ax=axes[7], addplot=ap, vlines=vlines, datetime_format='%y-%m-%d')
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
  return fig

def daily_change_plot(day_data: utils.trading_day_data.TradingDayData, alines=None, title_add='', atr_vlines=dict(vlines=[], colors=[]),
                      ad=True, basetime='5m'):
  # |-------------------------|
  # |        5m/10m/15m       |
  # | ------------------------|
  # |   D   |       30m       |
  # | ------------------------|
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

  mpf.plot(df_base, type='candle', ax=ax1, columns=utils.influx.MPF_COLUMN_MAPPING, xrotation=0,
           datetime_format='%H:%M', tight_layout=True,
           scale_width_adjustment=dict(candle=1.35), hlines=hlines, alines=alines, vlines=vlines,
           addplot=[ind_5m_ema20_plot, ind_5m_ema240_plot, ind_vwap3_plot])
  mpf.plot(df_base, type='line', ax=ax4, columns=['lh'] * 5, xrotation=0, datetime_format='%H:%M', vlines=atr_vlines,
           tight_layout=True)

  mpf.plot(day_data.df_day, type='candle', ax=ax2, columns=utils.influx.MPF_COLUMN_MAPPING, xrotation=0,
           datetime_format='%m-%d', tight_layout=True,
           hlines=hlines_day, warn_too_much_data=700, addplot=[ind_day_ema20_plot])
  mpf.plot(df_30m, type='candle', ax=ax3, columns=utils.influx.MPF_COLUMN_MAPPING, xrotation=0, datetime_format='%H:%M',
           tight_layout=True,
           scale_width_adjustment=dict(candle=1.35), hlines=hlines, addplot=[ind_30m_ema20_plot])

  # Use MaxNLocator to increase the number of ticks
  ax1.xaxis.set_major_locator(mticker.MaxNLocator(nbins=20))  # Increase number of ticks on x-axis
  ax4.xaxis.set_major_locator(mticker.MaxNLocator(nbins=20))  # Increase number of ticks on x-axis
  ax1.yaxis.set_major_locator(mticker.MaxNLocator(nbins=10))  # Increase number of ticks on y-axis
  plt.tight_layout(h_pad=0.1)


def last_date_from_files(directory):
  files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
  # Sort files by name in descending order
  files_sorted = sorted(files, reverse=True)
  first_file = files_sorted[0] if files_sorted else None
  if first_file is not None:
    # Define a regex pattern to extract the date (format: YYYY-MM-DD)
    date_pattern = r'\d{4}-\d{2}-\d{2}'  # Adjust the pattern to match your date format

    # Find the date in the filename
    match = re.search(date_pattern, first_file)
    if match:
      date_str = match.group()  # Extract the date string
      parsed_date = datetime.strptime(date_str, "%Y-%m-%d")  # Parse into a datetime object
      print(f"Date string: {date_str}")
      print(f"Parsed date: {parsed_date}")
      return parsed_date
    else:
      print("No date found in filename.")
  return None


def heatmap(df_corr, mask=None, name='Correlation Matrix', ax = None):
  # 3. Setup Plot
  fig, ax = plt.subplots(figsize=(24, 14)) if ax is None else (ax.figure, ax)
  # We use the original 'corr' for imshow but can set masked values to NaN/Alpha
  im = ax.imshow(df_corr, cmap='RdYlGn', vmin=-1, vmax=1)

  # 4. Labels and Ticks
  cols = df_corr.columns
  ax.set_xticks(np.arange(len(cols)))
  ax.set_yticks(np.arange(len(cols)))
  ax.set_xticklabels(cols)
  ax.set_yticklabels(cols)

  plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

  # 5. Add text labels only for the non-masked parts
  for i in range(len(cols)):
    for j in range(len(cols)):
      if not mask[i, j]:  # Only draw text if it's not in mask
        val = df_corr.iloc[i, j]
        ax.text(j, i, f"{val:.2f}",
                ha="center", va="center", color="black", fontsize=9)

  # 6. Aesthetics
  ax.set_title(name)
  # Remove the top and right spines to clean up the "empty" space
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
  """
  Boxplot for all t-columns, annotated with:
    Q1, median, Q3, lower whisker, upper whisker (actual numeric values).
  Whiskers follow Matplotlib's default logic: most extreme data within [Q1 - whis*IQR, Q3 + whis*IQR].
  """
  text_kwargs = text_kwargs or {}

  cols = df.columns.tolist()

  data = [df[c].dropna().to_numpy(dtype=float) for c in cols]

  fig, ax = plt.subplots(figsize=figsize) if ax is None else (ax.figure, ax)
  ax.boxplot(data, tick_labels=cols, showfliers=showfliers, whis=whis)

  ax.axhline(0, color="gray", linewidth=1)
  ax.set_title("Distribution of values")
  ax.grid(True, axis="y", alpha=0.25)
  plt.setp(ax.get_xticklabels(), rotation=rotate, ha="right")

  # Annotate each box with Q1/Median/Q3 and whiskers
  for i, y in enumerate(data, start=1):  # positions are 1..N
    y = y[np.isfinite(y)]
    if y.size == 0:
      continue

    q1, med, q3 = np.percentile(y, [25, 50, 75])
    iqr = q3 - q1
    lo_fence = q1 - whis * iqr
    hi_fence = q3 + whis * iqr

    # "Actual" whiskers = most extreme points inside fences
    in_lo = y[y >= lo_fence]
    in_hi = y[y <= hi_fence]
    wlo = np.min(in_lo) if in_lo.size else np.min(y)
    whi = np.max(in_hi) if in_hi.size else np.max(y)

    # place labels slightly to the right of each box
    x = i
    bbox = dict(facecolor="white", edgecolor="none", alpha=0.75)

    ax.text(x, whi, f"whi={fmt.format(whi)}", va="bottom", ha="left", fontsize=8, bbox=bbox, **text_kwargs)
    ax.text(x, q3, f"q3={fmt.format(q3)}", va="bottom", ha="left", fontsize=8, bbox=bbox, **text_kwargs)
    ax.text(x, med, f"m ={fmt.format(med)}", va="center", ha="left", fontsize=8, bbox=bbox, **text_kwargs)
    ax.text(x, q1, f"q1={fmt.format(q1)}", va="top", ha="left", fontsize=8, bbox=bbox, **text_kwargs)
    ax.text(x, wlo, f"wlo={fmt.format(wlo)}", va="top", ha="left", fontsize=8, bbox=bbox, **text_kwargs)

  plt.tight_layout()
  return fig, ax


def violinplot_columns_with_labels(
    df,
    whis=1.5,
    figsize=(23, 12),
    rotate=45,
    fmt="{:.2f}",
    text_kwargs=None,
    title='',
    ax=None
):
  """
  Violin plot for columns, clipped at the whisker levels (Q1 - whis*IQR, Q3 + whis*IQR).
  """
  text_kwargs = text_kwargs or {}
  cols = df.columns.tolist()

  # Calculate stats and filter data for each column to "cut off" extremes
  filtered_data = []
  stats = []

  for c in cols:
    y = df[c].dropna().to_numpy(dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
      filtered_data.append(np.array([]))
      stats.append(None)
      continue

    q1, med, q3 = np.percentile(y, [25, 50, 75])
    iqr = q3 - q1
    lo_fence = q1 - whis * iqr
    hi_fence = q3 + whis * iqr

    # Clip the data to the fences (same logic as boxplot without fliers)
    y_clipped = y[(y >= lo_fence) & (y <= hi_fence)]
    filtered_data.append(y_clipped)

    # Store actual whiskers (extreme points inside fences) for labels
    wlo = np.min(y_clipped) if y_clipped.size else q1
    whi = np.max(y_clipped) if y_clipped.size else q3
    stats.append({'q1': q1, 'med': med, 'q3': q3, 'wlo': wlo, 'whi': whi})

  fig, ax = plt.subplots(figsize=figsize) if ax is None else (ax.figure, ax)

  # Plot only the data within the "whisker" range
  # Filter out empty arrays to prevent violinplot errors
  plot_indices = [i for i, d in enumerate(filtered_data) if d.size > 0]
  plot_data = [filtered_data[i] for i in plot_indices]
  plot_positions = [i + 1 for i in plot_indices]

  if plot_data:
    parts = ax.violinplot(plot_data, positions=plot_positions, showmeans=False, showmedians=False, showextrema=False)
    for pc in parts['bodies']:
      pc.set_facecolor('tab:blue')
      pc.set_edgecolor('black')
      pc.set_alpha(0.7)

  ax.set_xticks(np.arange(1, len(cols) + 1))
  ax.set_xticklabels(cols, fontsize=10)
  ax.axhline(0, color="gray", linewidth=1)
  if title:
    ax.set_title(title)
  ax.grid(True, axis="y", alpha=0.25)
  plt.setp(ax.get_xticklabels(), rotation=rotate, ha="right")

  # Annotate with stats
  for i, s in enumerate(stats, start=1):
    if s is None: continue

    x, bbox = i, dict(facecolor="white", edgecolor="none", alpha=0.75)
    ax.hlines(s['med'], x - 0.1, x + 0.1, color='black', lw=2)
    ax.hlines([s['q1'], s['q3']], x - 0.1, x + 0.1, color='black', lw=1, linestyle='-', alpha=0.6, zorder=3)

    ax.text(x, s['whi'], f"whi={fmt.format(s['whi'])}", va="bottom", ha="left", fontsize=8, bbox=bbox, **text_kwargs)
    ax.text(x, s['q3'], f"q3={fmt.format(s['q3'])}", va="bottom", ha="left", fontsize=8, bbox=bbox, **text_kwargs)
    ax.text(x, s['med'], f"m ={fmt.format(s['med'])}", va="center", ha="left", fontsize=8, bbox=bbox, **text_kwargs)
    ax.text(x, s['q1'], f"q1={fmt.format(s['q1'])}", va="top", ha="left", fontsize=8, bbox=bbox, **text_kwargs)
    ax.text(x, s['wlo'], f"wlo={fmt.format(s['wlo'])}", va="top", ha="left", fontsize=8, bbox=bbox, **text_kwargs)

  plt.tight_layout()
  return fig, ax


# %% Probability Tree Analysis
def plot_probability_tree(series, depth=4, title='', lower_limit=None, upper_limit=0.55):
  """
  Creates a tree graph showing probabilities of consecutive moves.
  """
  # Convert to Up (1) or Down (0)
  moves = (series > 0).astype(int).tolist()

  G = nx.DiGraph()
  # root node: (depth_level, sequence_string)
  root = (0, "")
  G.add_node(root, label="Start")

  nodes_at_level = [root]

  for d in range(depth):
    next_level = []
    for parent in nodes_at_level:
      p_depth, p_seq = parent

      # Filter full moves list to find occurrences of the parent sequence
      # We look for the sequence followed by 1 or 0
      count_up = 0
      count_down = 0

      # Simple sequence matching
      seq_len = len(p_seq)
      for i in range(len(moves) - seq_len - 1):
        # Check if previous moves match current path
        match = True
        for j in range(seq_len):
          if moves[i + j] != int(p_seq[j]):
            match = False
            break

        if match:
          next_move = moves[i + seq_len]
          if next_move == 1:
            count_up += 1
          else:
            count_down += 1

      total = count_up + count_down
      if total > 0:
        prob_up = count_up / total
        prob_down = count_down / total

        node_up = (d + 1, p_seq + "1")
        node_down = (d + 1, p_seq + "0")

        G.add_node(node_up, label="U")
        G.add_node(node_down, label="D")

        G.add_edge(parent, node_up, weight=prob_up, label=f"{prob_up:.0%} | {count_up}")
        G.add_edge(parent, node_down, weight=prob_down, label=f"{prob_down:.0%} | {count_down}")

        next_level.extend([node_up, node_down])
    nodes_at_level = next_level

  # Layout for a tree
  pos = {}

  def set_pos(node, x, y, width):
    p_depth, p_seq = node
    pos[node] = (x, -p_depth)
    children = [n for n in G.successors(node)]
    if children:
      # Down child left, Up child right
      set_pos(children[1], x - width / 2, y - 1, width / 2)  # Down (0)
      set_pos(children[0], x + width / 2, y - 1, width / 2)  # Up (1)

  set_pos(root, 0, 0, 10)

  plt.figure(figsize=(24, 14))
  plt.title(f"Transition Probabilities - {title}", fontsize=15, pad=20)

  # Determine edge colors and widths based on limits
  edge_colors = []
  edge_widths = []
  for u, v, data in G.edges(data=True):
    prob = data['weight']
    highlight = False
    if lower_limit is not None and prob <= lower_limit:
      highlight = True
    if upper_limit is not None and prob >= upper_limit:
      highlight = True

    edge_colors.append('magenta' if highlight else 'gray')
    edge_widths.append(3.0 if highlight else 1.0)

  # Determine node colors: Start=lightblue, U=green, D=red
  node_colors = []
  for n, data in G.nodes(data=True):
    label = data.get('label', '')
    if label == "U":
      node_colors.append('green')
    elif label == "D":
      node_colors.append('red')
    else:
      node_colors.append('lightblue')

  nx.draw(G, pos, with_labels=False, node_size=50, node_color=node_colors, arrows=True, edge_color=edge_colors,
          width=edge_widths)

  # Draw node labels (Up/Down)
  node_labels = nx.get_node_attributes(G, 'label')
  nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)

  # Draw edge labels (Probabilities)
  edge_labels = nx.get_edge_attributes(G, 'label')
  nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

  plt.tight_layout()
  plt.show()

#%% # Global Plot Configurations
EMA_CONFIGS = {
  'ema10':  {'color': '#9b7f6c', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine}, # Wheat (Lightest)
  'ema20':  {'color': '#b26529', 'width': 2.0, 'style': QtCore.Qt.PenStyle.SolidLine},
  'ema50':  {'color': '#7b4326', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},
  'ema100': {'color': '#703b24', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},
  'ema200': {'color': '#5a2c27', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},  # Darkest
  'vwap3': {'color': '#47a3b9', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine}  # Darkest
}

ATR_CONFIGS = {
  'atrp1': {'color': '#f5a1df', 'width': 1.0, 'style': QtCore.Qt.PenStyle.DashLine}, # Steel Blue
  'atrp9':  {'color': '#f81cfc', 'width': 1.0, 'style': QtCore.Qt.PenStyle.DashLine}, # Light Steel Blue
  'atrp20': {'color': '#b72494', 'width': 2.0, 'style': QtCore.Qt.PenStyle.SolidLine},  # Navy
  'atrp50': {'color': '#6b1255', 'width': 1.0, 'style': QtCore.Qt.PenStyle.DashLine}  # Navy
}

STD_CONFIGS = {
  'std' : {'color':  '#ba68c8', 'width': 1.5, 'style': QtCore.Qt.PenStyle.SolidLine}  # Blue
}

AC_CONFIGS = {
  'ac100_lag_1':  {'color': '#e0ffff', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine}, # Light Cyan
  'ac100_lag_5':  {'color': '#00ced1', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine}, # Dark Turquoise
  'ac100_lag_10': {'color': '#00bfff', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine}, # Deep Sky Blue
  'ac100_lag_20': {'color': '#008080', 'width': 1.5, 'style': QtCore.Qt.PenStyle.SolidLine}  # Teal
}

# New configuration for the Autocorrelation Regime pane
AC_REGIME_CONFIGS = {
  'ac_mom':  {'color': '#e1bee7', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine}, # Light Purple (Momentum)
  'ac_mr':   {'color': '#ba68c8', 'width': 1.5, 'style': QtCore.Qt.PenStyle.SolidLine}, # Medium Purple (Mean Reversion)
  'ac_comp': {'color': '#4a148c', 'width': 2.2, 'style': QtCore.Qt.PenStyle.SolidLine}  # Indigo/Darkest (Composite)
}

SLOPE_CONFIGS = {
  'ema10_slope':  {'color': '#ffccbc', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine}, # Deep Orange Light
  'ema20_slope':  {'color': '#ff8a65', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},
  'ema50_slope':  {'color': '#ff5722', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},
  'ema100_slope': {'color': '#e64a19', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},
  'ema200_slope': {'color': '#bf360c', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine}  # Deep Orange Dark
}

VOL_CONFIGS = {
  'v':  {'color': '#49bdd9', 'width': 1.0, 'style': QtCore.Qt.PenStyle.DashLine}, # Deep Orange Light
  'v9':  {'color': '#fcec98', 'width': 1.0, 'style': QtCore.Qt.PenStyle.DashLine},
  'v20':  {'color': '#f3cb21', 'width': 1.5, 'style': QtCore.Qt.PenStyle.SolidLine},
  'v50': {'color': '#dab312', 'width': 1.5, 'style': QtCore.Qt.PenStyle.DashLine}
}

DIST_CONFIGS = {
  'ema10_dist':  {'color': '#9b7f6c', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine}, # Deep Orange Light
  'ema20_dist':  {'color': '#b26529', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},
  'ema50_dist':  {'color': '#7b4326', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},
  'ema100_dist': {'color': '#703b24', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},
  'ema200_dist': {'color': '#5a2c27', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine}  # Deep Orange Dark
}

HURST_CONFIGS = {
  'hurst50':  {'color': '#fff59d', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine}, # Light Yellow
  'hurst100': {'color': '#fbc02d', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine}  # Dimmed Gold/Brownish
}

HV_CONFIGS = {
  'hv9':  {'color': '#b7a3db', 'width': 1.0, 'style': QtCore.Qt.PenStyle.DashLine}, # Light Green
  'hv20': {'color': '#6539b4', 'width': 2.0, 'style': QtCore.Qt.PenStyle.SolidLine}, # Medium Green
  'hv50': {'color': '#583098', 'width': 1.0, 'style': QtCore.Qt.PenStyle.DashLine},  # Dark Green
  'iv':   {'color': '#49bcd8', 'width': 2.0, 'style': QtCore.Qt.PenStyle.SolidLine}  # Magenta (Standout)
}

IVPCT_CONFIGS = {
  'iv_pct':  {'color': '#b72494', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine}, # Light Green
}

# TTM Squeeze Colors
TTM_COLORS = {
  'pos_up': '#00ff00',   # Bright Green (Bullish rising)
  'pos_down': '#006400', # Dark Green (Bullish falling)
  'neg_down': '#ff0000', # Red (Bearish falling)
  'neg_up': '#8b0000',   # Dark Red (Bearish rising)
  'sq_on': '#ff0000',    # Red dot (Squeeze ON)
  'sq_off': '#00ff00'    # Green dot (Squeeze OFF)
}

##%%

class DateAxis(pg.AxisItem):
  def __init__(self, dates, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.dates = dates

  def tickStrings(self, values, scale, spacing):
    return [self.dates[int(v)].strftime('%y-%m-%d') if 0 <= int(v) < len(self.dates) else "" for v in values]

  def tickValues(self, minVal, maxVal, size):
    # Increase the number of ticks by reducing the required spacing
    return super().tickValues(minVal, maxVal, size * 0.5)

class OHLCItem(pg.GraphicsObject):
  def __init__(self, data):
    pg.GraphicsObject.__init__(self)
    self.data = data # [time, open, high, low, close]
    self.generatePicture()

  def generatePicture(self):
    self.picture = QtGui.QPicture()
    p = QtGui.QPainter(self.picture)
    p.setPen(pg.mkPen('w'))
    for t, open, high, low, close in self.data:
      color = pg.mkPen('g') if close >= open else pg.mkPen('r')
      p.setPen(color)
      p.drawLine(QtCore.QPointF(t, low), QtCore.QPointF(t, high))
      # Open tick (left) and Close tick (right)
      p.drawLine(QtCore.QPointF(t-0.3, open), QtCore.QPointF(t, open))
      p.drawLine(QtCore.QPointF(t, close), QtCore.QPointF(t+0.3, close))
    p.end()

  def paint(self, p, *args):
    p.drawPicture(0, 0, self.picture)

  def boundingRect(self):
    return QtCore.QRectF(self.picture.boundingRect())

class TTMSqueezeItem(pg.GraphicsObject):
  def __init__(self, data):
    pg.GraphicsObject.__init__(self)
    self.data = data # list of (x, mom, squeeze_on)
    self.generatePicture()

  def generatePicture(self):
    self.picture = QtGui.QPicture()
    p = QtGui.QPainter(self.picture)

    # Determine a dynamic radius based on the momentum range to keep dots visible
    mom_values = [d[1] for d in self.data if not np.isnan(d[1])]
    max_abs_mom = max(abs(min(mom_values)), abs(max(mom_values))) if mom_values else 1.0
    dot_radius_y = max_abs_mom * 0.05  # 5% of the max range
    dot_radius_x = 0.2                 # Fixed width in x-units (bars)

    for i in range(len(self.data)):
      x, mom, sq_on = self.data[i]
      if np.isnan(mom): continue

      prev_mom = self.data[i-1][1] if i > 0 else mom

      # 1. Determine Histogram Color
      if mom >= 0:
        color = TTM_COLORS['pos_up'] if mom >= prev_mom else TTM_COLORS['pos_down']
      else:
        color = TTM_COLORS['neg_down'] if mom <= prev_mom else TTM_COLORS['neg_up']

      p.setPen(pg.mkPen(None))
      p.setBrush(pg.mkBrush(color))
      # drawRect(x, y, w, h) -> y is top-left, so for positive mom, y=0 is bottom
      # For negative mom, y=0 is top.
      p.drawRect(QtCore.QRectF(x - 0.4, 0, 0.8, -mom)) # Negating mom handles the y-direction correctly

      # 2. Draw Squeeze Dot
      dot_color = TTM_COLORS['sq_on'] if sq_on else TTM_COLORS['sq_off']
      p.setPen(pg.mkPen(None))
      p.setBrush(pg.mkBrush(dot_color))
      # Use specific X and Y radii to account for axis scaling
      p.drawEllipse(QtCore.QPointF(x, 0), dot_radius_x, dot_radius_y)

    p.end()

  def paint(self, p, *args):
    p.drawPicture(0, 0, self.picture)

  def boundingRect(self):
    return QtCore.QRectF(self.picture.boundingRect())

def plot_pyqtgraph(full_df, initial_max_date=None, export_path=None, vlines=None, display_range=250,
                   export_width=1920, export_height=1080):
  app = pg.mkQApp()

  # Main Window Setup
  main_win = QtWidgets.QMainWindow()
  main_win.setWindowTitle(f"{full_df.symbol[0]} Multi-Pane Analysis")
  central_widget = QtWidgets.QWidget()
  main_win.setCentralWidget(central_widget)
  layout = QtWidgets.QVBoxLayout(central_widget)

  # 1. Date Selector Toolbar
  toolbar = QtWidgets.QHBoxLayout()

  years = [str(y) for y in sorted(full_df.index.year.unique(), reverse=True)]
  months = [f"{m:02d}" for m in range(1, 13)]
  days = [f"{d:02d}" for d in range(1, 32)]

  year_cb = QtWidgets.QComboBox(); year_cb.addItems(years)
  month_cb = QtWidgets.QComboBox(); month_cb.addItems(months)
  day_cb = QtWidgets.QComboBox(); day_cb.addItems(days)

  # Set initial values from initial_max_date or last date in df
  start_date = pd.to_datetime(initial_max_date) if initial_max_date else full_df.index[-1]
  year_cb.setCurrentText(str(start_date.year))
  month_cb.setCurrentText(f"{start_date.month:02d}")
  day_cb.setCurrentText(f"{start_date.day:02d}")

  toolbar.addWidget(QtWidgets.QLabel("Max Date Filter:"))
  toolbar.addWidget(year_cb); toolbar.addWidget(month_cb); toolbar.addWidget(day_cb)
  toolbar.addStretch()
  layout.addLayout(toolbar)

  # 2. Graphics Layout
  win = pg.GraphicsLayoutWidget()
  layout.addWidget(win)

  # Store proxy in a list or as a property to prevent garbage collection
  state = {'proxy': None, 'df': None, 'x_range': None, 'x_dates': None}
  p1 = p3 = p4 = p5 = p6 = p7 = p8 = p1_vol = None

  # Resize window to match export proportions initially to help layout calculation
  main_win.resize(export_width, export_height)
  main_win.show()
  if export_path:
    main_win.hide()

  def update_y_views():
    if state['df'] is None or p1 is None: return

    vr = p1.viewRange()[0]
    s, e = max(0, int(vr[0])), min(len(state['df']), int(vr[1]))
    if s < e:
      chunk = state['df'].iloc[s:e]
      p1.setYRange(chunk.l.min()*0.99, chunk.h.max()*1.01, padding=0)

      # Scale Indicator Panes
      scale_map = [
        (p7, list(VOL_CONFIGS.keys())), (p3, list(DIST_CONFIGS.keys())),
        (p4, list(ATR_CONFIGS.keys())), (p5, list(HV_CONFIGS.keys())),
        (p6, list(IVPCT_CONFIGS.keys())), (p8, ['ttm_mom'])]

      for p, cols in scale_map:
        valid_cols = [c for c in cols if c in state['df'].columns]
        if valid_cols:
          chunk_data = chunk[valid_cols]
          if p == p7: # Scale Volume MAs to thousands
            chunk_data = chunk_data / 1000
          p.setYRange(chunk_data.min().min()*1.1, chunk_data.max().max()*1.1, padding=0)

  def update_plot():
    nonlocal p1, p3, p4, p5, p6, p7, p8
    win.clear()
    target_str = f"{year_cb.currentText()}-{month_cb.currentText()}-{day_cb.currentText()}"
    df = full_df[full_df.index <= target_str]
    if df.empty: return

    state['df'] = df
    state['x_dates'] = df.index
    state['x_range'] = np.arange(len(df))
    x_range = state['x_range']

    # --- Plot Setup ---
    p1 = win.addPlot(row=0, col=0)

    # Indicator Panes - Reordered: Vol MA is now first (row 1)
    p7 = win.addPlot(row=1, col=0)
    p3 = win.addPlot(row=2, col=0)
    p4 = win.addPlot(row=3, col=0)
    p5 = win.addPlot(row=4, col=0)
    p6 = win.addPlot(row=5, col=0)
    # Bottom pane gets the DateAxis
    p8 = win.addPlot(row=6, col=0, axisItems={'bottom': DateAxis(dates=state['x_dates'], orientation='bottom')})

    # Apply proportional heights
    win.ci.layout.setRowStretchFactor(0, 50) # Main Chart
    win.ci.layout.setRowStretchFactor(1, 8)  # p7 (Vol MA)
    win.ci.layout.setRowStretchFactor(2, 2)  # p3 (EMA Dist)
    win.ci.layout.setRowStretchFactor(3, 15) # p4 (ATR)
    win.ci.layout.setRowStretchFactor(4, 15) # p5 (IV/HV)
    win.ci.layout.setRowStretchFactor(5, 2)  # p6 (IVPct)
    win.ci.layout.setRowStretchFactor(6, 8)  # p8 (TTM Squeeze)

    plots = [p1, p7, p3, p4, p5, p6, p8]
    for p in plots:
      p.showGrid(x=True, y=True, alpha=0.3)
      p.getAxis('left').setWidth(70)
      p.getAxis('right').setWidth(70)
      p.showAxis('right')
      if p != p8:
        p.getAxis('bottom').hide() # Hide x-axis for all but bottom pane
      if p != p1:
        p.setXLink(p1)
        p.setMaximumHeight(16777215)

    p7.setLabels(left='Vol MA')
    p3.setLabels(left='EMA Dist')
    p4.setLabels(left='ATR %')
    p5.setLabels(left='IV/HV')
    p6.setLabels(left='IVPct')
    p8.setLabels(left='TTM Squeeze')

    hover_label = pg.TextItem(anchor=(0, 0), color='#ccc', fill='#000e')
    p1.addItem(hover_label, ignoreBounds=True)

    # --- Plotting Content ---
    p1.addItem(OHLCItem([(i, df.o.iloc[i], df.h.iloc[i], df.l.iloc[i], df.c.iloc[i]) for i in x_range]))
    for col, cfg in EMA_CONFIGS.items():
      if col in df.columns:
        p1.plot(x=x_range, y=df[col].values, pen=pg.mkPen(cfg['color'], width=cfg['width'], style=cfg.get('style', QtCore.Qt.PenStyle.SolidLine)), connect='finite')

    # Indicators
    for col, cfg in VOL_CONFIGS.items():
      if col in df.columns:
        p7.plot(x=x_range, y=df[col].values / 1000, pen=pg.mkPen(cfg['color'], width=cfg['width'], style=cfg.get('style', QtCore.Qt.PenStyle.SolidLine)), connect='finite')
    p7.addLine(y=0, pen=pg.mkPen('#666', width=1))

    for col, cfg in DIST_CONFIGS.items():
      if col in df.columns:
        p3.plot(x=x_range, y=df[col].values, pen=pg.mkPen(cfg['color'], width=cfg['width'], style=cfg.get('style', QtCore.Qt.PenStyle.SolidLine)), connect='finite')
    p3.addLine(y=1.2, pen=pg.mkPen('#666', width=1))

    for col, cfg in ATR_CONFIGS.items():
      if col in df.columns:
        p4.plot(x=x_range, y=df[col].values, pen=pg.mkPen(cfg['color'], width=cfg['width'], style=cfg.get('style', QtCore.Qt.PenStyle.SolidLine)), connect='finite')
    p4.addLine(y=0, pen=pg.mkPen('#666', width=1))

    for col, cfg in HV_CONFIGS.items():
      if col in df.columns:
        p5.plot(x=x_range, y=df[col].values, pen=pg.mkPen(cfg['color'], width=cfg['width'], style=cfg.get('style', QtCore.Qt.PenStyle.SolidLine)), connect='finite')
    p5.addLine(y=0, pen=pg.mkPen('#666', width=1))

    for col, cfg in IVPCT_CONFIGS.items():
      if col in df.columns:
        p6.plot(x=x_range, y=df[col].values, pen=pg.mkPen(cfg['color'], width=cfg['width'], style=cfg.get('style', QtCore.Qt.PenStyle.SolidLine)), connect='finite')
    p6.addLine(y=0.5, pen=pg.mkPen('#666', style=QtCore.Qt.PenStyle.DashLine))

    if 'ttm_mom' in df.columns and 'squeeze_on' in df.columns:
      p8.addItem(TTMSqueezeItem([(i, df.ttm_mom.iloc[i], df.squeeze_on.iloc[i]) for i in x_range]))
    p8.addLine(y=0, pen=pg.mkPen('#666', width=1))

    # --- Vertical Marker Lines ---
    if vlines:
      for v_date in vlines:
        v_dt = pd.to_datetime(v_date)
        if v_dt in df.index:
          idx = df.index.get_loc(v_dt)
          for p in plots:
            marker_line = pg.InfiniteLine(pos=idx, angle=90, pen=pg.mkPen('darkviolet', width=0.8, style=QtCore.Qt.PenStyle.DashLine))
            p.addItem(marker_line)

    # 9. CROSSHAIRS & Interaction Logic
    v_lines = []
    h_lines = []
    for p in plots:
      v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#666', style=QtCore.Qt.PenStyle.DashLine))
      h_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('#666', style=QtCore.Qt.PenStyle.DashLine))
      p.addItem(v_line, ignoreBounds=True)
      p.addItem(h_line, ignoreBounds=True)
      v_lines.append(v_line)
      h_lines.append(h_line)
      h_line.hide()

    def update_hover(evt):
      pos = evt[0]
      for hl in h_lines: hl.hide()
      for i, active_p in enumerate(plots):
        if active_p.sceneBoundingRect().contains(pos):
          mousePoint = active_p.vb.mapSceneToView(pos)
          idx = int(mousePoint.x() + 0.5)

          if 0 <= idx < len(df):
            row = df.iloc[idx]
            for line in v_lines: line.setPos(idx)
            h_lines[i].setPos(mousePoint.y())
            h_lines[i].show()

            # Formatted Hover Text
            txt = f"<span style='font-size: 11pt; color: white; font-weight: bold;'>{state['x_dates'][idx].strftime('%a %Y-%m-%d')}</span><br>"
            txt += f"O:{row.o:.2f} H:{row.h:.2f} L:{row.l:.2f} C:{row.c:.2f} V:{row.v/1000:,.0f}k<br>"
            dists = " | ".join([f"<span style='color:{DIST_CONFIGS[c]['color']};'>{c.split('_')[0]}:{row[c]:.2f}</span>" for c in DIST_CONFIGS if c in df.columns])
            emas = " | ".join([f"<span style='color:{EMA_CONFIGS[c]['color']};'>{c.upper()}:{row[c]:.2f}</span>" for c in EMA_CONFIGS if c in df.columns])
            atrs = " | ".join([f"<span style='color:{ATR_CONFIGS[c]['color']};'>{c.upper()}:{row[c]:.2f}%</span>" for c in ATR_CONFIGS if c in df.columns])
            hvs = " | ".join([f"<span style='color:{HV_CONFIGS[c]['color']};'>{c.upper()}:{row[c]:.2f}</span>" for c in HV_CONFIGS if c in df.columns])
            ivpct = " | ".join([f"<span style='color:{IVPCT_CONFIGS[c]['color']};'>{c}:{row[c]:.2f}</span>" for c in IVPCT_CONFIGS if c in df.columns])
            vols = " | ".join([f"<span style='color:{VOL_CONFIGS[c]['color']};'>{c.upper()}:{row[c]/1000:.2f}k</span>" for c in VOL_CONFIGS if c in df.columns])

            hover_label.setHtml(txt + dists + "<br>" + atrs + "<br>" + hvs + "<br>" + ivpct + "<br>" + vols )
            vb_range = p1.vb.viewRange()
            hover_label.setPos(vb_range[0][0] + (vb_range[0][1]-vb_range[0][0])*0.01,
                               vb_range[1][1] - (vb_range[1][1]-vb_range[1][0])*0.01)

    state['proxy'] = pg.SignalProxy(p1.scene().sigMouseMoved, rateLimit=60, slot=update_hover)
    p1.sigXRangeChanged.connect(update_y_views)

    # Use the display_range parameter to set the initial view
    p1.setXRange(max(0, len(df) - display_range), len(df))
    update_y_views()

  # Connect signals and run initial draw
  for cb in [year_cb, month_cb, day_cb]:
    cb.currentIndexChanged.connect(update_plot)

  update_plot()

  if export_path:
    win.ci.layout.activate()
    app.processEvents()
    update_y_views()
    app.processEvents()

    exporter = pg.exporters.ImageExporter(win.scene())
    exporter.parameters()['width'] = export_width
    exporter.parameters()['height'] = export_height
    exporter.export(export_path)
    main_win.close()
  else:
    pg.exec()
