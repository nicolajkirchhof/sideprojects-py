import os
import re
from datetime import timedelta, datetime

import mplfinance as mpf
import numpy as np
import networkx as nx
import pandas as pd
from matplotlib import gridspec, pyplot as plt
from matplotlib.pyplot import tight_layout

from finance import utils
from finance.utils.trading_day_data import TradingDayData
import matplotlib.ticker as mticker


def daily_change_plot(day_data: TradingDayData, alines=None, title_add='', atr_vlines=dict(vlines=[], colors=[]),
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


def swing_plot(day_data, alines=None, title_add='', atr_vlines=dict(vlines=[], colors=[]), ad=True):
  # |-------------------------|
  # |     1D        |    1W   |
  # |   hv,iv,atr   |    1M   |
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
