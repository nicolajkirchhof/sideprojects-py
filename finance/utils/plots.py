"""
finance.utils.plots
=====================
Active matplotlib plotting helpers used by the swing plot dashboard.
"""
import numpy as np
import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt


def annotate_violin(ax, data, positions, labels):
    """
    Annotate a matplotlib violinplot with Q25 / median / Q75 values.

    Parameters
    ----------
    ax        : matplotlib Axes
    data      : list of array-like — one entry per violin
    positions : array-like          — x-positions matching data order
    labels    : list of str         — unused (kept for call-site compat)
    """
    for i, d in enumerate(data):
        if len(d) > 0:
            q25, q50, q75 = np.percentile(d, [25, 50, 75])
            ax.text(positions[i], q50, f'{q50:.1f}', ha='center', va='bottom',
                    color='white', fontsize='x-small', fontweight='bold')
            ax.text(positions[i], q25, f'{q25:.1f}', ha='center', va='top',
                    color='gray', fontsize='xx-small')
            ax.text(positions[i], q75, f'{q75:.1f}', ha='center', va='bottom',
                    color='gray', fontsize='xx-small')


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

    # Use background color to determine text and bbox colors
    is_dark = plt.rcParams['axes.facecolor'] in ['black', '#111111', '#1c1c1c'] or plt.style.library.get('dark_background') is not None
    bg_color = "black" if is_dark else "white"
    text_color = "white" if is_dark else "black"
    box_alpha = 0.85 if is_dark else 0.75

    x, bbox = i, dict(facecolor=bg_color, edgecolor="none", alpha=box_alpha)
    ax.hlines(s['med'], x - 0.1, x + 0.1, color=text_color, lw=2)
    ax.hlines([s['q1'], s['q3']], x - 0.1, x + 0.1, color=text_color, lw=1, linestyle='-', alpha=0.6, zorder=3)

    ax.text(x, s['whi'], f"whi={fmt.format(s['whi'])}", va="bottom", ha="left", fontsize=8, bbox=bbox, color=text_color, **text_kwargs)
    ax.text(x, s['q3'], f"q3={fmt.format(s['q3'])}", va="bottom", ha="left", fontsize=8, bbox=bbox, color=text_color, **text_kwargs)
    ax.text(x, s['med'], f"m ={fmt.format(s['med'])}", va="center", ha="left", fontsize=8, bbox=bbox, color=text_color, **text_kwargs)
    ax.text(x, s['q1'], f"q1={fmt.format(s['q1'])}", va="top", ha="left", fontsize=8, bbox=bbox, color=text_color, **text_kwargs)
    ax.text(x, s['wlo'], f"wlo={fmt.format(s['wlo'])}", va="top", ha="left", fontsize=8, bbox=bbox, color=text_color, **text_kwargs)

  plt.tight_layout()
  return fig, ax


# %% Probability Tree Analysis
def plot_probability_tree(series, depth=4, title='', lower_limit=None, upper_limit=0.55, ax=None):
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

  if ax is None:
    fig, ax = plt.subplots(figsize=(24, 14))
  else:
    fig = ax.get_figure()

  # Use background color to determine node and edge colors
  face_color = ax.get_facecolor()
  if isinstance(face_color, tuple):
    from matplotlib.colors import to_hex
    face_color_hex = to_hex(face_color)
  else:
    face_color_hex = face_color

  is_dark = face_color_hex in ['black', '#111111', '#1c1c1c'] or plt.style.library.get('dark_background') is not None
  edge_default_color = '#aaaaaa' if is_dark else 'gray'
  label_color = 'white' if is_dark else 'black'
  start_node_color = 'skyblue' if is_dark else 'lightblue'

  ax.set_title(f"Transition Probabilities - {title}", fontsize=15, pad=20, color=label_color)

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

    edge_colors.append('magenta' if highlight else edge_default_color)
    edge_widths.append(3.0 if highlight else 1.0)

  # Determine node colors: Start=lightblue, U=green, D=red
  node_colors = []
  for n, data in G.nodes(data=True):
    label = data.get('label', '')
    if label == "U":
      node_colors.append('#228B22') # ForestGreen
    elif label == "D":
      node_colors.append('#B22222') # FireBrick
    else:
      node_colors.append(start_node_color)

  nx.draw(G, pos, with_labels=False, node_size=50, node_color=node_colors, arrows=True, edge_color=edge_colors,
          width=edge_widths, ax=ax)

  # Draw node labels (Up/Down)
  node_labels = nx.get_node_attributes(G, 'label')
  nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, ax=ax, font_color=label_color)

  # Draw edge labels (Probabilities)
  edge_labels = nx.get_edge_attributes(G, 'label')
  nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, ax=ax, font_color=label_color,
                               bbox=dict(facecolor=ax.get_facecolor(), edgecolor='none', alpha=0.6))

  return fig, ax
