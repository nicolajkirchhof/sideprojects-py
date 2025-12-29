#%%

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2


#%%
df_gaps = pd.read_pickle(f'finance/_data/gaps_sym.pkl')

#%%
t_cols = df_gaps.filter(regex=r"^t-?\d+$").columns

def boxplot_t_columns_with_labels(
    df,
    regex=r"^t-?\d+$",
    whis=1.5,
    showfliers=False,
    figsize=(23, 12),
    rotate=45,
    fmt="{:.2f}",
    text_kwargs=None,
):
  """
  Boxplot for all t-columns, annotated with:
    Q1, median, Q3, lower whisker, upper whisker (actual numeric values).
  Whiskers follow Matplotlib's default logic: most extreme data within [Q1 - whis*IQR, Q3 + whis*IQR].
  """
  text_kwargs = text_kwargs or {}

  t_cols = df.filter(regex=regex).columns.tolist()
  if not t_cols:
    raise ValueError(f"No columns matched regex: {regex}")

  # Sort: t-1, t0, t1, ...
  def t_key(c: str) -> int:
    return int(c[1:])  # 't-1' -> -1, 't14' -> 14
  t_cols = sorted(t_cols, key=t_key)

  data = [df[c].to_numpy(dtype=float) for c in t_cols]

  fig, ax = plt.subplots(figsize=figsize)
  ax.boxplot(data, tick_labels=t_cols, showfliers=showfliers, whis=whis)

  ax.axhline(0, color="gray", linewidth=1)
  ax.set_title("Distribution of values by t (with quantile + whisker labels)")
  ax.set_xlabel("t")
  ax.set_ylabel("Value")
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
    ax.text(x, q3,  f"q3={fmt.format(q3)}",  va="bottom", ha="left", fontsize=8, bbox=bbox, **text_kwargs)
    ax.text(x, med, f"m ={fmt.format(med)}", va="center", ha="left", fontsize=8, bbox=bbox, **text_kwargs)
    ax.text(x, q1,  f"q1={fmt.format(q1)}",  va="top",    ha="left", fontsize=8, bbox=bbox, **text_kwargs)
    ax.text(x, wlo, f"wlo={fmt.format(wlo)}", va="top",   ha="left", fontsize=8, bbox=bbox, **text_kwargs)

  plt.tight_layout()
  return fig, ax

def plot_gap_stats_df(gap_stats_df, alpha=0.2, lw=1.0, show_mean=True, title=''):
  fig, ax1 = plt.subplots(figsize=(23, 12))

  # Ensure we pick the window columns in the correct order
  t_cols = gap_stats_df.filter(regex=r"^t-?\d+$").columns

  data = [gap_stats_df[c].to_numpy(dtype=float) for c in t_cols]
  ax1.boxplot(data, tick_labels=t_cols, showfliers=False)  # set True if you want outliers shown
  ax1.axhline(0, color="gray", linewidth=1)
  ax1.set_title("Distribution of normalized returns by t column")
  ax1.set_xlabel("t")
  ax1.set_ylabel("Value")
  ax1.grid(True, axis="y", alpha=0.25)
  plt.tight_layout()
  return fig, ax1

def plot_gap_lines_df(gap_stats_df, alpha=0.2, lw=1.0, show_mean=True, title=''):
  """
  gap_stats_df: dataframe returned by gap_statistics(), containing columns:
    symbol, date, earnings, gappct, t-1, t0, ..., t14

  Plots each row's [t-1..t14] as y over x = 0..len(y)-1 (same behavior as before),
  plus an optional mean line.
  """
  fig, ax0 = plt.subplots(nrows=2, figsize=(23, 12))

  # Ensure we pick the window columns in the correct order
  t_cols = gap_stats_df.filter(regex=r"^t-?\d+$").columns

  ys = []
  for _, row in gap_stats_df.iterrows():
    y = row[t_cols].to_numpy(dtype=float)
    if y.size == 0:
      continue
    x = np.arange(y.size)
    ax0.plot(x, y, color="tab:blue", alpha=alpha, linewidth=lw)
    ys.append(y)

  if show_mean and ys:
    max_len = max(len(y) for y in ys)
    mat = np.full((len(ys), max_len), np.nan, dtype=float)
    for i, y in enumerate(ys):
      mat[i, :len(y)] = y
    mean_y = np.nanmean(mat, axis=0)
    ax0.plot(np.arange(max_len), mean_y, color="black", linewidth=2.5, label=f"Mean (n={len(ys)})")
    ax0.legend()

  # --- NEW: clamp y-axis to 5th..95th quantile (finite values only) ---
  vals = gap_stats_df.loc[:, t_cols].to_numpy(dtype=float).ravel()
  vals = vals[np.isfinite(vals)]
  if vals.size:
    y_lo, y_hi = np.quantile(vals, [0.05, 0.95])
    ax0.set_ylim(y_lo, y_hi)

  ax0.set_title(title)
  ax0.set_xlabel("Index (0..len(y)-1)")
  ax0.set_ylabel("Normalized % change (baseline = t-1)")
  ax0.grid(True, alpha=0.25)

  plt.tight_layout()
  return fig, ax0

#%%
#%%
max_diff = 75
gap_diffs = [3, 4] #, 5, 6, 7, 9, 11, 13, 15, 20, 30, 50, 100]
# Uncommon behavior
df_gap_uc = df_gaps[(df_gaps[t_cols] > 100).any(axis=1)]
for i, g_min in enumerate(gap_diffs[:-1]):
  g_max = gap_diffs[i+1]
  df_gaps_plt = df_gaps[(df_gaps.is_etf == 0) & (df_gaps.gappct >= g_min) & (df_gaps.gappct < g_max) & (df_gaps.t0 > g_min)]
  # print(f'{i}: {len(df_gaps_plt)}')
  # plot_gap_stats_df(df_gaps_plt, title=f'Gap Up {g_min}% - {g_max+1}%')
  boxplot_t_columns_with_labels(df_gaps_plt)
  plt.show()
#%%
# Evaluate gaps by pct above BOP and pct below BOP, mean / std performance
# symbol, date, gap%, t-1, t0, t1, t2, t3, t4, ..., t14
# Ensure we pick the window columns in the correct order
t_cols = df_gaps.filter(regex=r"^t-?\d+$").columns

# Uncommon behavior
df_gap_uc = df_gaps[(df_gaps[t_cols] > 100).any(axis=1)]

#%%
gap = next(gaps.itertuples())
for gap in gaps.itertuples():
  idx = gap[0]
  #%%
  aXdays = []
  for i in [1, 3, 5, 7, 10, 14]:
    data = df_stk.iloc[idx + i, :]
    data.rename(columns={'date': f'date{i}', 'o': f'o{i}', 'h': f'h{i}', 'c':f'c{i}', 'l':f'l{i}', 'v': f'v{i}', 'gap'})
    aXdays.append(data)

