# %%
from datetime import datetime, timedelta

import dateutil
import numpy as np

import pandas as pd
import os

import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import mplfinance as mpf
from matplotlib.pyplot import tight_layout

from finance import utils

pd.options.plotting.backend = "matplotlib"
pd.set_option("display.max_columns", None)  # None means "no limit"
pd.set_option("display.max_rows", None)  # None means "no limit"
pd.set_option("display.width", 140)  # Set the width to 80 characters


mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2


# %%
symbols = ['DAX', 'ESTX50', 'SPX', 'INDU', 'NDX', 'USGOLD']
# Create a directory_evals
directory_evals = f'N:/My Drive/Trading/Strategies/thu_fri_mon'
directory_plots = f'N:/My Drive/Trading/Plots/thu_fri_mon'
os.makedirs(directory_evals, exist_ok=True)
os.makedirs(directory_plots, exist_ok=True)

weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
symbol = symbols[1]

#%%
for symbol in symbols:
  #%%
  print(f"Processing {symbol}")
  os.makedirs(f'{directory_evals}/{symbol}', exist_ok=True)
  symbol_def = utils.influx.SYMBOLS[symbol]
  tz = symbol_def['EX']['TZ']

  first_day = dateutil.parser.parse('2017-01-01T00:00:00').replace(tzinfo=tz)
  # last_day = dateutil.parser.parse('2016-03-19T00:00:00').replace(tzinfo=tz)
  last_day = dateutil.parser.parse('2025-04-09T00:00:00').replace(tzinfo=tz)
  df = utils.influx.get_candles_range_aggregate(first_day, last_day, symbol, '1d')
  df['weekday'] = df.index.day_name()
  df['year'] = df.index.year

  df = df.dropna()

  ##%%
  # conditions
  # higher high => lower low
  # higher high => higher high

  df['yst_hh'] = df.shift(2).h < df.shift(1).h
  df['yst_ll'] = df.shift(2).l > df.shift(1).l
  # high > close
  df['hc'] = df.shift(1).c < df.h
  df['hc_pct'] = (df.h - df.shift(1).c)*100/df.shift(1).c
  # low < low
  df['lc'] = df.shift(1).c > df.l
  df['lc_pct'] = (df.l - df.shift(1).c)*100/df.shift(1).c
  df['clc'] = df.shift(1).c > df.c
  df['chc'] = df.shift(1).c < df.c
  df['cc_pct'] = (df.c - df.shift(1).c)*100/df.shift(1).c

  df['hh_hc'] = df.apply(lambda x: np.nan if not x.yst_hh else x.hc, axis=1)
  df['hh_lc'] = df.apply(lambda x: np.nan if not x.yst_hh else x.lc, axis=1)
  df['ll_hc'] = df.apply(lambda x: np.nan if not x.yst_ll else x.hc, axis=1)
  df['ll_lc'] = df.apply(lambda x: np.nan if not x.yst_ll else x.lc, axis=1)

  df['hh_ll_hc'] = df.apply(lambda x: x.hc if x.yst_hh and x.yst_ll else np.nan, axis=1)
  df['hh_ll_lc'] = df.apply(lambda x: x.lc if x.yst_hh and x.yst_ll else np.nan, axis=1)
  df['hh_hl_hc'] = df.apply(lambda x: x.hc if x.yst_hh and not x.yst_ll else np.nan, axis=1)
  df['hh_hl_lc'] = df.apply(lambda x: x.lc if x.yst_hh and not x.yst_ll else np.nan, axis=1)
  df['lh_hl_hc'] = df.apply(lambda x: x.hc if not x.yst_hh and not x.yst_ll else np.nan, axis=1)
  df['lh_hl_lc'] = df.apply(lambda x: x.lc if not x.yst_hh and not x.yst_ll else np.nan, axis=1)
  df['lh_ll_hc'] = df.apply(lambda x: x.hc if not x.yst_hh and x.yst_ll else np.nan, axis=1)
  df['lh_ll_lc'] = df.apply(lambda x: x.lc if not x.yst_hh and x.yst_ll else np.nan, axis=1)

  # df.to_pickle(f'{directory_evals}/{symbol}_thu_fri_mon.pkl')
  # df.to_csv(f'{directory_evals}/{symbol}_fri_mon.csv')
  # df.to_excel(f'{directory_evals}/{symbol}_fri_mon.xlsx')
#%%
# evaluation
def pct(x):
  if x.count() == 0:
    return 0
  return np.round(x.sum()*100/x.count(), 2)

def format_axes(axes):
  for a in axes:
    a.legend(
      loc='lower center',
      bbox_to_anchor=(0.5, 0.0),
      ncol=3,  # Number of columns
      columnspacing=1.0,  # Space between columns
    )
    # Add styled grid
    a.grid(True, linestyle='--', color='gray', alpha=0.7, linewidth=0.5)
    for container in a.containers:
      a.bar_label(container, fmt='%.1f', padding=3, rotation=90)
#%%
# Create subplots

for day_change in ['_hc$', '_lc$']:
  fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(19, 11))
  axes = axes.flatten()
  agg_year_weekday = df.filter(regex=day_change+"|weekday|year").groupby(['year', 'weekday']).agg(pct)

  for i, weekday in enumerate(weekday_names):
    df_plot = agg_year_weekday[agg_year_weekday.index.get_level_values('weekday') == weekday].filter(regex=day_change)
    df_plot.plot.bar(ax=axes[i], width=0.8)

    axes[i].set_title(f'{weekday_names[i]} ')
    axes[i].set_xlabel('year')
    axes[i].set_ylabel('PCT')

    # Hide x-axis ticks for clarity
    axes[i].set_xticklabels(df_plot.index.get_level_values('year') , rotation=0)

  # Remove the empty subplot
  agg_weekday = df.filter(regex=day_change+'|weekday').groupby(['weekday']).agg(pct).reindex(weekday_names)
  agg_weekday.plot.bar(ax=axes[-1], width=0.8)
  # Hide x-axis ticks for clarity
  axes[-1].set_xticklabels(axes[-1].get_xticklabels() , rotation=0)

  format_axes(axes)

  plt.suptitle('PCT success rate', fontsize=12)
  plt.tight_layout()
  plt.savefig(f'{directory_plots}/{symbol}_all.png', bbox_inches='tight')  # High-quality save
  # plt.show()

  print(f"Done...")

