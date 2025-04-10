# %%
from datetime import datetime, timedelta

import dateutil
import numpy as np

import pandas as pd
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import mplfinance as mpf
from matplotlib.pyplot import tight_layout

from finance import utils
import pytz

pd.options.plotting.backend = "matplotlib"
pd.set_option("display.max_columns", None)  # None means "no limit"
pd.set_option("display.max_rows", None)  # None means "no limit"
pd.set_option("display.width", 140)  # Set the width to 80 characters


mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2


# %%
symbols = ['DAX', 'ESTX50', 'SPX', 'INDU', 'NDX']
# Create a directory
directory = f'N:/My Drive/Projects/Trading/Research/Strategies/thu_fri_mon'
os.makedirs(directory, exist_ok=True)

weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
symbol = symbols[1]

#%%
for symbol in symbols:
  ##%%
  os.makedirs(f'{directory}/{symbol}', exist_ok=True)
  symbol_def = utils.influx.SYMBOLS[symbol]
  tz = symbol_def['EX']['TZ']

  first_day = tz.localize(dateutil.parser.parse('2016-01-01T00:00:00'))
  # last_day = tz.localize(dateutil.parser.parse('2016-03-19T00:00:00'))
  last_day = tz.localize(dateutil.parser.parse('2025-04-09T00:00:00'))
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

  df['hh_hc'] = df.apply(lambda x: np.NAN if not x.yst_hh else x.hc, axis=1)
  df['hh_lc'] = df.apply(lambda x: np.NAN if not x.yst_hh else x.lc, axis=1)
  df['ll_hc'] = df.apply(lambda x: np.NAN if not x.yst_ll else x.hc, axis=1)
  df['ll_lc'] = df.apply(lambda x: np.NAN if not x.yst_ll else x.lc, axis=1)

  df['n_hh_hc'] = df.apply(lambda x: np.NAN if x.yst_hh else x.hc, axis=1)
  df['n_hh_lc'] = df.apply(lambda x: np.NAN if x.yst_hh else x.lc, axis=1)
  df['n_ll_hc'] = df.apply(lambda x: np.NAN if x.yst_ll else x.hc, axis=1)
  df['n_ll_lc'] = df.apply(lambda x: np.NAN if x.yst_ll else x.lc, axis=1)

  df['hh_chc'] = df.apply(lambda x: np.NAN if not x.yst_hh else x.chc, axis=1)
  df['hh_clc'] = df.apply(lambda x: np.NAN if not x.yst_hh else x.clc, axis=1)
  df['ll_chc'] = df.apply(lambda x: np.NAN if not x.yst_ll else x.chc, axis=1)
  df['ll_clc'] = df.apply(lambda x: np.NAN if not x.yst_ll else x.clc, axis=1)

  df['hh_ll_hc'] = df.apply(lambda x: np.NAN if not x.yst_hh or not x.yst_ll else x.hc, axis=1)
  df['hh_ll_lc'] = df.apply(lambda x: np.NAN if not x.yst_hh or not x.yst_ll else x.lc, axis=1)
  df['n_hh_ll_hc'] = df.apply(lambda x: np.NAN if x.yst_ll or x.yst_ll else x.hc, axis=1)
  df['n_hh_ll_lc'] = df.apply(lambda x: np.NAN if x.yst_ll or x.yst_ll else x.lc, axis=1)


##%%
  df.to_pickle(f'{directory}/{symbol}_thu_fri_mon.pkl')
  df.to_csv(f'{directory}/{symbol}_fri_mon.csv')
  # df.to_excel(f'{directory}/{symbol}_fri_mon.xlsx')
##%%
  flt =['hc', 'lc', 'hh_hc', 'hh_lc', 'll_hc', 'll_lc', 'n_hh_hc', 'n_hh_lc', 'n_ll_hc', 'n_ll_lc', 'hh_ll_hc', 'hh_ll_lc', 'n_hh_ll_hc', 'n_hh_ll_lc']

  cm = df[flt].corr()
  # cm = df[['hh_hc', 'hh_lc', 'll_hc', 'll_lc']].corr()
  # print(cm)
  cm.to_csv(f'{directory}/{symbol}_corr.csv')
  cm.to_excel(f'{directory}/{symbol}_corr.xlsx')

  ##%%

  for f in flt:
    aggs = df[df[f] == True][f'{f[-2:]}_pct'].agg(['mean', 'median', 'std'])
    print(f'{f}\n{aggs}')

  print('------------------ Close to Close --------------------')
##%%
  sigma_mult = 3
  for wd in weekday_names:
    fig, axes = plt.subplots(nrows=2, ncols=7, figsize=(24, 13), tight_layout=True)
    fig.suptitle(f'{symbol} {wd}')
    axes = axes.flatten()
    for i,f in enumerate(flt):
      aggs = df[(df[f] == True) & (df['weekday']==wd)]['cc_pct'].agg(['mean', 'median', 'std'])

      # print(f'{f}\n{aggs}')
      sigma = df[(df[f] == True) & (df['weekday']==wd)]['cc_pct'].std()
      mean = df[(df[f] == True) & (df['weekday']==wd)]['cc_pct'].mean()
      df[(df[f] == True) & (mean - sigma_mult*sigma < df.cc_pct) & (df.cc_pct < mean + sigma_mult*sigma )].cc_pct.hist(bins=100, ax=axes[i])
      num_outliers_above = df[(df[f] == True) & (df.cc_pct > mean + sigma_mult*sigma )].cc_pct.count()
      num_outliers_below = df[(df[f] == True) & (mean - sigma_mult*sigma > df.cc_pct)].cc_pct.count()
      sum_outliers_above = df[(df[f] == True) & (df.cc_pct > mean + sigma_mult*sigma )].cc_pct.sum()
      sum_outliers_below = df[(df[f] == True) & (mean - sigma_mult*sigma > df.cc_pct)].cc_pct.sum()
      axes[i].set_title(f'{f} Mean {mean:.2f} Std {sigma:.2f} \n #outliers above/below {num_outliers_above}/{num_outliers_below} \n ∑ {sum_outliers_above:.2f}/{sum_outliers_below:.2f}')
      axes[i].set_xlabel('PCT')
      axes[i].set_ylabel('num')
    plt.savefig(f'{directory}/{symbol}_{wd}_pct.png', bbox_inches='tight')  # High-quality save
    plt.close()
  # plt.show()

#%%
# Create options chain
# Pick options in 0.1 pct intervals to the downside
# Imply volatility skew
pct_test = np.arange(11)*0.1-1




#%%
day = None
for i in range(len(df)):
  if not df[flt].iloc[i, :].any():
    continue
  try:
    true_column_names = df[flt].columns[df[flt].iloc[i] == True].tolist()
    day = df.index[i]
    print(f'{symbol} {day} {true_column_names}')
    fig = utils.plots.daily_change_plot(symbol, day)
    st = fig.get_suptitle()
    fig.suptitle(st + f'\n {true_column_names}')

    date_str = day.strftime('%Y-%m-%d')
    plt.savefig(f'{directory}/{symbol}/{symbol}_{date_str}.png', bbox_inches='tight')  # High-quality save
  except Exception as e:
    print(f'{day} {e}')
  plt.close()


##%%
# evaluation
def pct(x):
  return np.round(x.sum()*100/x.count(), 2)

##%%
print(df.agg({'hh_hc': ['count','sum', pct], 'hh_lc': ['count', 'sum', pct], 'll_hc': ['count', 'sum', pct], 'll_lc': ['count', 'sum', pct]}))
print(df.groupby(['weekday']).agg({'hh_hc': ['count','sum', pct], 'hh_lc': ['count', 'sum', pct], 'll_hc': ['count', 'sum', pct], 'll_lc': ['count', 'sum', pct]}))
print(df.groupby(['year', 'weekday']).agg({'hh_hc': ['count','sum', pct], 'hh_lc': ['count', 'sum', pct], 'll_hc': ['count', 'sum', pct], 'll_lc': ['count', 'sum', pct]}))

##%%
agg_df = df[flt].agg(pct)
agg_weekday = df.groupby(['weekday'])[flt].agg(pct)
agg_year_weekday =  df.groupby(['year', 'weekday'])[flt].agg(pct)
print(agg_df)
print(agg_weekday)
print(agg_year_weekday)

agg_df.to_csv(f'{directory}/{symbol}_agg_df.csv')
agg_weekday.to_csv(f'{directory}/{symbol}_agg_weekday.csv')
agg_year_weekday.to_csv(f'{directory}/{symbol}_agg_year_weekday.csv')

agg_df.to_excel(f'{directory}/{symbol}_agg_df.xlsx')
agg_weekday.to_excel(f'{directory}/{symbol}_agg_weekday.xlsx')
agg_year_weekday.to_excel(f'{directory}/{symbol}_agg_year_weekday.xlsx')
##%%
# print(df.agg({'hh_hc': [pct], 'hh_lc': [pct], 'll_hc': [pct], 'll_lc': [pct], 'hh_chc': [pct], 'hh_clc': [pct], 'll_chc': [pct], 'll_clc': [pct]}))
# print(df.groupby(['weekday']).agg({'hh_hc': [pct], 'hh_lc': [pct], 'll_hc': [pct], 'll_lc': [pct], 'hh_chc': [pct], 'hh_clc': [pct], 'll_chc': [pct], 'll_clc': [pct]}))
# print(df.groupby(['year', 'weekday']).agg({'hh_hc': [pct], 'hh_lc': [pct], 'll_hc': [pct], 'll_lc': [pct], 'hh_chc': [pct], 'hh_clc': [pct], 'll_chc': [pct], 'll_clc': [pct]}))

##%%
# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(24, 13))
axes = axes.flatten()

# Loop through each weekday and plot
for i, weekday in enumerate(weekday_names):
  agg_year_weekday[agg_year_weekday.index.get_level_values('weekday') == weekday].plot.bar(ax=axes[i])

  # axes[i].axhline(agg_df, color='red', linestyle='--', linewidth=1.5)

  axes[i].set_title(f'{weekday_names[i]} ')
  axes[i].set_xlabel('year')
  axes[i].set_ylabel('PCT')

  # Hide x-axis ticks for clarity
  axes[i].set_xticklabels(agg_year_weekday[agg_year_weekday.index.get_level_values('weekday') == weekday].index.get_level_values('year') , rotation=0)


# Remove the empty subplot
agg_weekday.plot.bar(ax=axes[-1])

plt.suptitle('PCT success rate', fontsize=16)
plt.tight_layout()
plt.savefig(f'{directory}/{symbol}_all.png', bbox_inches='tight')  # High-quality save
# plt.show()


#%%
# df.groupby(['weekday']).agg({'hh_hc': [pct], 'hh_lc': [pct], 'll_hc': [pct], 'll_lc': [pct], 'hh_chc': [pct], 'hh_clc': [pct], 'll_chc': [pct], 'll_clc': [pct]}).plot.bar()
agg_df.plot.bar()
agg_weekday.plot.bar()
agg_year_weekday.plot.bar()
plt.show()

#%%
