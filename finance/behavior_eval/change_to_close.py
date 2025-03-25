# %%
from datetime import datetime, timedelta

import dateutil
import numpy as np
import scipy

import pandas as pd
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import mplfinance as mpf

import finance.utils as utils

pd.options.plotting.backend = "matplotlib"

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2

# %%
directory = f'N:/My Drive/Projects/Trading/Research/Strategies/noon_to_close'
os.makedirs(directory, exist_ok=True)
# symbol = 'IBDE40'
# symbols = [('IBDE40', pytz.timezone('Europe/Berlin')), ('IBGB100', pytz.timezone('Europe/London')),
#            *[(x, pytz.timezone('America/New_York')) for x in ['IBUS30', 'IBUS500', 'IBUST100']]]
# symbols = ['IBDE40', 'IBEU50', 'IBUS500']
symbols = ['DAX', 'ESTX50', 'SPX']

symbol = symbols[0]
for symbol in symbols:
  ##%%
  # Create a directory
  symbol_def = utils.influx.SYMBOLS[symbol]
  tz = symbol_def['EX']['TZ']

  first_day = tz.localize(dateutil.parser.parse('2022-01-02T00:00:00'))
  # first_day = tz.localize(dateutil.parser.parse('2025-03-14T00:00:00'))
  last_day = tz.localize(dateutil.parser.parse('2025-03-20T00:00:00'))
  day_start = first_day
  prior_close = None
  dfs = []
  offset = timedelta(minutes=5)
  ## %%
  while day_start < last_day:
    # get the following data for daily assignment
    noon = day_start + symbol_def['EX']['Open'] + timedelta(hours=3)
    close_time = day_start + symbol_def['EX']['Close']
    df_noon = utils.influx.get_candles_range_aggregate(day_start + timedelta(hours=9), noon + offset, symbol, '1h')
    day_end = day_start + timedelta(days=1)
    # df_day = utils.influx.get_candles_range_aggregate(day_start, day_start + timedelta(days=1), symbol, '1h')
    if df_noon is None or 'h' not in df_noon.columns or 'ivc' not in df_noon.columns:
      print(f'no data for {day_start.isoformat()}')
      day_start = day_end
      continue
    df_close = utils.influx.get_candles_range_aggregate(noon, close_time, symbol)
    if df_close is None:
      print(f'no data for close')
      day_start = day_end
      continue
    noon = day_start + symbol_def['EX']['Open'] + timedelta(hours=3)
    noon_value = df_noon.iloc[-1].c
    end_of_day = df_close.iloc[-1].c
    pct_change = (end_of_day - noon_value) / noon_value * 100
    noon_hvc = df_noon.hvc if 'hvc' in df_noon else np.nan
    close_hvc = df_close.hvc if 'hvc' in df_close else np.nan
    dfs.append({'date': day_start,'noon': noon_value, 'close': end_of_day, 'pct_change': pct_change, 'symbol': symbol,
                'noon_hv':noon_hvc, 'noon_iv': df_noon.ivc, 'close_hv':close_hvc, 'close_iv': df_close.ivc})
    day_start = day_end
    print(f'{datetime.now().isoformat()}: Done {symbol} {day_start.isoformat()}')

  df_all = pd.DataFrame(dfs)
  df_all.to_pickle(f'{directory}/{symbol}_noon_to_close.pkl')

# %%
mpf.plot(df_day, style='yahoo', figsize=(20, 12), tight_layout=True, type='candle',
         columns=utils.influx.MPF_COLUMN_MAPPING, xrotation=0, datetime_format='%H:%M')
date_str = day_start.strftime('%Y-%m-%d')
prior_close_str = f'Prior Close: {prior_close:.2f}' if prior_close is not None else 'N/A'
plt.gcf().suptitle(f'{symbol} {date_str} {prior_close_str}')

# %%
longs = df_follow[(df_follow['type'] == 'long') & (df_follow['strategy'] == S_cbc)]
longs_df = pd.DataFrame(index=df_day.index)
longs_df['entry'] = float('nan')
longs_df.loc[longs['start'].tolist(), 'entry'] = longs['entry'].tolist()  # Fill
longs_df['exit'] = float('nan')
longs_df.loc[longs['end'].tolist(), 'exit'] = longs['stopout'].tolist()  # Fill

shorts = df_follow[(df_follow['type'] == 'short') & (df_follow['strategy'] == S_cbc)]
shorts_df = pd.DataFrame(index=df_day.index)
shorts_df['entry'] = float('nan')
shorts_df.loc[shorts['start'].tolist(), 'entry'] = shorts['entry'].tolist()  # Fill
shorts_df['exit'] = float('nan')
shorts_df.loc[shorts['end'].tolist(), 'exit'] = shorts['stopout'].tolist()  # Fill

add_plot = [
  mpf.make_addplot(longs_df['entry'], color="blue", marker='^', type="scatter", width=1.5),
  mpf.make_addplot(longs_df['exit'], color="orange", marker="v", type="scatter", width=1.5),
  mpf.make_addplot(shorts_df['entry'], color="cyan", marker='v', type="scatter", width=1.5),
  mpf.make_addplot(shorts_df['exit'], color="brown", marker='^', type="scatter", width=1.5),
]

mpf.plot(df_day, style='yahoo', addplot=add_plot, figsize=(20, 12), tight_layout=True, type='candle',
         columns=utils.influx.MPF_COLUMN_MAPPING, xrotation=0, datetime_format='%H:%M')
date_str = day_start.strftime('%Y-%m-%d')
prior_close_str = f'Prior Close: {prior_close:.2f}' if prior_close is not None else 'N/A'
plt.gcf().suptitle(f'{symbol} {date_str} {timerange} {prior_close_str}')
plt.show()
# # plt.savefig(f'{directory}/{symbol}_{date_str}.png', bbox_inches='tight')  # High-quality save
# # plt.savefig(f'finance/_data/dax_plots_mpl/IBDE40_{date_str}.png', dpi=300, bbox_inches='tight')  # High-quality save
# plt.show()
# # plt.close()
