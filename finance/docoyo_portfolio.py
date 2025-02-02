# %%
import os

import yfinance as yf
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

import finplot as fplt
import mplfinance as mpf

from finance import utils

mpl.use('TkAgg')
mpl.use('QtAgg')

# %% ETFS
df_etfs = pd.read_csv('finance/etf_symbols.csv')

## %%

etf_data = {}
for etf_symbol in df_etfs['symbol']:
  filename = f'finance/y_finance_data/{etf_symbol}.csv'
  if os.path.isfile(filename):
    etf_data[etf_symbol] = pd.read_csv(filename)
    etf_data[etf_symbol]['Date'] =  pd.to_datetime(etf_data[etf_symbol]['Date'], utc=True).dt.tz_localize(None)
    # etf_data[etf_symbol] = etf_data[etf_symbol].set_index('Date').groupby(pd.Grouper(freq='7D', origin='start_day')).mean()
    etf_data[etf_symbol] = etf_data[etf_symbol].set_index('Date').groupby(pd.Grouper(freq='7D', origin='epoch')).mean().reset_index()
    continue
  etf_data[etf_symbol] = yf.Ticker(etf_symbol).history(start="2015-01-01", interval="1d")
  pd.DataFrame(etf_data[etf_symbol]).to_csv(filename)

## %%

portfolios = [k for k in df_etfs.keys() if 'p_' in k]
df_portfolios = []
for p in portfolios:
  pf = {'name': p, 'data': [], 'symbols': []}
  df_portfolios.append(pf)
  for index, row in df_etfs.iterrows():
    ## %%
    # index, row = [*df_etfs.iterrows()][4]
    if row[p] == 0:
      continue
    print(row['symbol'], p)
    pf['symbols'].append(row['description'])
    df = row[p] * etf_data[row['symbol']][['Open', 'High', 'Low', 'Close']] / etf_data[row['symbol']]['Open'][0]
    df['Date'] = etf_data[row['symbol']]['Date']
    df['Volume'] = etf_data[row['symbol']]['Volume']
    df = df.astype({'Date':'datetime64[ns]'})
    # df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'close': 'Close', 'Volume': 'volume', 'Date':'time'}, inplace=True)
    pf['data'].append(df)


##%%
df_portfolios_clean=[]
for pf in df_portfolios:
  pfc={ 'name': pf['name'], 'data': [], 'symbols': pf['symbols']}
  df_portfolios_clean.append(pfc)
  for df in pf['data']:
    pfc['data'].append(df.set_index('Date'))
  all_df = pd.concat(pfc['data'], join='inner', axis=1).T.groupby(level=0).sum().T
  pfc['data'].append(all_df)
  pfc['symbols'].append('ALL')
  for df in pfc['data']:
   df = df[df.index.isin(all_df.index)]

# for p in df_portfolios.values():
#   # Concatenate all portfolio dataframes
#   df_concatenated = pd.concat(p)
#   # Group by 'Date' and then calculate the sum of the other columns
#   # df_filtered = df_concatenated.groupby('Date').filter(lambda x: len(x) < len(p))
#   # df_sum = df_filtered.groupby('Date').sum().reset_index()
#   df_sum = df_concatenated.groupby('Date').sum().reset_index()
#   p.append(df_sum)


#%%
for pf in df_portfolios_clean:
  axs = fplt.create_plot(pf['name'], rows=len(pf['data']))
  for ax, df, sym in zip(axs, pf['data'], pf['symbols']):
    fplt.candlestick_ochl(df[['Open', 'Close', 'High', 'Low']], ax=ax)
    fplt.volume_ocv(df[['Open', 'Close', 'Volume']], ax=ax.overlay())
    fplt.add_legend(sym, ax=ax)
  fplt.show()

#%%

for pf in df_portfolios_clean:
  first = pf['data'][-1].head(1)
  last = pf['data'][-1].tail(1)
  PCT = utils.percentage_change(first['Open'].values[0], last['Close'].values[0])
  print(pf['name'], PCT)

# %%

fplt.candlestick_ochl(df[['Open', 'Close', 'High', 'Low']])
fplt.show()

# %%
fig, ax = mpf.plot(df, returnfig=True, volume=True, type='candle', tight_layout=True)
