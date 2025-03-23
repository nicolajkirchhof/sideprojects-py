# %%
import datetime
import glob

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
from finance.behavior_eval.noon_to_close import df_noon, df_close

import blackscholes as bs

pd.options.plotting.backend = "matplotlib"
pd.set_option("display.max_columns", None)  # None means "no limit"
pd.set_option("display.max_rows", None)  # None means "no limit"
pd.set_option("display.width", 140)  # Set the width to 80 characters


mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2

# %%
directory = f'N:/My Drive/Projects/Trading/Research/Strategies/noon_to_close'
os.makedirs(directory, exist_ok=True)
symbols = ['DAX', 'ESTX50', 'SPX']
symbol = symbols[0]
#%%
eu_interest=pd.read_csv('finance/ECB_Interest.csv', index_col='DATE', parse_dates=True)

# risk_free_rate_year = df_treasury[df_treasury['observation_date'] <= stock_info.date.iat[0]].tail(1)['THREEFY1'].iat[0]/100
iv = df_noon.ivc.iat[0]
iv_day = iv / np.sqrt(252)
underlying = df_noon.c.iat[0]
risk_free_rate_year = eu_interest[eu_interest.index < str(df_noon.index[0].date())].iloc[-1].Main/100
multiple = 25
#%%
underlying_low = underlying - underlying * 2 * iv_day
underlying_high = underlying + underlying * 2 * iv_day
underlying_low_boundary = int(np.floor(underlying_low/multiple)*multiple)
underlying_high_boundary = int(np.ceil(underlying_high/multiple)*multiple)
strikes = range(underlying_low_boundary, underlying_high_boundary, multiple)

#%%
opts = []
for strike in strikes:
  S = underlying
  K = strike
  T = 0.5 / 365
  r = risk_free_rate_year
  sigma = iv #*np.sqrt(252)

  call = bs.BlackScholesCall(S, K, T, r, sigma)
  put = bs.BlackScholesPut(S, K, T, r, sigma)
  v_call = vars(call)
  v_call['right'] = 'C'
  v_put = vars(put)
  v_put['right'] = 'P'
  opts.append({'right':'C', 'delta': call.delta(), 'theta': call.theta(), 'gamma': call.gamma(), 'vega': call.vega(), 'price': call.price(), 'strike': K, 'pos':0})
  opts.append({'right':'P', 'delta': put.delta(), 'theta': put.theta(), 'gamma': put.gamma(), 'vega': put.vega(), 'price': put.price(), 'strike': K, 'pos':0})
  print(f'{call.price():.3f} Δ {call.delta():.3f} Θ {call.theta():.3f} Γ {call.gamma():.3f} ν {call.vega():.3f} {K} C -- {K} -- P {put.price():.3f} Δ {put.delta():.3f} Θ {put.theta():.3f} Γ {put.gamma():.3f} ν {put.vega():.3f} ')

df_opts = pd.DataFrame(opts)

#%%
# search butterfly
wing_call = df_opts[(df_opts['delta'] > 0.2) & (df_opts['right'] == 'C')].iloc[-1]
wing_call.pos = 1
atm_call = df_opts[(df_opts['strike'] < underlying) & (df_opts['right'] == 'C')].iloc[-1]
atm_call.pos = -1
wing_put = df_opts[(df_opts['delta'] < -0.2) & (df_opts['right'] == 'P')].iloc[0]
wing_put.pos = 1
atm_put = df_opts[(df_opts['strike'] > underlying) & (df_opts['right'] == 'P')].iloc[0]
atm_put.pos = -1

close = df_close.c.iat[0]
pnl = utils.options.iron_butterfly_profit_loss(close, wing_call, atm_call, atm_put, wing_put)
print(f'''
  Created butterfly 
  Noon: {underlying} 
  \t\t\tWingPut: Buy {wing_put.strike} @ {wing_put.price} 
  AtmCall:  Sell {atm_call.strike} @ -{atm_call.price}
  \t\t\tAtmPut:  Sell {atm_call.strike} @ -{atm_put.price}
  WingCall: Buy {wing_call.strike} @ {wing_call.price}
  
  Close: {close}
  PnL: {pnl:.2f}
''')
#%%

#%%


#%%
# Generate a range of stock prices
stock_prices = np.linspace(underlying_low_boundary, underlying_high_boundary, 50)  # From $560 to $575

# Calculate P/L for each stock price
pnl_values = [utils.options.iron_butterfly_profit_loss(S, wing_call, atm_call, atm_put, wing_put) for S in stock_prices]

# Plot the P/L curve
plt.figure(figsize=(10, 6))
plt.plot(stock_prices, pnl_values, label="Iron Butterfly P/L", color="blue")
plt.axhline(0, color="black", linestyle="--", linewidth=1, label="Breakeven Line")  # Breakeven line
plt.axvline(wing_put.strike, color="green", linestyle="--", label=f"Protective Put Strike (K0={wing_put.strike})")
plt.axvline(atm_put.strike, color="orange", linestyle="--", label=f"Short Put Strike (K1={atm_put.strike})")
plt.axvline(atm_call.strike, color="red", linestyle="--", label=f"Short Call Strike (K3={atm_call.strike})")
plt.axvline(wing_call.strike, color="purple", linestyle="--", label=f"Protective Call Strike (K4={wing_call.strike})")
plt.title("Iron Butterfly Profit/Loss at Expiration")
plt.xlabel("Stock Price at Expiration (S)")
plt.ylabel("Profit/Loss ($)")
plt.legend()
plt.grid()
plt.show()

#%%
# Strategy parameters
K0 = 562  # Protective Put Strike
K1 = 567  # Short Put Strike
K3 = 566  # Short Call Strike
K4 = 571  # Protective Call Strike
P0 = 0.78  # Premium Paid for Protective Put
P1 = 2.21  # Premium Received for Short Put
P3 = 2.65  # Premium Received for Short Call
P4 = 0.64  # Premium Paid for Protective Call

#%%
file = f'{directory}/{symbol}_noon_to_close.pkl'
df = pd.read_pickle(file)

#%%
df.agg({'pct_change':['mean', 'median', 'std']})

#%%
df['pct_change'].plot.hist(bins=100)
plt.show()

#%%
df_filter = df['date'] > '2025-01-01'
for i in range(4, 10):
  working = (df[df['date'] > '2025-01-01' ]['pct_change'].abs() < i/10).sum()
  print(f'{i/10}: {working} of {df_filter.sum()} {working/df_filter.sum() * 100}')

#%%
df.plot.scatter(y='pct_change', x='date')
plt.show()
#%%
def move_max(x):
  if x.loss:
    if x.type == 'long':
      return utils.pct.percentage_change(x.entry, x.stopout)
    else:
      return utils.pct.percentage_change(x.stopout, x.entry)
  else:
    if x.type == 'long':
      return utils.pct.percentage_change(x.entry, x.high)
    else:
      return utils.pct.percentage_change(x.low, x.entry)

df_follow['move'] = df_follow.apply(lambda x: utils.pct.percentage_change(x.entry, x.stopout) if x.type == 'long' else utils.pct.percentage_change(x.stopout, x.entry), axis=1)
df_follow['move_max'] = df_follow.apply(move_max, axis=1)
df_follow['move_pts'] = df_follow.apply(lambda x:  x.stopout - x.entry - 2 if x.type == 'long' else x.entry - x.stopout -2, axis=1)
#%%
df_follow['low_5'] = np.nan
for i in range(1, 5):
  df_follow[f'move_{i}'] = df_follow.apply(lambda x:  utils.pct.percentage_change( x.entry, x[f'high_{i}']) if x.type == 'long' else utils.pct.percentage_change(x[f'low_{i}'], x.entry), axis=1)
#%%
print(df_follow.groupby(['timerange', 'strategy', 'type']).agg({'move':['mean', 'median', 'std', 'sum']}))
# print(df_follow.groupby(['timerange', 'strategy', 'type']).agg({'move_1':['mean', 'median', 'std', 'sum']}))
print(df_follow.groupby(['timerange', 'strategy', 'type']).agg({'move_2':['mean', 'median', 'std', 'sum']}))
print(df_follow.groupby(['timerange', 'strategy', 'type']).agg({'move_3':['mean', 'median', 'std', 'sum']}))
# print(df_follow.groupby(['timerange', 'strategy', 'type']).agg({'move_4':['mean', 'median', 'std', 'sum']}))
#%%
df_filtred = df_follow[df_follow['timerange'].isin(['2m', '5m', '10m']) & df_follow['strategy'].isin([S_cbc_10_pct, S_cbc, S_cbc_10_pct_up])]
print(df_filtred.groupby(['timerange', 'strategy', 'type']).agg({'move':['sum'], 'move_1':['sum'], 'move_2':['sum'], 'move_3':['sum'], 'move_4':['sum']}))


# df_follow['loss'] = df_follow.apply(lambda x: x.low > x.stopout if x.type == 'long' else x.stopout > x.high, axis=1)
# %%
def pct_loss(x):
  return x.sum()/x.count()

# df_follow[df_follow['type'] == 'long'].groupby(['strategy']).agg({'loss':['sum', 'count', pct_loss]})
print(df_follow.groupby(['timerange', 'strategy', 'type']).agg({'loss':['sum', 'count', pct_loss], }))
print(df_follow.groupby(['timerange', 'strategy', 'type']).agg({'candles':['mean', 'median', 'std']}))
print(df_follow.groupby(['timerange', 'strategy', 'type']).agg({'move':['mean', 'median', 'std', 'sum']}))
print(df_follow.groupby(['timerange', 'strategy', 'type']).agg({'move_max':['mean', 'median', 'std', 'sum']}))
print(df_follow.groupby(['timerange', 'strategy', 'type']).agg({'move_pts':['mean', 'median', 'std', 'sum']}))
#%%
# print(df_follow.groupby(['timerange', 'strategy', 'type', 'candles']).agg({'loss':['sum', 'count', pct_loss]}))
df_filtred = df_follow[df_follow['timerange'].isin(['2m', '5m']) & df_follow['strategy'].isin([S_cbc_10_pct, S_cbc_10_pct_up])]
print(df_filtred.groupby(['timerange', 'strategy', 'type', 'candles']).agg({'loss':['sum', 'count', pct_loss]}))
#%%
df_follow[df_follow['loss']].agg({'move':['sum']})
# %%
df_follow.groupby(['strategy', 'type']).agg({'loss':['sum', 'count', pct_loss]})

#%%
# df_follow[(df_follow['type'] == 'long') & (~df_follow['loss'])].groupby(['strategy']).agg({'candles':['max', 'mean', 'std', 'min'], 'move':['max', 'mean', 'std', 'min']})

#%%
# df_follow[df_follow['type'] == 'long'].groupby(['strategy']).agg({'candles':['max', 'mean', 'std', 'min'], 'move':['max', 'mean', 'std', 'min']})
# print(df_follow.groupby(['type','strategy']).agg({'candles':['mean', 'median', 'std'], 'move':['mean', 'median', 'std', 'sum']}))
print(df_follow.groupby(['timerange', 'type','loss', 'strategy']).agg({'candles':['mean', 'median', 'std'], 'move':['mean', 'median', 'std', 'sum']}))
print(df_follow.groupby(['timerange', 'type','loss', 'strategy']).agg({'candles':['mean', 'median', 'std'], 'move_max':['mean', 'median', 'std', 'sum']}))

#%%

print(df_follow.groupby(['timerange', 'type','loss', 'strategy']).agg({'move_max':['mean', 'median', 'std', 'sum']}))

#%%
S_01_pct = '01_pct'
S_02_pct = '02_pct'
S_cbc = 'cbc'
S_cbc_10_pct = 'cbc_10_pct'
S_cbc_20_pct = 'cbc_20_pct'

strategies = [S_01_pct, S_02_pct, S_cbc, S_cbc_10_pct, S_cbc_20_pct]
strategiesToNumber = dict(zip(strategies,  [0, 1, 2, 3, 4]))

df_follow['strategyId'] = df_follow.apply(lambda x: strategiesToNumber[x.strategy], axis=1)


#%%
for timerange in timeranges:
  fig, ax = plt.subplots(2, 3, tight_layout=True, figsize=(24, 13))
  fig.suptitle(f'{symbol} {timerange}')
  axes = ax.flatten()
    for i, strategy in enumerate(strategies):
      scatter = df_follow[(df_follow['type'] == 'long') & (df_follow['strategy'] == strategy) & (df_follow['timerange'] == timerange)].plot.scatter(x='candles', y='move', ax=axes[i])
      axes[i].set_title(f'{symbol} {strategy}')
      plt.show()

#%%
scatter = df_follow[(df_follow['type'] == 'long') & (~df_follow['loss'])].plot.scatter(x='candles', y='move', c='strategyId', colormap='viridis')
# plt.colorbar(scatter.collections[0], label='Z  [S_01_pct, S_02_pct, S_cbc, S_cbc_10_pct, S_cbc_20_pct]')
plt.show()
