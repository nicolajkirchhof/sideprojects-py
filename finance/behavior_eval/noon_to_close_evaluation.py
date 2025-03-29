# %%
from datetime import datetime
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
pd.set_option("display.max_columns", None)  # None means "no limit"
pd.set_option("display.max_rows", None)  # None means "no limit"
pd.set_option("display.width", 140)  # Set the width to 80 characters


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
symbol = symbols[1]

#%%
file = f'{directory}/{symbol}_noon_to_close.pkl'
df = pd.read_pickle(file)
#%%
for field in ['noon_iv', 'noon_hv', 'close_iv', 'close_hv']:
  df[field] = df[field].apply(lambda x: x.values[0] if type(x) == pd.Series else x)

df.to_pickle(file)
#%%
row = list(df.itertuples())[-4]
results = []
# is_log = False
is_log = True
for row in df.itertuples():
# iv = 0.16
# underlying = 22891.4
# risk_free_rate_year = eu_interest[eu_interest.index < '2025-03-23'].iloc[-1].Main/100
# S = underlying
# T = 3 / 365

# row = list(pd.DataFrame([{'noon_iv': 0.1761, 'noon':22987, 'close_iv':0.18091, 'close':22847, 'date':datetime.now()}]).itertuples())[0]
# row = list(pd.DataFrame([{'noon_iv': 0.17667, 'noon':23080, 'close_iv':0.18091, 'close':23014, 'date':datetime.now()}]).itertuples())[0]
row = list(pd.DataFrame([{'noon_iv': 0.29, 'noon':5390.4, 'close_iv':0.18091, 'close':5390, 'date':datetime.now()}]).itertuples())[0]
#%%
  eu_interest=pd.read_csv('finance/ECB_Interest.csv', index_col='DATE', parse_dates=True)

  # risk_free_rate_year = df_treasury[df_treasury['observation_date'] <= stock_info.date.iat[0]].tail(1)['THREEFY1'].iat[0]/100
  iv = row.noon_iv if row.noon_iv > 0.05 else row.noon_iv * np.sqrt(252)
  iv_day = iv / np.sqrt(252)
  underlying = row.noon
  close = row.close
  risk_free_rate_year = eu_interest[eu_interest.index < str(row.date.date())].iloc[-1].Main/100
  # multiple = 25
  multiple = 5
  T = 0.5 / 365
  ##%%
  underlying_low = underlying - underlying * 2 * iv_day
  underlying_high = underlying + underlying * 2 * iv_day
  underlying_low_boundary = int(np.floor(underlying_low/multiple)*multiple)
  underlying_high_boundary = int(np.ceil(underlying_high/multiple)*multiple)
  strikes = range(underlying_low_boundary, underlying_high_boundary, multiple)

  ##%%

  opts = []
  for strike in strikes:
    K = strike

    call = bs.BlackScholesCall(underlying, K, T, risk_free_rate_year, iv)
    put = bs.BlackScholesPut(underlying, K, T, risk_free_rate_year, iv)
    v_call = vars(call)
    v_call['right'] = 'C'
    v_put = vars(put)
    v_put['right'] = 'P'
    opts.append({'right':'C', 'delta': call.delta(), 'theta': call.theta(), 'gamma': call.gamma(), 'vega': call.vega(), 'price': call.price(), 'strike': K, 'pos':0})
    opts.append({'right':'P', 'delta': put.delta(), 'theta': put.theta(), 'gamma': put.gamma(), 'vega': put.vega(), 'price': put.price(), 'strike': K, 'pos':0})
    if is_log:
      print(f'{call.vega():.3f} ν {call.gamma():.3f} Γ {call.theta():.3f} Θ {call.delta():.3f} Δ {call.price():.3f} C -- {K} -- P {put.price():.3f} Δ {put.delta():.3f} Θ {put.theta():.3f} Γ {put.gamma():.3f} ν {put.vega():.3f} ')

  delta_cutoff = 0.2
  df_opts = pd.DataFrame(opts)

  ##%%
  # search butterfly
  wing_call = df_opts[(df_opts['delta'] > -delta_cutoff) & (df_opts['right'] == 'C')].iloc[-1]
  atm_call = df_opts[(df_opts['strike'] < underlying) & (df_opts['right'] == 'C')].iloc[-1]
  wing_put = df_opts[(df_opts['delta'] < delta_cutoff) & (df_opts['right'] == 'P')].iloc[0]
  atm_put = df_opts[(df_opts['strike'] > underlying) & (df_opts['right'] == 'P')].iloc[0]

  pnl = utils.options.iron_butterfly_profit_loss(close, wing_call, atm_call, atm_put, wing_put)

  trade = {'date':row.date, 'pnl':pnl, 'underlying':underlying, 'close':close,
           'wing_call':wing_call.strike, 'wing_put':wing_put.strike,
           'atm_call':atm_call.strike, 'atm_put':atm_put.strike,
           'noon_iv':row.noon_iv, 'close_iv':row.close_iv}
  results.append(trade)
  if is_log:
    print(f'''
      Created butterfly
      Noon: {underlying}
      \t\t\tWingPut: Buy {wing_put.strike} @ {wing_put.price}
      AtmCall:  Sell {atm_call.strike} @ -{atm_call.price}
      \t\t\tAtmPut:  Sell {atm_put.strike} @ -{atm_put.price}
      WingCall: Buy {wing_call.strike} @ {wing_call.price}

      Close: {close}
      PnL: {pnl:.2f}
    ''')
#%%
results_df = pd.DataFrame(results)
file = f'{directory}/{symbol}_noon_to_close_results_0_1.pkl'
results_df.to_pickle(file)
#%%
file = f'{directory}/dax_noon_to_close_results.pkl'
results_df_dax = pd.read_pickle(file)
#%%
file = f'{directory}/dax_noon_to_close_results_0_1.pkl'
results_df_dax_01 = pd.read_pickle(file)

#%%
file = f'{directory}/{symbol}_noon_to_close_results.pkl'
results_df_dax = pd.read_pickle(file)

#%%
df_pnl_dax = results_df_dax[['date', 'pnl']].copy()
df_pnl_estx = results_df_estx[['date', 'pnl']].copy()

df_pnl_dax.rename(columns={'pnl': 'pnl_dax'}, inplace=True)
df_pnl_dax['date'] = df_pnl_dax['date'].apply(lambda x: x.date())
df_pnl_estx.rename(columns={'pnl': 'pnl_estx'}, inplace=True)
df_pnl_estx['date'] = df_pnl_estx['date'].apply(lambda x: x.date())

df_comb = pd.merge(df_pnl_dax, df_pnl_estx, on='date', how='inner')

df_comb['sign_dax'] = df_comb.pnl_dax.apply(np.sign)
df_comb['sign_estx'] = df_comb.pnl_estx.apply(np.sign)
#%%
results_df_dax['pct_change'] = results_df_dax.apply(lambda x: (x.close - x.underlying) * 100 / x.underlying, axis=1)
results_df_dax.groupby(['year', 'week']).agg({'pnl':['sum'], 'pct_change': ['mean', 'median', 'std']})

#%%
longest = 0
chain = 0
for row in results_df.itertuples():
  if row.pnl < 0:
    chain += 1
  else:
    longest = max(longest, chain)
    chain = 0


#%%
print(f'Wins {(results_df.pnl < 0).sum()} Losses {(results_df.pnl > 0).sum()}')
print(f'AvgWin {results_df[results_df.pnl > 0].pnl.mean():.2f} AvgLoss {results_df[results_df.pnl < 0].pnl.mean():.2f}')

#%%
# results_df = results_df_dax_01
results_df = results_df_dax
results_df['year'] = results_df['date'].apply(lambda x: x.year)
results_df['month'] = results_df['date'].apply(lambda x: x.month)
results_df['week'] = results_df['date'].apply(lambda x: x.strftime('%U'))
results_df['weekday'] = results_df['date'].apply(lambda x: x.strftime('%A'))
# results_df['iv_diff'] = results_df['close_iv'] - results_df['noon_iv']
#%%
results_df.groupby(['year', 'week']).agg({'pnl':['sum'], 'noon_iv': ['mean', 'median', 'std'], 'close_iv': ['mean', 'median', 'std']})
results_df.groupby(['year', 'month', 'weekday']).agg({'pnl':['sum'], 'noon_iv': ['mean', 'median', 'std'], 'close_iv': ['mean', 'median', 'std']})
results_df.groupby(['year', 'month']).agg({'pnl':['sum'], 'noon_iv': ['mean', 'median', 'std'], 'close_iv': ['mean', 'median', 'std']})
results_df[results_df.month > 3].groupby(['year']).agg({'pnl':['sum'], 'noon_iv': ['mean', 'median', 'std'], 'close_iv': ['mean', 'median', 'std']})
#%%
results_df.groupby(['weekday']).agg({'pnl':['sum'], 'noon_iv': ['mean', 'median', 'std'], 'close_iv': ['mean', 'median', 'std']})
#%%
for year in results_df['year'].unique():
  results_df[results_df.year == year].plot.scatter(x='noon_iv', y='pnl')
  plt.gcf().suptitle(f'{symbol} {year} Iron Butterfly P/L at Expiration PNL {results_df[results_df.year == year].pnl.sum():.2f}', fontsize=16)

plt.show()

#%%
results_df.plot.scatter(x='date', y='pnl', c='noon_iv', cmap='viridis')
plt.gcf().suptitle(f'{symbol} Iron Butterfly P/L at Expiration PNL {results_df.pnl.sum():.2f}', fontsize=16)
plt.show()
#%%
# wing_call = pd.Series({'strike': 571, 'price': 0.64})
# wing_put = pd.Series({'strike': 562, 'price': 0.78})
# atm_call = pd.Series({'strike': 566, 'price': 2.65})
# atm_put = pd.Series({'strike': 567, 'price': 2.21})

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

#%%
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

