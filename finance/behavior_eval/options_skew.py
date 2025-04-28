# %%
from datetime import datetime, timedelta

from scipy.optimize import curve_fit
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

import blackscholes as bs
from scipy.stats import norm

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
selected_columns = ['Date','Symbol','Expiry','Right','Strike','Pos','P/L','Last','Price','Mult']
df_ref = pd.read_csv('finance/_data/Options Strategies - NoonIronButterfly.csv', parse_dates=['Date'], usecols=selected_columns)
df_ref['Date'] = df_ref['Date'].dt.tz_localize(pytz.timezone('Europe/Berlin'))
df_ref.set_index('Date', inplace=True)

#%%

df_ref_dex = df_ref[df_ref['Symbol'].isin(['DAX'])].copy()
df_ref_estx = df_ref[df_ref['Symbol'].isin(['ESTX50'])].copy()

#%%
# symbol = 'DAX'
symbol = 'ESTX50'
df_ref_symbol = df_ref[df_ref['Symbol'].isin([symbol])].copy()

# ref_opt = next(df_ref[df_ref.index == '2025-03-21 12:16:00+01:00'].itertuples())
dfs = {}
#%% ATM options
comps = []
for ref_opt in df_ref_symbol.itertuples():
  if not ref_opt.Index in dfs:
    df_candles = utils.influx.get_candles_range_raw(ref_opt.Index, ref_opt.Index + timedelta(minutes=5), ref_opt.Symbol, with_volatility=True)
  else:
    df_candles = dfs[ref_opt.Index]
  if df_candles is None:
    print('No data for ', ref_opt.Index)
    continue
  dfs[ref_opt.Index] = df_candles

  if df_candles.ivl.iat[0] * 2 < df_candles.ivh.iat[0]:
    print(f'Non-consistent IV l {df_candles.ivl.iat[0]} h {df_candles.ivh.iat[0]} using Low')
    iv = df_candles.ivl.iat[0]
  else:
    iv = (df_candles.ivo.iat[0] + df_candles.ivc.iat[0] + df_candles.ivh.iat[0] + df_candles.ivl.iat[0]) / 4
  risk_free_rate_year = utils.options.risk_free_rate(ref_opt.Index, 'EU')
  strike = ref_opt.Strike
  ##%%
  vwap = (df_candles.h.iat[0] + df_candles.l.iat[0] +df_candles.o.iat[0] + df_candles.c.iat[0])/4
  comp_dict = {**ref_opt._asdict(), 'ivh': df_candles.ivh.iat[0], 'ivl': df_candles.ivl.iat[0],  'iv': iv, 'underlying': vwap, 'pct_dist': (vwap-strike)/vwap}
  td = 0.7
  t = td / 365
  calc_opt_bs = bs.BlackScholesPut(comp_dict['underlying'], strike, t, risk_free_rate_year, iv) if ref_opt.Right == 'P' else bs.BlackScholesCall(comp_dict['underlying'], strike, t, risk_free_rate_year, iv)
  comp_dict[f'bs_price'] = calc_opt_bs.price()
  skewed_iv = utils.options.implied_volatility(comp_dict['underlying'], strike, t, risk_free_rate_year, ref_opt.Price, ref_opt.Right)
  comp_dict[f'real_iv'] = skewed_iv
  comps.append(comp_dict)
  print(f'{ref_opt.Index.date()} {ref_opt.Symbol}')

df_all = pd.DataFrame(comps)

df_all['right_num'] = df_all.apply(lambda x: 0 if x.Right == 'C' else 1 , axis=1)
df_all['price_diff'] = df_all['Price'] - df_all['bs_price']
df_all['price_diff_pct'] = (df_all['Price'] - df_all['bs_price']) / df_all['Price']
df_all['iv_pct'] = df_all['iv'] * df_all['pct_dist'].abs()
df_all['log_k_s'] = np.log(df_all['Strike'] / df_all['underlying'])*df_all['iv']
df_all['log_mn'] = (np.log(df_all['Strike'] / df_all['underlying']) ** 2) * df_all['iv']
df_all['iv_diff_pct'] = (df_all['real_iv'] - df_all['iv']) / df_all['iv']
df_all['mny'] = np.log(df_all['Strike'] / df_all['underlying'])**2

#%%
def plot_relations(df, title):
  fig, ax = plt.subplots(3, 3, tight_layout=True, figsize=(24, 14))
  axes = ax.ravel()
  df.plot.scatter(ax=axes[0], x='iv_pct', y='price_diff_pct', c='right_num', cmap='viridis')
  df.plot.scatter(ax=axes[1],x='iv', y='price_diff_pct', c='right_num', cmap='viridis')
  df.plot.scatter(ax=axes[2],x='iv', y='iv_diff_pct', c='right_num', cmap='viridis')
  df.plot.scatter(ax=axes[3],x='iv', y='real_iv', c='right_num', cmap='viridis')
  df.plot.scatter(ax=axes[4],x='price_diff_pct', y='iv_diff_pct', c='right_num', cmap='viridis')
  df.plot.scatter(ax=axes[5],x='log_k_s', y='iv_diff_pct', c='right_num', cmap='viridis')
  df.plot.scatter(ax=axes[6],x='log_mn', y='iv_diff_pct', c='right_num', cmap='viridis')
  df.plot.scatter(ax=axes[7],x='mny', y='real_iv', c='right_num', cmap='viridis')
  df.plot.scatter(ax=axes[8],x='mny', y='iv_diff_pct', c='right_num', cmap='viridis')

  fig.suptitle(title)
  plt.show()

#%%
df_atm = df_all[df_all['Pos'] == -1].copy()
# df_atm = df_all[df_all['iv_pct'] < 0.0005].copy()
plot_relations(df_atm, f'{symbol} ATM options')

#%%
df_otm = df_all[df_all['Pos'] == 1].copy()
# df_otm = df_all[df_all['iv_pct'] >= 0.0005].copy()
plot_relations(df_otm, f'{symbol} OTM options')

#%%
plot_relations(df_all, f'{symbol} ALL options')

#%%

x_col = 'iv'
y_col = 'price_diff_pct'

# df_atm_call = df_atm[df_atm['Right'] == 'C']
# popt_atm_call, r_squared_atm_call, df_fitted_atm_call = utils.fitlog.fit_log_curve_df(df_atm_call, x_col, y_col)
# utils.fitlog.analyze_residuals_df(df_fitted_atm_call, x_col, y_col, 'predicted')
#
# df_atm_put = df_atm[df_atm['Right'] == 'P']
# popt_atm_put, r_squared_atm_put, df_fitted_atm_put = utils.fitlog.fit_log_curve_df(df_atm_put, x_col, y_col)
# utils.fitlog.analyze_residuals_df(df_fitted_atm_put, x_col, y_col, 'predicted')

popt_atm, r_squared_atm, df_fitted_atm = utils.fitlog.fit_log_curve_df(df_atm, x_col, y_col)
utils.fitlog.analyze_residuals_df(df_fitted_atm, x_col, y_col, 'predicted')


#%%

x_col = 'mny'
y_col = 'price_diff_pct'

# df_otm_call = df_otm[df_otm['Right'] == 'C']
# popt_otm_call, r_squared_otm_call, df_fitted_otm_call = utils.fitlog.fit_log_curve_df(df_otm_call, x_col, y_col)
# utils.fitlog.analyze_residuals_df(df_fitted_otm_call, x_col, y_col, 'predicted')
#
# df_otm_put = df_otm[df_otm['Right'] == 'P']
# popt_otm_put, r_squared_otm_put, df_fitted_otm_put = utils.fitlog.fit_log_curve_df(df_otm_put, x_col, y_col)
# utils.fitlog.analyze_residuals_df(df_fitted_otm_put, x_col, y_col, 'predicted')

popt_otm, r_squared_otm, df_fitted_otm = utils.fitlog.fit_log_curve_df(df_otm, x_col, y_col)
utils.fitlog.analyze_residuals_df(df_fitted_otm, x_col, y_col, 'predicted')


#%% Backtest results

def apply_skew(row):
  # ATM
  if row.iv_pct < 0.0005:
    pct_diff_fct = utils.fitlog.log_function_with_offset(row.iv, *popt_atm)
    return utils.options.option_bs_price_correction(row.bs_price, pct_diff_fct)
  else:
    pct_diff_fct = utils.fitlog.log_function_with_offset(row.iv_pct, *popt_otm)
    return utils.options.option_bs_price_correction(row.bs_price, pct_diff_fct)

df_all['bs_price_skew'] = df_all.apply(apply_skew, axis=1)
# skew = df_all.apply(apply_skew, axis=1)

print((df_all.Price - df_all.bs_price_skew).describe())
print((df_all.Price - df_all.bs_price).describe())

#%%
o = bs.BlackScholesPut(4780, 4705, 0.7/365, 0.265, 0.50278096)
print(o.price())

#%%
def implied_volatility(S, K, T, r, market_price, option_type="call", tol=1e-6, max_iter=1000):
  """
  Calculate implied volatility using the bisection method.

  Parameters:
  - S: Current stock price (float)
  - K: Strike price (float)
  - T: Time to maturity in years (float)
  - r: Risk-free interest rate (float, e.g., 0.05 for 5%)
  - market_price: The actual (market) price of the option (float)
  - option_type: "call" or "put" (default="call")
  - tol: Tolerance for stopping the iteration (default=1e-6)
  - max_iter: Maximum number of iterations (default=1000)

  Returns:
  - Implied volatility (float).
  """
  # Initial bounds for volatility
  lower_vol = 1e-5    # Volatility cannot be zero
  upper_vol = 5.0     # Arbitrary high value for initial upper bound

  for i in range(max_iter):
    # Calculate the midpoint
    mid_vol = (lower_vol + upper_vol) / 2.0
    # Calculate the theoretical price with the mid volatility
    option = bs.BlackScholesCall(S, K, T, r, mid_vol) if option_type == "call" else bs.BlackScholesPut(S, K, T, r, mid_vol)
    theoretical_price = option.price()

    # Check the difference between theoretical price and market price
    price_diff = theoretical_price - market_price

    # If the difference is within the tolerance, return the implied volatility
    if abs(price_diff) < tol:
      return mid_vol

    # Adjust the bounds
    if price_diff > 0:
      # If theoretical price is too high, reduce upper bound
      upper_vol = mid_vol
    else:
      # If theoretical price is too low, raise lower bound
      lower_vol = mid_vol

  # If no result is found, raise an exception
  raise ValueError("Implied volatility did not converge within the given iterations.")

implied_volatility(4780, 4705, 0.7/365, 0.265, 15.2, option_type="put")
