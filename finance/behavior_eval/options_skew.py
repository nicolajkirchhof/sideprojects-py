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

import blackscholes as bs
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

df_ref_dax = df_ref[df_ref['Symbol'].isin(['DAX'])].copy()
df_ref_estx = df_ref[df_ref['Symbol'].isin(['ESTX50'])].copy()

#%%

ref_opt =  next(df_ref_dax[df_ref_dax.Pos == -1].itertuples())
#%% ATM options
comps = []
for ref_opt in df_ref_dax[df_ref_dax.Pos == -1].itertuples():
  df_candles = utils.influx.get_candles_range_raw(ref_opt.Index, ref_opt.Index + timedelta(minutes=5), ref_opt.Symbol, with_volatility=True)
  if df_candles is None:
    print('No data for ', ref_opt.Index)
    continue

  iv = df_candles.ivo.iat[0]
  risk_free_rate_year = utils.options.risk_free_rate(ref_opt.Index, 'EU')
  strike = ref_opt.Strike
  ##%%
  # underlying = df_candles.o.iat[0]
  comp_dict = {**ref_opt._asdict(), 'iv': iv, 'underlying_h': df_candles.h.iat[0], 'underlying_l': df_candles.l.iat[0]}
  for td in [0.4, 0.5, 0.6, 0.7, 0.8]:
    t = td / 365
    calc_opt_bs_h = bs.BlackScholesPut(comp_dict['underlying_h'], strike, t, risk_free_rate_year, iv) if ref_opt.Right == 'P' else bs.BlackScholesCall(comp_dict['underlying_h'], strike, t, risk_free_rate_year, iv)
    calc_opt_bs_l = bs.BlackScholesPut(comp_dict['underlying_l'], strike, t, risk_free_rate_year, iv) if ref_opt.Right == 'P' else bs.BlackScholesCall(comp_dict['underlying_l'], strike, t, risk_free_rate_year, iv)
    # calc_opt = {'right': ref_opt.Right, 'delta': calc_opt_bs.delta(), 'theta': calc_opt_bs.theta(), 'gamma': calc_opt_bs.gamma(), 'vega': calc_opt_bs.vega(), 'price': calc_opt_bs.price(), 'strike': strike}
    # print(calc_opt)
    comp_dict[f'bs_price_{td}_h'] = calc_opt_bs_h.price()
    comp_dict[f'bs_price_{td}_l'] = calc_opt_bs_l.price()
  comps.append(comp_dict)
  print(f'{ref_opt.Index.date()} {ref_opt.Symbol}')

#%%
df_atm = pd.DataFrame(comps)

#%%

pattern = 'bs_price_'
test_cols = [col for col in df_atm.columns if pattern in col]

for col in test_cols:
  df_atm[f'diff_{col}'] = df_atm[col] - df_atm.Price

#%%
df_atm.filter(regex='diff_').abs().agg(['sum', 'mean'])

#%% OTM options
comps = []
for ref_opt in df_ref_dax[df_ref_dax.Pos == 1].itertuples():
  df_candles = utils.influx.get_candles_range_raw(ref_opt.Index, ref_opt.Index + timedelta(minutes=5), ref_opt.Symbol, with_volatility=True)
  if df_candles is None:
    print('No data for ', ref_opt.Index)
    continue

  iv = df_candles.ivo.iat[0]
  risk_free_rate_year = utils.options.risk_free_rate(ref_opt.Index, 'EU')
  strike = ref_opt.Strike
  ##%%
  # underlying = df_candles.o.iat[0]
  vwap = (df_candles.h.iat[0] + df_candles.l.iat[0] +df_candles.o.iat[0] + df_candles.c.iat[0])/4
  comp_dict = {**ref_opt._asdict(), 'iv': iv, 'underlying': vwap, 'pct_dist_h': 100*(vwap-strike)/vwap}
  # for td in [0.4, 0.5, 0.6, 0.7, 0.8]:
  # for skew in range(1,50,5):
  td = 0.7
  t = td / 365
  # applied_iv = iv + iv * skew * 0.01
  applied_iv = iv
  # calc_opt_bs_h = bs.BlackScholesPut(comp_dict['underlying_h'], strike, t, risk_free_rate_year, applied_iv) if ref_opt.Right == 'P' else bs.BlackScholesCall(comp_dict['underlying_h'], strike, t, risk_free_rate_year,  applied_iv)
  calc_opt_bs = bs.BlackScholesPut(comp_dict['underlying'], strike, t, risk_free_rate_year, applied_iv) if ref_opt.Right == 'P' else bs.BlackScholesCall(comp_dict['underlying'], strike, t, risk_free_rate_year, applied_iv)
  # calc_opt = {'right': ref_opt.Right, 'delta': calc_opt_bs.delta(), 'theta': calc_opt_bs.theta(), 'gamma': calc_opt_bs.gamma(), 'vega': calc_opt_bs.vega(), 'price': calc_opt_bs.price(), 'strike': strike}
  # print(calc_opt)
  # comp_dict[f'bs_price_h_{skew}'] = calc_opt_bs_h.price()
  comp_dict[f'bs_price'] = calc_opt_bs.price()
  comps.append(comp_dict)
  print(f'{ref_opt.Index.date()} {ref_opt.Symbol}')

##%%
df_otm = pd.DataFrame(comps)

#%%
df_otm['right_num'] = df_otm.apply(lambda x: 0 if x.Right == 'C' else 1 , axis=1)
df_otm['price_diff'] = df_otm['Price'] - df_otm['bs_price']
df_otm['price_diff_pct'] = (df_otm['Price'] - df_otm['bs_price']) / df_otm['Price']
df_otm['iv_pct'] = df_otm['iv'] * df_otm['pct_dist_h'].abs()
df_otm['log_k_s'] = np.log(df_otm['Strike'] / df_otm['Price'])*df_otm['iv']

#%%

df_otm.plot.scatter(x='iv_pct', y='price_diff_pct', c='right_num', cmap='viridis')
df_otm.plot.scatter(x='iv', y='price_diff_pct')
df_otm.plot.scatter(x='pct_dist_h', y='price_diff_pct')
df_otm.plot.scatter(x='log_k_s', y='price_diff_pct', c='right_num', cmap='viridis')
plt.show()

#%%
from scipy.optimize import curve_fit

def log_function_with_offset(x, a, b, c, d):
  """
  Generic logarithmic function with offset: y = a * log(bx + c) + d

  Parameters:
  -----------
  x : array-like
      Input values
  a : float
      Scaling factor for log
  b : float
      Scaling factor inside log
  c : float
      Shift inside log (prevents log(0))
  d : float
      Vertical offset
  """
  return a * np.log(b * x + c) + d

df_otm_call = df_otm[df_otm['Right'] == 'C']
p0 = [1.0, 1.0, 1.0, df_otm_call['price_diff_pct'].mean()]
# Fit the curve
popt, pcov = curve_fit(log_function_with_offset, df_otm_call['iv_pct'], df_otm_call['price_diff_pct'], p0=p0, maxfev=10000)

# Generate predictions
# y_pred = log_function_with_offset(x, *popt)
