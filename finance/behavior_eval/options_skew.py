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
df_ref = df_ref_estx

#%% ATM options
comps = []
for ref_opt in df_ref.itertuples():
  df_candles = utils.influx.get_candles_range_raw(ref_opt.Index, ref_opt.Index + timedelta(minutes=5), ref_opt.Symbol, with_volatility=True)
  if df_candles is None:
    print('No data for ', ref_opt.Index)
    continue

  iv = (df_candles.ivo.iat[0] + df_candles.ivc.iat[0] + df_candles.ivh.iat[0] + df_candles.ivl.iat[0]) / 4
  risk_free_rate_year = utils.options.risk_free_rate(ref_opt.Index, 'EU')
  strike = ref_opt.Strike
  ##%%
  vwap = (df_candles.h.iat[0] + df_candles.l.iat[0] +df_candles.o.iat[0] + df_candles.c.iat[0])/4
  comp_dict = {**ref_opt._asdict(), 'iv': iv, 'underlying': vwap, 'pct_dist': (vwap-strike)/vwap}
  td = 0.7
  t = td / 365
  calc_opt_bs = bs.BlackScholesPut(comp_dict['underlying'], strike, t, risk_free_rate_year, iv) if ref_opt.Right == 'P' else bs.BlackScholesCall(comp_dict['underlying'], strike, t, risk_free_rate_year, iv)
  comp_dict[f'bs_price'] = calc_opt_bs.price()
  comps.append(comp_dict)
  print(f'{ref_opt.Index.date()} {ref_opt.Symbol}')

df_all = pd.DataFrame(comps)

df_all['right_num'] = df_all.apply(lambda x: 0 if x.Right == 'C' else 1 , axis=1)
df_all['price_diff'] = df_all['Price'] - df_all['bs_price']
df_all['price_diff_pct'] = (df_all['Price'] - df_all['bs_price']) / df_all['Price']
df_all['iv_pct'] = df_all['iv'] * df_all['pct_dist'].abs()
df_all['log_k_s'] = np.log(df_all['Strike'] / df_all['Price'])*df_all['iv']

##%%
df_atm = df_all[df_all['Pos'] == -1].copy()

df_atm.plot.scatter(x='iv_pct', y='price_diff_pct', c='right_num', cmap='viridis')
df_atm.plot.scatter(x='iv', y='price_diff_pct', c='right_num', cmap='viridis')
df_atm.plot.scatter(x='pct_dist', y='price_diff_pct', c='right_num', cmap='viridis')
df_atm.plot.scatter(x='log_k_s', y='price_diff_pct', c='right_num', cmap='viridis')
plt.show()

#%%

x_col = 'iv'
y_col = 'price_diff_pct'

df_atm_call = df_atm[df_atm['Right'] == 'C']
popt_atm_call, r_squared_atm_call, df_fitted_atm_call = utils.fitlog.fit_log_curve_df(df_atm_call, x_col, y_col)
utils.fitlog.analyze_residuals_df(df_fitted_atm_call, x_col, y_col, 'predicted')

df_atm_put = df_atm[df_atm['Right'] == 'P']
popt_atm_put, r_squared_atm_put, df_fitted_atm_put = utils.fitlog.fit_log_curve_df(df_atm_put, x_col, y_col)
utils.fitlog.analyze_residuals_df(df_fitted_atm_put, x_col, y_col, 'predicted')

popt_atm, r_squared_atm, df_fitted_atm = utils.fitlog.fit_log_curve_df(df_atm, x_col, y_col)
utils.fitlog.analyze_residuals_df(df_fitted_atm, x_col, y_col, 'predicted')

##%%
df_otm = df_all[df_all['Pos'] == 1].copy()
#%%
df_otm['right_num'] = df_otm.apply(lambda x: 0 if x.Right == 'C' else 1 , axis=1)
df_otm['price_diff'] = df_otm['Price'] - df_otm['bs_price']
df_otm['price_diff_pct'] = (df_otm['Price'] - df_otm['bs_price']) / df_otm['Price']
df_otm['iv_pct'] = df_otm['iv'] * df_otm['pct_dist'].abs()
df_otm['log_k_s'] = np.log(df_otm['Strike'] / df_otm['Price'])*df_otm['iv']

#%%
df_otm.plot.scatter(x='iv_pct', y='price_diff_pct', c='right_num', cmap='viridis')
df_otm.plot.scatter(x='iv', y='price_diff_pct', c='right_num', cmap='viridis')
df_otm.plot.scatter(x='pct_dist', y='price_diff_pct', c='right_num', cmap='viridis')
df_otm.plot.scatter(x='log_k_s', y='price_diff_pct', c='right_num', cmap='viridis')

plt.show()

#%%

x_col = 'iv_pct'
y_col = 'price_diff_pct'

df_otm_call = df_otm[df_otm['Right'] == 'C']
popt_otm_call, r_squared_otm_call, df_fitted_otm_call = utils.fitlog.fit_log_curve_df(df_otm_call, x_col, y_col)
utils.fitlog.analyze_residuals_df(df_fitted_otm_call, x_col, y_col, 'predicted')

df_otm_put = df_otm[df_otm['Right'] == 'P']
popt_otm_put, r_squared_otm_put, df_fitted_otm_put = utils.fitlog.fit_log_curve_df(df_otm_put, x_col, y_col)
utils.fitlog.analyze_residuals_df(df_fitted_otm_put, x_col, y_col, 'predicted')

popt_otm, r_squared_otm, df_fitted_otm = utils.fitlog.fit_log_curve_df(df_otm, x_col, y_col)
utils.fitlog.analyze_residuals_df(df_fitted_otm, x_col, y_col, 'predicted')

#%% ALL options
comps = []
for ref_opt in df_ref.itertuples():
  df_candles = utils.influx.get_candles_range_raw(ref_opt.Index, ref_opt.Index + timedelta(minutes=5), ref_opt.Symbol, with_volatility=True)
  if df_candles is None:
    print('No data for ', ref_opt.Index)
    continue

  iv = (df_candles.ivo.iat[0] + df_candles.ivc.iat[0] + df_candles.ivh.iat[0] + df_candles.ivl.iat[0]) / 4
  risk_free_rate_year = utils.options.risk_free_rate(ref_opt.Index, 'EU')
  strike = ref_opt.Strike
  ##%%
  vwap = (df_candles.h.iat[0] + df_candles.l.iat[0] +df_candles.o.iat[0] + df_candles.c.iat[0])/4
  comp_dict = {**ref_opt._asdict(), 'iv': iv, 'underlying': vwap, 'pct_dist': (vwap-strike)/vwap}
  td = 0.7
  t = td / 365
  calc_opt_bs = bs.BlackScholesPut(comp_dict['underlying'], strike, t, risk_free_rate_year, iv) if ref_opt.Right == 'P' else bs.BlackScholesCall(comp_dict['underlying'], strike, t, risk_free_rate_year, iv)
  comp_dict[f'bs_price'] = calc_opt_bs.price()
  comps.append(comp_dict)
  print(f'{ref_opt.Index.date()} {ref_opt.Symbol}')

df_all = pd.DataFrame(comps)

df_all['right_num'] = df_all.apply(lambda x: 1 * x.Pos if x.Right == 'C' else 2 * x.Pos, axis=1)
df_all['price_diff'] = df_all['Price'] - df_all['bs_price']
df_all['price_diff_pct'] = (df_all['Price'] - df_all['bs_price']) / df_all['Price']
df_all['iv_pct'] = df_all['iv'] * df_all['pct_dist'].abs()
df_all['log_k_s'] = np.log(df_all['Strike'] / df_all['Price'])*df_all['iv']
#%%
df_all.plot.scatter(x='iv_pct', y='price_diff_pct', c='right_num', cmap='viridis')
df_all.plot.scatter(x='iv', y='price_diff_pct', c='right_num', cmap='viridis')
df_all.plot.scatter(x='pct_dist', y='price_diff_pct', c='right_num', cmap='viridis')
df_all.plot.scatter(x='log_k_s', y='price_diff_pct', c='right_num', cmap='viridis')

plt.show()
#%%
x_col = 'iv_pct'
y_col = 'price_diff_pct'

df_all_call = df_all[df_all['Right'] == 'C']
popt_all_call, r_squared_all_call, df_fitted_all_call = utils.fitlog.fit_log_curve_df(df_all_call, x_col, y_col)
utils.fitlog.analyze_residuals_df(df_fitted_all_call, x_col, y_col, 'predicted')

df_all_put = df_all[df_all['Right'] == 'P']
popt_all_put, r_squared_all_put, df_fitted_all_put = utils.fitlog.fit_log_curve_df(df_all_put, x_col, y_col)
utils.fitlog.analyze_residuals_df(df_fitted_all_put, x_col, y_col, 'predicted')

popt_all, r_squared_all, df_fitted_all = utils.fitlog.fit_log_curve_df(df_all, x_col, y_col)
utils.fitlog.analyze_residuals_df(df_fitted_all, x_col, y_col, 'predicted')


#%% Backtest results

def apply_skew(row):
  # ATM
  if row.iv_pct < 0.0005:
    return utils.options.option_bs_price_correction(row.bs_price, utils.fitlog.log_function_with_offset(row.iv, *popt_atm))
  else:
    return utils.options.option_bs_price_correction(row.bs_price, utils.fitlog.log_function_with_offset(row.iv_pct, *popt_otm))

df_all['bs_price_skew'] = df_all.apply(apply_skew, axis=1)
