from glob import glob

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from finance import utils

# %%
hist_data_name = '_daily_historical-data'
options_data_name = '_options-overview-history'
symbols = ['SPY']


# %%
def daily_w_volatility(symbol):
  # %%
  # if symbol not in symbols: return None

  file_prices = glob(f'finance/_data/barchart/{symbol}{hist_data_name}*.csv')
  file_volatility = glob(f'finance/_data/barchart/{symbol}{options_data_name}*.csv')

  # if len(file_prices) != 1 or len(file_volatility) != 1: return None

  df_price = pd.read_csv(file_prices[0])

  # Define a converter function
  pct_to_float = lambda x: float(x.strip('%')) / 100 if isinstance(x, str) and '%' in x else np.nan

  df_vol = pd.read_csv(file_volatility[0],
                       converters={'IV Pctl': pct_to_float, 'IV Rank': pct_to_float, 'Imp Vol': pct_to_float,
                                   '1D IV Chg': pct_to_float})

  # Remove barchart footer
  df_price = df_price.iloc[:-1]
  df_vol = df_vol.iloc[:-1]

  # Time,Open,High,Low,Latest,Change,%Change,Volume
  df_price = df_price.rename(
    columns={'Time': 'date', 'Open': 'o', 'Latest': 'c', 'High': 'h', 'Low': 'l', 'Volume': 'v'})
  df_price['date'] = pd.to_datetime(df_price.date)
  df_price.set_index('date', inplace=True)

  df_vol = df_vol.rename(
    columns={'Date': 'date', 'Imp Vol': 'iv', 'IV Rank': 'iv_rank', 'IV Pctl': 'iv_pct', 'P/C Vol': 'pc_vol',
             'Options Vol': 'opt_vol', 'P/C OI': 'pc_oi', 'Total OI': 'tot_oi', '1D IV Chg': 'iv_chg_1d'})
  df_vol['date'] = pd.to_datetime(df_vol.date)
  df_vol.set_index('date', inplace=True)

  df_comb = pd.merge(df_price[['o', 'c', 'h', 'l', 'v']], df_vol, on='date', how='outer')
  df_comb['symbol'] = symbol

  # %%
  return df_comb
