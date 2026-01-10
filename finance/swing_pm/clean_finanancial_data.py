#%%
import pandas as pd
import matplotlib as mpl
from sqlalchemy import create_engine, text

from finance import utils

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2

#%%
df_liquid_symbols = pd.read_csv('finance/_data/stocks-screener-eval-stocks-etfs-01-07-2026.csv', skipfooter=1, engine='python')
df_liquid_stocks = df_liquid_symbols[df_liquid_symbols.Employees >= 0].copy().reset_index(drop=True).dropna(subset=['Symbol'])

for ticker in df_liquid_stocks.Symbol:
  #%%
  df_financial = utils.dolt_data.financial_info(ticker)

  df_financial.to_csv(f'finance/_data/financials/{ticker}.csv')
#%%

