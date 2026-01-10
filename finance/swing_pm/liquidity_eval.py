#%%
import glob
import pickle
import pandas as pd

import matplotlib as mpl
from sqlalchemy import create_engine, text

from finance import utils
from finance.swing_pm.earnings_data_processing import liquid_symbols

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2


#%%

ticker = 'MSFT'
# MySQL connection setup (localhost:3306)
# Note: This requires a driver like 'pymysql'. Install it via: pip install pymysql
# Format: mysql+driver://user:password@host:port/database
db_connection_str = 'mysql+pymysql://root:@localhost:3306/stocks'
db_connection = create_engine(db_connection_str)
# for ticker in utils.underlyings.us_stock_symbols:

#%% SQL-side Quantile Calculation for ALL tickers
# Using Window Functions partitioned by symbol to get stats for everyone at once
all_stats_query = text("""
                       WITH Ordered AS (
                           SELECT
                               act_symbol,
                               volume,
                               ROW_NUMBER() OVER (PARTITION BY act_symbol ORDER BY volume) as rn,
                               COUNT(*) OVER (PARTITION BY act_symbol) as total
                           FROM ohlcv
                           WHERE date >= '2020-01-01'
                       )
                       SELECT
                           act_symbol as symbol,
                           MAX(CASE WHEN rn = 1 THEN volume END) as min_val,
                           MAX(CASE WHEN rn = GREATEST(1, FLOOR(total * 0.25)) THEN volume END) as q1,
                           MAX(CASE WHEN rn = GREATEST(1, FLOOR(total * 0.50)) THEN volume END) as median,
                           MAX(CASE WHEN rn = GREATEST(1, FLOOR(total * 0.75)) THEN volume END) as q3,
                           MAX(CASE WHEN rn = total THEN volume END) as max_val
                       FROM Ordered
                       GROUP BY act_symbol
                       """)

df_all_stats = pd.read_sql(all_stats_query, db_connection)
# Set symbol as index for easier lookups
df_all_stats.set_index('symbol', inplace=True)
print(df_all_stats.head())
df_all_stats.to_pickle(f'finance/_data/stocks_liquidity_stats.pkl')
#%%
df_all_stats = pd.read_pickle(f'finance/_data/stocks_liquidity_stats.pkl')
liquid_names = df_all_stats[df_all_stats['median'] > 1000000].index.tolist()
df_liquid_symbols = pd.read_csv('finance/_data/stocks-screener-eval-stocks-etfs-01-07-2026.csv', skipfooter=1, engine='python', keep_default_na=False, na_values=['N/A'])
liquid_names = list(set(df_liquid_symbols.Symbol.tolist() + liquid_names))
with open('finance/_data/liquid_stocks.pkl', 'wb') as f: pickle.dump(liquid_names, f)

#%%

liquid_symbols = pickle.load(open('finance/_data/liquid_symbols.pkl', 'rb'))
liquid_stock_symbols = []
liquid_etf_symbols = []
for symbol in liquid_symbols:
  sym_info = utils.dolt_data.symbol_info(symbol)
  if sym_info is None:
    print(f'{symbol} has no info')
    continue
  if sym_info.is_etf:
    print(f'{symbol} is etf')
    liquid_etf_symbols.append(symbol)
  else:
    print(f'{symbol} is not liquid stock')
    liquid_stock_symbols.append(symbol)

# no_info = [sym for sym in liquid_symbols if sym not in liquid_etf_symbols+liquid_stock_symbols]
#%%
with open('finance/_data/liquid_stocks.pkl', 'wb') as f: pickle.dump(liquid_stock_symbols, f)
with open('finance/_data/liquid_etfs.pkl', 'wb') as f: pickle.dump(liquid_etf_symbols, f)
