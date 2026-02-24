#%%
import pickle
import pandas as pd
import glob

from sqlalchemy import create_engine, text

from finance import utils
from finance.swing_pm.create_data_momentum_earnings_analysis import liquid_stocks

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
min_vol_stocks = text("""SELECT DISTINCT o.act_symbol
                          FROM ohlcv o
                                   left join symbol s on o.act_symbol = s.act_symbol
                          WHERE volume > 750000
                            and s.is_etf = 0""")

df_min_vol_stocks = pd.read_sql(min_vol_stocks, db_connection)

min_vol_stocks = df_min_vol_stocks.act_symbol.tolist()

#%%
min_vol_etfs = text("""
                    SELECT o.act_symbol
                    FROM ohlcv o
                             LEFT JOIN symbol s ON o.act_symbol = s.act_symbol
                    WHERE s.is_etf = 1
                    GROUP BY o.act_symbol
                    HAVING AVG(o.volume) > 2000000
                    """)

df_min_vol_etfs = pd.read_sql(min_vol_etfs, db_connection)

min_vol_etfs = df_min_vol_etfs.act_symbol.tolist()
#%%
barchart_stocks = glob.glob('finance/_data/stocks-screener-eval-stocks*.csv')
dfs = []
for file in barchart_stocks:
  dfs.append(pd.read_csv(file, skipfooter=1, engine='python', keep_default_na=False, na_values=['N/A']))

df_barchart_stocks = pd.concat(dfs)

#%%
barchart_etfs = glob.glob('finance/_data/etf-screener-eval-etfs*.csv')
dfs = []
for file in barchart_etfs:
  dfs.append(pd.read_csv(file, skipfooter=1, engine='python', keep_default_na=False, na_values=['N/A']))

df_barchart_etfs = pd.concat(dfs)

#%%
liquid_stocks = set(min_vol_stocks + df_barchart_stocks.Symbol.tolist())
liquid_etfs = set(min_vol_etfs + df_barchart_etfs.Symbol.tolist())

duplicated = liquid_stocks & liquid_etfs
liquid_stocks -= duplicated
liquid_etfs -= duplicated
#%%
with open('finance/_data/liquid_stocks.pkl', 'wb') as f: pickle.dump(liquid_stocks, f)
with open('finance/_data/liquid_etfs.pkl', 'wb') as f: pickle.dump(liquid_etfs, f)
