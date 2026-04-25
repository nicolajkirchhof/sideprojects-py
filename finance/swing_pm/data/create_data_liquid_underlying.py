#%%
import pandas as pd
import glob

from sqlalchemy import create_engine, text

# %load_ext autoreload
# %autoreload 2


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
min_vol_stocks = text("""
                    SELECT o.act_symbol
                    FROM ohlcv o
                             LEFT JOIN symbol s ON o.act_symbol = s.act_symbol
                    WHERE 
                        s.is_etf = 0 AND o.act_symbol not like '%$%' AND o.act_symbol not like '%.%'
                    GROUP BY o.act_symbol
                    HAVING MAX(o.volume) > 750000 AND MAX(o.close) > 1 AND MAX(o.close) < 5000
                    """)

df_min_vol_stocks = pd.read_sql(min_vol_stocks, db_connection)

min_vol_stocks = df_min_vol_stocks.act_symbol.tolist()

##%%
min_vol_etfs = text("""
                    SELECT o.act_symbol
                    FROM ohlcv o
                             LEFT JOIN symbol s ON o.act_symbol = s.act_symbol
                    WHERE s.is_etf = 1 AND o.act_symbol not like '%$%' AND o.act_symbol not like '%.%'
                    GROUP BY o.act_symbol
                    HAVING AVG(o.volume) > 2000000 AND MAX(o.close) > 1 AND MAX(o.close) < 5000
                    """)

df_min_vol_etfs = pd.read_sql(min_vol_etfs, db_connection)

min_vol_etfs = df_min_vol_etfs.act_symbol.tolist()
#%%
barchart_stocks = glob.glob('finance/_data/barchart/stocks-screener-eval-stocks*.csv')
dfs = []
for file in barchart_stocks:
  dfs.append(pd.read_csv(file, skipfooter=1, engine='python', keep_default_na=False, na_values=['N/A']))

df_barchart_stocks = pd.concat(dfs)

#%%
barchart_stocks_excluded = glob.glob('finance/_data/barchart/stocks-screener-excluded-eval-stocks*.csv')
dfs = []
for file in barchart_stocks_excluded:
  dfs.append(pd.read_csv(file, skipfooter=1, engine='python', keep_default_na=False, na_values=['N/A']))

df_barchart_stocks_excluded = pd.concat(dfs)
#%%
barchart_etfs = glob.glob('finance/_data/barchart/etf-screener-eval-etfs*.csv')
dfs = []
for file in barchart_etfs:
  dfs.append(pd.read_csv(file, skipfooter=1, engine='python', keep_default_na=False, na_values=['N/A']))

df_barchart_etfs = pd.concat(dfs)

#%%
barchart_stocks_final = [x for x in df_barchart_stocks.Symbol.tolist() if x not in df_barchart_stocks_excluded.Symbol.tolist()]
liquid_stocks = set(min_vol_stocks + barchart_stocks_final)
liquid_etfs = set(min_vol_etfs + df_barchart_etfs.Symbol.tolist())

duplicated = liquid_stocks & liquid_etfs
liquid_stocks -= duplicated
liquid_etfs -= duplicated
#%%
pd.DataFrame({'symbol': sorted(liquid_stocks)}).to_parquet('finance/_data/state/liquid_stocks.parquet', index=False)
pd.DataFrame({'symbol': sorted(liquid_etfs)}).to_parquet('finance/_data/state/liquid_etfs.parquet', index=False)
