#%%
import os

import pandas as pd
import matplotlib as mpl
from sqlalchemy import create_engine, text

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2

#%%

df_ecsv = pd.read_csv('finance/_data/hist_earnings/earnings_latest.csv', sep=',', parse_dates=['date'] )
df_ecsv = df_ecsv.rename(columns={'symbol':'symbol_csv', 'eps_est':'eps_est_csv', 'eps':'eps_csv', 'qtr':'qtr_csv', 'date':'date_csv', 'release_time':'when_csv'})


#%%
db_connection_str = 'mysql+pymysql://root:@localhost:3306/earnings'
db_connection = create_engine(db_connection_str)

liquid_symbols = pd.read_pickle('finance/_data/liquid_stocks.pkl')

for ticker in liquid_symbols:
  if os.path.exists(f'finance/_data/earnings_cleaned/{ticker}.csv'): continue
  #%%
  query = """select ec.act_symbol, ec.date, ec.when, eh.period_end_date, eh.reported, eh.estimate from earnings_calendar ec
   left join eps_history eh on ec.act_symbol = eh.act_symbol
   and eh.period_end_date = ( select max(period_end_date) from eps_history eh_sub
   where eh_sub.act_symbol = ec.act_symbol and eh_sub.period_end_date < ec.date )
   where ec.act_symbol = :ticker"""
  stmt = text(query)
  # Example usage with pandas:
  df_edb = pd.read_sql(stmt, db_connection, params={'ticker': ticker})

  df_edb = df_edb[['act_symbol', 'date', 'when', 'period_end_date', 'reported', 'estimate']].rename(columns={'act_symbol':'symbol', 'reported':'eps', 'estimate':'eps_est'})
  df_edb['date'] = pd.to_datetime(df_edb.date)
  df_edb['when'] = df_edb.when.str.replace('After market close', 'post', regex=False)
  df_edb['when'] = df_edb.when.str.replace('Before market open', 'pre', regex=False)

  #%%
  df_escv_ticker = df_ecsv[df_ecsv.symbol_csv == ticker]

  df_join = df_edb.merge(df_escv_ticker,left_on='date', right_on='date_csv', how='outer')

  ##%% merge columns
  for key in ['symbol', 'date', 'when', 'eps_est', 'eps']:
    df_join[key] = df_join[key].combine_first(df_join[f'{key}_csv'])

  if df_join.empty:
    continue

  print(df_join.to_string())
  df_join.to_csv(f'finance/_data/earnings_cleaned/{ticker}.csv', index=False)
#%%

