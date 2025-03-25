# %%
from datetime import datetime, timedelta

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

pd.options.plotting.backend = "matplotlib"

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2

# %%
directory = f'N:/My Drive/Projects/Trading/Research/Strategies/change_to_close'
os.makedirs(directory, exist_ok=True)
# symbol = 'IBDE40'
# symbols = [('IBDE40', pytz.timezone('Europe/Berlin')), ('IBGB100', pytz.timezone('Europe/London')),
#            *[(x, pytz.timezone('America/New_York')) for x in ['IBUS30', 'IBUS500', 'IBUST100']]]
# symbols = ['IBDE40', 'IBEU50', 'IBUS500']
symbols = ['DAX', 'ESTX50', 'SPX']

symbol = symbols[0]
symbol_def = utils.influx.SYMBOLS[symbol]
tz = symbol_def['EX']['TZ']

#%%
first_day = tz.localize(dateutil.parser.parse('2022-01-01T00:00:00'))
last_day = tz.localize(dateutil.parser.parse('2025-03-19T00:00:00'))
df = utils.influx.get_candles_range_aggregate(first_day, last_day, symbol, '30m')

# Group data by the date part of the index
df_grp = df.groupby(df.index.date)

#%%
results = []
# date, group = next(iter(df_grp))
for date, group in df_grp:
  # Extract value for 17:30
  value_at_1730 = group.loc[group.index.time == pd.to_datetime('17:30').time(), 'c']

  if value_at_1730.empty:
    print(f'No data for {date}')
    continue

  value_at_1730 = value_at_1730.iat[0]
  result = {'date': date, 'value_at_1730': value_at_1730}
  for half_hour in ['09:30', '10:00', '10:30', '11:00', '11:30', '12:00', '12:30', '13:00']:
    # Extract mean of 09:30 to 13:00
    row = group.loc[group.index.time == pd.to_datetime(half_hour).time(), :]

    if row.empty:
      print(f'No data for {date} {half_hour}')
      result[f'{half_hour}_pct'] = np.nan
      result[f'{half_hour}'] = np.nan
      continue
    value_at_time = row.c.iat[0]
    pct_o_c = (row.o - row.c)/row.o * 100
    pct_l_h = np.sign(pct_o_c) * (row.h - row.l)/row.l * 100

    # Calculate percentage difference
    percentage_difference = ((value_at_1730 - value_at_time) / value_at_time) * 100
    result[f'{half_hour}_pct'] = percentage_difference
    result[f'{half_hour}'] = value_at_time
    result[f'{half_hour}_pct_o_c'] = pct_o_c.iat[0]
    result[f'{half_hour}_pct_l_h'] = pct_l_h.iat[0]
  results.append(result)
  # print(f'Done {date}')
#%%
results_df = pd.DataFrame(results)
results_df.set_index('date', inplace=True)

results_df_2024 = results_df
results_df.to_pickle(f'{directory}/{symbol}_change_to_close_results.pkl')
#%%
X = results_df.filter(regex='^(09|10|11).*_pct_')

#%%
# predict 12 to 17:30 change based on 9:30 - 11:30 Data

# Splitting the Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.svm import SVR

model = SVR(kernel='rbf', C=1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)
predictions = model.predict(X_test)


#%%
results_df_all = pd.concat([results_df, results_df_2024])
results_df_all.filter(regex='_pct').agg('mean')
results_df_all.filter(regex='_pct').agg('std')
#%%
ax = results_df_all.where(results_df_all.abs() < 0.6).filter(regex='_pct').plot(subplots=True, layout=(4, 2), title="changes")

# Show the plot
plt.tight_layout()
plt.show()
