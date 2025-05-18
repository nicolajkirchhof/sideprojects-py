# %%
from datetime import datetime, timedelta

import dateutil
import numpy as np
import scipy
from datetime import date

import pandas as pd
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import mplfinance as mpf
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVR


import finance.utils as utils

pd.options.plotting.backend = "matplotlib"

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2

# %%
directory = f'N:/My Drive/Trading/Strategies/change_to_close'
os.makedirs(directory, exist_ok=True)
# symbol = 'IBDE40'
# symbols = [('IBDE40', pytz.timezone('Europe/Berlin')), ('IBGB100', pytz.timezone('Europe/London')),
#            *[(x, pytz.timezone('America/New_York')) for x in ['IBUS30', 'IBUS500', 'IBUST100']]]
# symbols = ['IBDE40', 'IBEU50', 'IBUS500']
symbols = ['DAX', 'ESTX50', 'SPX']

symbol = symbols[1]
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
results_df = pd.read_pickle(f'{directory}/{symbol}_change_to_close_results.pkl')
results_df.dropna(inplace=True)
X = results_df.filter(regex='^(09|10|11).*_pct_').iloc[:,:]
y = results_df['12:00_pct'].iloc[:]
y = y.abs() < 0.5
# bins = pd.cut(y.abs(), bins=3, labels=["Low", "Medium", "High"])

#%%
results_df[results_df.date > date(2025, 1, 1)].plot.scatter(x='date', y='12:00_pct')
results_df[results_df.date > date(2025, 1, 1)].plot.scatter(x='date', y='13:00_pct')
plt.show()
#%%

results_df[results_df.date > date(2025, 1, 1)]['12:00_pct'].hist(bins=30)
plt.figure()
results_df[(date(2024, 1, 1) < results_df.date) &  (results_df.date < date(2025, 1, 1))]['12:00_pct'].hist(bins=30)
plt.show()


results_df[(date(2024, 1, 1) < results_df.date) &  (results_df.date < date(2025, 1, 1))]['12:00_pct'].hist(bins=30)
#%%
pos = (results_df[(date(2023, 1, 1) < results_df.date) &  (results_df.date < date(2024, 1, 1))]['12:00_pct'].abs() < 0.3).sum()
neg = (results_df[(date(2023, 1, 1) < results_df.date) &  (results_df.date < date(2024, 1, 1))]['12:00_pct'].abs() > 0.3 ).sum()
print(pos, neg)

#%%
pos = (results_df[(date(2022, 1, 1) < results_df.date) &  (results_df.date < date(2023, 1, 1))]['12:00_pct'].abs() < 0.3).sum()
neg = (results_df[(date(2022, 1, 1) < results_df.date) &  (results_df.date < date(2023, 1, 1))]['12:00_pct'].abs() > 0.3 ).sum()
print(pos, neg)

#%%
# predict 12 to 17:30 change based on 9:30 - 11:30 Data

# Splitting the Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Print the sizes of the split
print("Training Set Size:", X_train.shape, y_train.shape)
print("Testing Set Size:", X_test.shape, y_test.shape)

# Create and train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Extract coefficients (model.coef_) and feature names
coefficients = model.coef_[0]
features = X_train.columns

# Pair features with their absolute coefficients
feature_importance = pd.DataFrame({
  'Feature': features,
  'Coefficient': coefficients,
  'Absolute Importance': np.abs(coefficients)
}).sort_values(by='Absolute Importance', ascending=False)

print(feature_importance)

#%%
model = SVR(kernel='rbf', C=1)
model.fit(X_train, y_train)

# Make Predictions on the Test Set
y_pred = model.predict(X_test)

# Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")

# Optional: Display Predicted vs Actual Values (Visualization)
import matplotlib.pyplot as plt

plt.scatter(y_pred, y_test, color="blue", label="Diff")
# plt.scatter(X_test, y_pred, color="red", label="Predicted Values")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Predicted vs Actual Values")
plt.legend()
plt.show()

#%%
# Compute permutation feature importance
perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)

# Display feature importance
print("Feature Importance:")
for i, importance in enumerate(perm_importance.importances_mean):
  print(f"Feature {i + 1}: {importance}")

# Optional visualization
import matplotlib.pyplot as plt

plt.bar(range(X.shape[1]), perm_importance.importances_mean)
plt.xticks(range(X.shape[1]), [f'Feature {i + 1}' for i in range(X.shape[1])])
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Permutation Feature Importance")
plt.show()



#%%
model = RandomForestRegressor(n_estimators=100)

model.fit(X_train, y_train)
# Make Predictions on the Test Set
y_pred = model.predict(X_test)

# Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")

# Optional: Display Predicted vs Actual Values (Visualization)
import matplotlib.pyplot as plt

plt.scatter(y_pred, y_test, color="blue", label="Diff")
# plt.scatter(X_test, y_pred, color="red", label="Predicted Values")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Predicted vs Actual Values")
plt.legend()
plt.show()


#%%
results_df_all = pd.concat([results_df, results_df_2024])
results_df_all.filter(regex='_pct').agg('mean')
results_df_all.filter(regex='_pct').agg('std')
#%%
ax = results_df_all.where(results_df_all.abs() < 0.6).filter(regex='_pct').plot(subplots=True, layout=(4, 2), title="changes")

# Show the plot
plt.tight_layout()
plt.show()
