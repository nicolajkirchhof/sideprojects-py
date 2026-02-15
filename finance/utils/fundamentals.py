import numpy as np
import pandas as pd

df_market_cap_thresholds = pd.read_csv('finance/_data/MarketCapThresholds.csv')

def classify_market_cap(mcap_value, year):
  """Classifies market cap based on historical thresholds (from reprocess.py)."""
  if mcap_value is None or np.isnan(mcap_value):
    return "Unknown"

  # Get thresholds for the closest available year
  year_idx = df_market_cap_thresholds['Year'].sub(year).abs().idxmin()
  row = df_market_cap_thresholds.loc[year_idx]

  if mcap_value >= row['Large-Cap Entry (S&P 500)']:
    return "Large"
  elif mcap_value >= row['Mid-Cap Entry (S&P 400)']:
    return "Mid"
  elif mcap_value >= row['Small-Cap Entry (Russell 2000)']:
    return "Small"
  else:
    return "Micro"

