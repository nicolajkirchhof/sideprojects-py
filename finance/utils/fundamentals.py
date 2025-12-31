import pandas as pd

MCAP_ORDER = ['Nano', 'Micro', 'Small', 'Mid', 'Large', 'Mega']
def market_cap_classifier(market_cap):
  """
  Classifies a stock based on its market capitalization using standard industry classes.
  Thresholds in USD.
  """
  if pd.isna(market_cap) or market_cap <= 0:
    return 'Unknown'
    
  if market_cap < 50_000_000:
    return 'Nano'
  elif market_cap < 300_000_000:
    return 'Micro'
  elif market_cap < 2_000_000_000:
    return 'Small'
  elif market_cap < 10_000_000_000:
    return 'Mid'
  elif market_cap < 200_000_000_000:
    return 'Large'
  else:
    return 'Mega'
