import ib_async as ib
import numpy as np
from datetime import datetime, timedelta


#%%
def get_options_data(ib_con, contracts, tick_list = "100, 101, 104, 105, 106, 165, 588", signalParameter="ask"):
  contracts = contracts if isinstance(contracts, list) else [contracts]
  ib_con.reqMarketDataType(2)  # Use free, delayed, frozen data
  snapshots = []
  for contract in contracts:
    snapshots.append(ib_con.reqMktData(contract, tick_list, False, False))
  while any([snapshot.last < 0 for snapshot in snapshots]):
    print("Waiting for frozen data...")
    ib_con.sleep(1)
  ib_con.reqMarketDataType(1)  # Use free, delayed, frozen data
  while any([ib.util.isNan(getattr(snapshot, signalParameter)) for snapshot in snapshots]):
    print("Waiting for live data...")
    ib_con.sleep(1)
  for contract in contracts:
    ib_con.cancelMktData(contract)
  return snapshots

def yearly_to_daily_iv(iv):
  return iv / np.sqrt(252)

def third_friday(year, month):
  """Get the third Friday of a specific month in a given year."""
  # Start at the 1st day of the month
  first_day = datetime(year, month, 1)

  # Find the first Friday of the month
  first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)

  # Add 14 days to get the third Friday
  third_friday = first_friday + timedelta(weeks=2)
  return third_friday


# Get the third Fridays of March, June, September, December for a given year
def third_fridays_of_months(year):
  # Relevant months
  months = [3, 6, 9, 12]
  third_fridays = {month: third_friday(year, month) for month in months}
  return third_fridays

