from typing import Literal

import ib_async as ib
import numpy as np
from datetime import datetime, timedelta


# %%
def get_options_data(ib_con, contracts, tick_list="100, 101, 104, 105, 106, 165, 588", signalParameterLive="ask",
                     signalParameterFrozen="last", max_waittime=120):
  contracts = contracts if isinstance(contracts, list) else [contracts]
  ib_con.reqMarketDataType(2)  # Use free, delayed, frozen data
  snapshots = []
  for contract in contracts:
    snapshots.append(ib_con.reqMktData(contract, tick_list, False, False))
  _wait_for_data(ib_con, snapshots, signalParameterFrozen, max_waittime)

  ib_con.reqMarketDataType(1)  # Use free, delayed, frozen data
  _wait_for_data(ib_con, snapshots, signalParameterLive, max_waittime)

  for contract in contracts:
    ib_con.cancelMktData(contract)
  return snapshots


def _wait_for_data(ib_con, objects, indicator_name, max_wait_time):
  wait_time = 0
  while wait_time < max_wait_time and any(
      [ib.util.isNan(getattr(obj, indicator_name)) or getattr(obj, indicator_name) is None for obj in objects]):
    print(f"Waiting {wait_time} / {max_wait_time} for {indicator_name} to be available.")
    ib_con.sleep(1)
    wait_time += 1


def yearly_to_daily_iv(iv):
  if iv is None:
    return None
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


def get_and_qualify_contract_details(ib_con, contract):
  details = ib_con.reqContractDetails(contract)
  print(details[0].longName)
  ib_con.qualifyContracts(details[0].contract)
  return details[0]


def get_sigma_move(contract_ticker, sigma, num_days):
  sigma_move = contract_ticker.impliedVolatility * np.sqrt(num_days / 365) * contract_ticker.last
  max_value = np.ceil(contract_ticker.last + sigma * sigma_move)
  min_value = np.floor(contract_ticker.last - sigma * sigma_move)
  return sigma_move, max_value, min_value

def get_last_available(ticker, type: Literal["last", "bid", "ask"] = "last"):
  value = getattr(ticker, type)
  if np.isnan(value) or value is None or value < 0:
    value = getattr(ticker, "prev" + type.capitalize())
  return value
