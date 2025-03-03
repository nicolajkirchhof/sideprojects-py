# %%
# from __future__ import (absolute_import, division, print_function, unicode_literals)
import datetime
import asyncio
import dateutil
import numpy as np
import scipy
# from ib_async import ib
import ib_async as ib
# util.startLoop()  # uncomment this line when in a notebook
import backtrader as bt

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import datetime
import finance.ibkr as ibkr
import dateutil

import influxdb as idb

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2

# %%
ib.util.startLoop()
ib_con = ib.IB()
tws_real_port = 7497
tws_paper_port = 7497
api_real_port = 4002
api_paper_port = 4002
# ib_con.connect('127.0.0.1', api_paper_port, clientId=12, readonly=True)
ib_con.connect('127.0.0.1', tws_paper_port, clientId=10, readonly=True)
ib_con.reqMarketDataType(2)  # Use free, delayed, frozen data

# %%
symbol = 'QQQ'
contract = ib.Stock(symbol=symbol, exchange='SMART', currency='USD')
details = ib_con.reqContractDetails(contract)
print(details[0].longName)

# Ensure the contract details are validated via IB
ib_con.qualifyContracts(contract)
#%%
def get_frozen_and_live_data(contracts, tick_list = "100, 101, 104, 105, 106, 165, 588", signalParameter="ask"):
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


#%%
contract_ticker = get_frozen_and_live_data(contract, signalParameter="impliedVolatility")

spot = contract_ticker.last
hv = contract_ticker.histVolatility
iv = contract_ticker.impliedVolatility
# %%
chains = ib_con.reqSecDefOptParams(underlyingSymbol=symbol, futFopExchange="", underlyingSecType="STK", underlyingConId=contract.conId)
chain = chains[0]

#%%
# get data for next 7 days
endDate = datetime.datetime.now().date() + datetime.timedelta(days=14)
valid_expirations = [e for e in chain.expirations if dateutil.parser.parse(e).date() <= endDate]

last_expiration = dateutil.parser.parse(valid_expirations[-1]).date()
num_days = (last_expiration - datetime.datetime.now().date()).days
sigma_move = iv * np.sqrt(num_days/365) * contract_ticker.last
max_value =  np.ceil(contract_ticker.last + 2 * sigma_move)
min_value =  np.floor(contract_ticker.last - 2* sigma_move)
relevant_strikes= [s for s in chain.strikes if min_value <= s <= max_value]
#%%
expiration = valid_expirations[0]
option_contract = ib.Option(symbol=symbol, exchange="SMART", multiplier=chain.multiplier, strike=0.0, lastTradeDateOrContractMonth=expiration, right="", currency=contract.currency)
# ib_con.qualifyContracts(option_contract)
# ib_con.sleep(1)
# chain_data = ib_con.reqMktData(option_contract, "", True, False)
option_contract_details = ib_con.reqContractDetails(option_contract)
ib_con.sleep(1)

#%%
num_days = (dateutil.parser.parse(expiration).date() - datetime.datetime.now().date()).days + 0.5
sigma_move = iv * np.sqrt(num_days/365) * contract_ticker.last
max_value =  np.ceil(contract_ticker.last + 2 * sigma_move)
min_value =  np.floor(contract_ticker.last - 2* sigma_move)

#%%
option_contracts = [od.contract for od in option_contract_details if min_value <= od.contract.strike <= max_value]

contracts = option_contracts
#%%
ib_con.reqMarketDataType(2)  # Use free, delayed, frozen data
snapshots = []
tick_list = "100, 101, 104, 105, 106, 165, 588"
for contract in option_contracts:
  snapshots.append(ib_con.reqMktData(contract, tick_list, False, False))
while any([snapshot.modelGreeks is None for snapshot in snapshots]):
  print("Waiting for frozen data...")
  ib_con.sleep(1)

ib_con.reqMarketDataType(1)  # Use free, delayed, frozen data
while any([ib.util.isNan(snapshot.callOpenInterest) for snapshot in snapshots]):
  print("Waiting for live data...")
  ib_con.sleep(1)
for contract in contracts:
  ib_con.cancelMktData(contract)

option_chain_data = snapshots


#%% Make gex dataframe
options_data_dict = [{'expiry': od.contract.lastTradeDateOrContractMonth,
                    'right': od.contract.right,
                    'multiplier': float(od.contract.multiplier),
                    'strike': od.contract.strike,
                      'gamma': od.modelGreeks.gamma,
                      'delta': od.modelGreeks.delta,
                      'vega': od.modelGreeks.vega,
                      'theta': od.modelGreeks.theta,
                      'impliedVolatility': od.modelGreeks.impliedVol,
                      'openInterest': od.putOpenInterest if od.contract.right == 'P' else od.callOpenInterest,
                      'volume': od.volume,
                    'bid': od.bid if od.bid >= 0 else od.prevBid,
                      'ask':od.ask if od.ask >=0 else od.prevAsk,
                      'last':od.last,
                      } for od in option_chain_data]

df_options = pd.DataFrame(options_data_dict)
df_options['expiry'] = pd.to_datetime(df_options['expiry'])
df_options['gex'] = df_options.apply(lambda x: spot * spot * 0.01 * x.openInterest * x.multiplier * x.volume, axis=1)
df_options.set_index(['expiry', 'strike'], inplace=True)
df_options.sort_index(inplace=True)

#%%



