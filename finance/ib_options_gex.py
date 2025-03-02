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
# ib_con.connect('127.0.0.1', api_paper_port, clientId=11, readonly=True)
ib_con.connect('127.0.0.1', tws_paper_port, clientId=10, readonly=True)
ib_con.reqMarketDataType(2)  # Use free, delayed, frozen data

# %%
symbol = 'QQQ'
contract = ib.Stock(symbol=symbol, exchange='SMART', currency='USD')
details = ib_con.reqContractDetails(contract)
print(details[0].longName)

# Ensure the contract details are validated via IB
ib_con.qualifyContracts(contract)

# %%
chains = ib_con.reqSecDefOptParams(underlyingSymbol=symbol, futFopExchange="", underlyingSecType="STK", underlyingConId=contract.conId)

chain = chains[0]

# %%
expiration = chain.expirations[0]
strike = 509
right = 'C'
# exchange = chain.exchange
option_contract = ib.Option(symbol=symbol, exchange="SMART", multiplier=chain.multiplier, strike=strike, lastTradeDateOrContractMonth=expiration, right=right, currency=contract.currency)

ib_con.qualifyContracts(option_contract)
ib_con.sleep(1)

#%%
ticker = None
events = []
tick_events = []
def store_ticker_events(ticker):
  global tick_events
  events.append(ticker)
  tick_events += ticker.ticks
  print(ticker)

ticker = ib_con.reqMktData(option_contract, "100, 101, 104, 105, 106, 225, 233, 375, 588", False, False)
# ticker = ib_con.reqMktData(option_contract, "", False, False)
ticker.updateEvent += store_ticker_events
# market_data = ib_con.reqMktData(option_contract, "", False, False)
# market_data = ib_con.reqMktData(option_contract, "", True, False)
ib_con.sleep(30)
ib_con.cancelMktData(option_contract)

#%%
snapshot = ib_con.reqMktData(option_contract, "", True, False)
# while snapshot.bid < 0:
ib_con.sleep(11)

data = {'strike': strike, 'kind': option_contract.right, 'close': snapshot.close, 'last': snapshot.last, 'bid': snapshot.bid, 'ask': snapshot.ask, 'volume': snapshot.volume}

