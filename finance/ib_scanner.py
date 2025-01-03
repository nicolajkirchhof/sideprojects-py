#%%
# from __future__ import (absolute_import, division, print_function, unicode_literals)
import numpy as np
import scipy
import ib_async as ib
import backtrader as bt

from finance.avg_sig_change_strategy import AverageSignificantChangeStrategy
from finance.bull_flag_dip_strategy import BullFlagDipStrategy
from finance.ema_strategy import EmaStrategy
from finance.in_out_strategy import InOutStrategy
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

from finance.utils import percentage_change

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2

#%%
ib.util.startLoop()
ib_conn = ib.IB()
tws_real_port = 7497
tws_paper_port = 7497
api_real_port = 4002 
api_paper_port = 4002 
ib_conn.connect('127.0.0.1', api_paper_port, clientId=11, readonly=True)
ib_conn.reqMarketDataType(1)  # Use free, delayed, frozen data

#%%
xml = ib_conn.reqScannerParameters()

print(len(xml), 'bytes')

path = 'finance/tmp/scanner_parameters.xml'
with open(path, 'w') as f:
  f.write(xml)
#%%
# parse XML document
import xml.etree.ElementTree as ET
tree = ET.fromstring(xml)

# find all tags that are available for filtering
tags = [elem.text for elem in tree.findall('.//AbstractField/code')]
print(len(tags), 'tags:')
print(tags)
#%%
sub = ib.ScannerSubscription(
  instrument='STK',
  locationCode='STK.US.MAJOR',
  scanCode='TOP_TRADE_RATE',
  abovePrice=2,
  belowPrice=50,
  numberOfRows=50,
  marketCapBelow=1000
)

tagValues = [
  ib.TagValue("changePercAbove", "10"),
  ib.TagValue('priceAbove', 5),
  ib.TagValue('tradeRateAbove', 10),
  ib.TagValue('marketCapBelow1e6', 1000),
  ib.TagValue('priceBelow', 50)]

#%%
ib_conn.scannerDataEvent += lambda scanData: [print(sd.contractDetails.contract.symbol) for sd in scanData]
# the tagValues are given as 3rd argument; the 2nd argument must always be an empty list
# (IB has not documented the 2nd argument and it's not clear what it does)
# scanData = ib_conn.reqScannerData(sub, [], tagValues)
scanData = ib_conn.reqScannerSubscription(sub, [], tagValues)
ib_conn.sleep(5)
ib_conn.cancelScannerSubscription(scanData)

# contracts = [sd.contractDetails.contract for sd in scanData]
#
# for contract in contracts:
#   print(contract.symbol, contract.secType, contract.currency, contract.exchange)
#   stock = ib.Stock(contract.symbol, contract.exchange)
#   stock_data = ib_conn.reqMktData(stock, '1,2,4,9,55,56,46,89', False, False)
#   print(stock_data)

#%%

events = []
def store_ticker_events(ticker):
  events.append(ticker)
  print(ticker)


# ticker.updateEvent += lambda ticker: print(ticker)
# ib_conn.pendingTickersEvent += lambda ticker: print(ticker)
f = ib.Stock('NVDA', 'SMART', 'USD')
ticker = ib_conn.reqMktData(f, '236, 233, 293, 294, 295, 318, 411, 595', False, False)
# ticker = ib_conn.reqMktData(f, '236, 233, 293, 294, 295, 318, 411, 595', True, False)
ticker.updateEvent += store_ticker_events
ib_conn.sleep(10)
ib_conn.cancelMktData(f)
#%%
def wait_for_market_data(tickers):
  """print tickers as they arrive"""
  print(tickers)

ib_conn.pendingTickersEvent += wait_for_market_data

#%%
ib_conn.disconnect()
