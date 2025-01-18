#%%
# from __future__ import (absolute_import, division, print_function, unicode_literals)
import numpy as np
import scipy
import ib_async as ib
import backtrader as bt
from tests.test_indicators import volume_ts
from vectorbt.portfolio import trade_dt

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
low_price_gainers = ib.ScannerSubscription(
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
  ib.TagValue('priceAbove', 2),
  ib.TagValue('tradeRateAbove', 10),
  ib.TagValue('marketCapBelow1e6', 1000),
  ib.TagValue('priceBelow', 50)]

# low_price_gainers = ib.ScannerSubscription(
#   instrument='STK',
#   locationCode='STK.US.MAJOR',
#   scanCode='TOP_TRADE_RATE',
#   abovePrice=2,
#   belowPrice=50,
#   numberOfRows=50,
#   marketCapBelow=1000
# )
#
# tagValues = [
#   ib.TagValue("changePercAbove", "10"),
#   ib.TagValue('priceAbove', 2),
#   ib.TagValue('tradeRateAbove', 10),
#   ib.TagValue('marketCapBelow1e6', 1000),
#   ib.TagValue('priceBelow', 50)]

#%%
def stock_ticker_events(ticker):
  spread = np.abs(ticker.ask - ticker.bid)
  shortable = 'S ' if ticker.shortableShares > 100 else ''
  trade_rate = f'{ticker.tradeRate}' if ticker.tradeRate < 1000 else f'{ticker.tradeRate/1000:.1f}K'
  volume_rate = f'{ticker.volumeRate}' if ticker.volumeRate < 1000 else f'{ticker.volumeRate/1000:.0f}K'
  change = percentage_change(ticker.vwap, ticker.close)
  print(f'{shortable} {ticker.contract.symbol} VWAP {ticker.vwap:.2f} CHG {change:.2f}% SPR {spread:.1f} T/M {trade_rate} V/M {volume_rate}')

#%%
# ib_conn.scannerDataEvent += lambda scanData: [print(sd.contractDetails.contract.symbol) for sd in scanData]
# the tagValues are given as 3rd argument; the 2nd argument must always be an empty list
# (IB has not documented the 2nd argument and it's not clear what it does)
scanData = ib_conn.reqScannerData(sub, [], tagValues)
# scanData = ib_conn.reqScannerSubscription(sub, [], tagValues)
# for sd in scanData:
#   print(sd.contractDetails.contract.symbol)
# ib_conn.sleep(20)
# ib_conn.cancelScannerSubscription(scanData)

# contracts = [sd.contractDetails.contract for sd in scanData]
#
#%%
for e in events:
  stock_ticker_events(e)
#%%
for sd in scanData:
  contract = sd.contractDetails.contract
  print(contract.symbol, contract.secType, contract.currency, contract.exchange)
  stock = ib.Stock(contract.symbol, contract.exchange)
  stock_ticker = ib_conn.reqMktData(stock, '236, 233, 293, 294, 295, 318, 411, 595', True, False)
  print(stock_ticker)

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
