# %%
import ib_async as ib
import pandas as pd

import matplotlib as mpl
from datetime import datetime
import dateutil
from finance import utils

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2

utils.influx.create_databases()
influx_client_df, influx_client = utils.influx.get_influx_clients()

tws_instance = 'real'
ib_con = utils.ibkr.connect(tws_instance, 3, 2)

#%%

no_ooi_indices = ['V2TX', 'V1X', 'VXN', 'RVX', 'VXSLV', 'GVZ']
no_hv_indices = ['VXSLV']
eu_indices = [ib.Index(x, 'EUREX', 'EUR') for x in ['DAX', 'ESTX50', 'V2TX', 'V1X']]
us_indices = [*[ib.Index(x, 'CBOE', 'USD') for x in ['VIX', 'VXN', 'RVX', 'VXSLV', 'GVZ', 'OVX']],
              ib.Index('SPX', 'CBOE', 'USD'),
              ib.Index('NDX', 'NASDAQ', 'USD'), ib.Index('RUT', 'RUSSELL', 'USD'),
              ib.Index('INDU', 'CME', 'USD')]
fr_index = ib.Index('CAC40', 'MONEP', 'EUR')
# es_index = ib.Index('IBEX35', 'MEFFRV', 'EUR')
jp_index = ib.Index('N225', 'OSE.JPN', 'JPY')
hk_index = ib.Index('HSI', 'HKFE', 'HKD')
indices = [*eu_indices, *us_indices, jp_index, fr_index, hk_index]

index_cfd_euro = ['IBGB100', 'IBEU50', 'IBDE40', 'IBFR40', 'IBES35', 'IBNL25', 'IBCH20']
index_cfd_us = ['IBUS500', 'IBUS30', 'IBUST100']
index_cfd_asia = ['IBHK50', 'IBJP225', 'IBAU200']

index_cfds = [ib.CFD(symbol=symbol, exchange='SMART') for symbol in [*index_cfd_euro, *index_cfd_us, *index_cfd_asia]]

commodity_cfds = [ib.Commodity("XAUUSD", exchange='SMART'), ib.Commodity("USGOLD", exchange='IBMETAL')]

forex = [ib.Forex(symbol=sym, exchange='IDEALPRO', currency=cur) for sym, cur in
         [('EUR', 'USD'), ('EUR', 'GBP'), ('EUR', 'CHF'), ('GBP', 'USD'), ('AUD', 'USD'), ('USD', 'CAD'),
          ('USD', 'JPY'), ('CHF', 'USD')]]

eu_futures = [ib.ContFuture(symbol=x, multiplier='1', exchange='EUREX',currency='EUR') for x in ['DAX', 'ESTX50']]
us_futures =[*[ib.ContFuture(symbol=x[0], multiplier=x[1], exchange=x[2], currency='USD') for x in
               [('MES', '5', 'CME'), ('MNQ', '2', 'CME'), ('RTY', '50', 'CME'), ('MYM', '0.5', 'CBOT'),
                ('VXM', '100', 'CFE'), ('ZB', '1000', 'CBOT'), ('ZC', '5000', 'CBOT'), ('ZF', '1000', 'CBOT'),
                ('ZN', '1000', 'CBOT'), ('ZT', '2000', 'CBOT'), ('ZW', '5000', 'CBOT'),
                ('SI', '5000', 'COMEX'), ('GC', '100', 'COMEX'), ('CL', '1000', 'NYMEX'), ('NG', '10000', 'NYMEX')]]]
jp_futures = [ib.ContFuture(symbol='N225M', multiplier='100', exchange='OSE.JPN',currency='JPY')]
swe_futures = [ib.ContFuture(symbol='OMXS30', multiplier='100', exchange='OMS',currency='SEK')]
#
# # contracts = indices
futures = [*eu_futures, *us_futures, *jp_futures, *swe_futures]

## %%
us_etf_symbols = [
  'EEM', 'EWZ', 'FXI', 'GDX', 'GLD', 'HYG', 'IEFA', 'IWM', 'LQD', 'QQQ', 'SLV', 'SMH', 'SPY', 'TLT', 'TQQQ',
  'UNG', 'USO', 'XLB', 'XLC', 'XLE',  'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV',
  'XLY', 'XOP']
us_etfs = [ib.Stock(symbol=x, exchange='SMART', currency='USD') for x in us_etf_symbols]

contracts = [*indices, *us_etfs, *commodity_cfds, *index_cfds, *forex, *futures,]
for contract in contracts:
  ib_con.qualifyContracts(contract)
  details = ib_con.reqContractDetails(contract)
  print(details[0].longName)
# # # contract_ticker = ib_con.reqMktData(contracts[0], '236, 233, 293, 294, 295, 318, 411, 595', True, False)

## %%
available_types_of_data = ['TRADES', 'MIDPOINT', 'BID', 'ASK', 'BID_ASK', 'HISTORICAL_VOLATILITY',
                           'OPTION_IMPLIED_VOLATILITY']
types_of_data = {'IND': ['TRADES', 'HISTORICAL_VOLATILITY','OPTION_IMPLIED_VOLATILITY'], 'CFD': ['MIDPOINT'], 'FUT': ['TRADES'],
                 'STK': ['TRADES', 'HISTORICAL_VOLATILITY','OPTION_IMPLIED_VOLATILITY'], 'CONTFUT': ['TRADES'], 'CASH': ['MIDPOINT'], 'CMDTY': ['MIDPOINT']}
rth = False
##%%
duration = '10 D'
offset_days = 9
barSize = '1 min'

startDateTime = dateutil.parser.parse('2013-06-01')
# startDateTime = dateutil.parser.parse('2025-03-10')
# endDateTime=dateutil.parser.parse('2013-06-08')
endDateTime = datetime.now()


def contract_to_fieldname(contract):
  if contract.secType == 'IND' or contract.secType == 'CFD' or contract.secType == 'STK' or contract.secType == 'CMDTY':
    return contract.symbol
  if 'FUT' in contract.secType:
    return f'F{contract.symbol}'
  if contract.secType == 'CASH':
    return contract.symbol + contract.currency

field_name_lu = {
  'TRADES': {'open': 'o', 'high': 'h', 'low': 'l', 'close': 'c', 'volume': 'v', 'average': 'a', 'barCount': 'bc'},
  'MIDPOINT': {'open': 'o', 'high': 'h', 'low': 'l', 'close': 'c', 'volume': 'v', 'average': 'a', 'barCount': 'bc'},
  'HISTORICAL_VOLATILITY': {'open': 'hvo', 'high': 'hvh', 'low': 'hvl', 'close': 'hvc', 'volume': 'hvv', 'average': 'hva', 'barCount': 'hvbc'},
  'OPTION_IMPLIED_VOLATILITY': {'open': 'ivo', 'high': 'ivh', 'low': 'ivl', 'close': 'ivc', 'volume': 'ivv', 'average': 'iva', 'barCount': 'ivbc'},
}
#%%
# for stock_name in stock_names:
current_date = startDateTime
for contract in contracts:
  dfs_contract = {}
  for typ in types_of_data[contract.secType]:
    if typ == 'OPTION_IMPLIED_VOLATILITY' and contract.symbol in no_ooi_indices \
        or typ == 'HISTORICAL_VOLATILITY' and contract.symbol in no_hv_indices:
      continue
    field_name = field_name_lu[typ]['close']
    c_last = influx_client_df.query(f'select last("{field_name}") from {contract_to_fieldname(contract)}',
                                   database=utils.influx.sec_type_to_database_name(contract.secType))
    if c_last:
      current_date = pd.Timestamp(c_last[contract_to_fieldname(contract)].index.values[0]).to_pydatetime()
    else:
      current_date = startDateTime
    while current_date < endDateTime:
      ##%%
      current_date = current_date + pd.Timedelta(days=offset_days)
      endDateTimeString = current_date.strftime('%Y%m%d %H:%M:%S') if not contract.secType == 'CONTFUT' else ""
      data = ib_con.reqHistoricalData(contract, endDateTime=endDateTimeString, durationStr=duration,
                                      barSizeSetting=barSize, whatToShow=typ, useRTH=rth)
      ##%%
      print(f'{datetime.now()} : {contract_to_fieldname(contract)} => {typ} {current_date} #{len(data)}')
      if not data:
        continue
      dfs_type = ib.util.df(data).rename(columns=field_name_lu[typ]).set_index('date').tz_localize('UTC')
      influx_client_df.write_points(dfs_type, contract_to_fieldname(contract),
                                   database=utils.influx.sec_type_to_database_name(contract.secType))

