# %%
import ib_async as ib
import pandas as pd

import matplotlib as mpl
from datetime import datetime, timedelta
import dateutil
from finance import utils
from finance.utils.influx import DB_DAILY

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2

utils.influx.create_databases()
influx_client_df, influx_client = utils.influx.get_influx_clients()

tws_instance = 'real'
ib_con = utils.ibkr.connect(tws_instance, 4, 2)

# %%

no_ooi_indices = utils.underlyings.cboe_volatility_indices + utils.underlyings.eu_volatility_indices
no_hv_indices = no_ooi_indices

eu_indices = [ib.Index(x, 'EUREX', 'EUR') for x in utils.underlyings.eu_indices]
us_indices = [*[ib.Index(x, 'CBOE', 'USD') for x in ['VIX'] + utils.underlyings.cboe_volatility_indices],
              ib.Index('SPX', 'CBOE', 'USD'),
              ib.Index('NDX', 'NASDAQ', 'USD'), ib.Index('RUT', 'RUSSELL', 'USD'),
              ib.Index('INDU', 'CME', 'USD')]
indices = [*eu_indices, *us_indices]

forex = [ib.Forex(symbol=sym, exchange='IDEALPRO', currency=cur) for sym, cur in
         [('EUR', 'USD'), ('EUR', 'GBP'), ('EUR', 'CHF'), ('GBP', 'USD'), ('AUD', 'USD'), ('USD', 'CAD'),
          ('USD', 'JPY'), ('CHF', 'USD')]]

eu_futures = [ib.ContFuture(symbol=x, multiplier='1', exchange='EUREX', currency='EUR') for x in ['DAX', 'ESTX50']]
us_futures = [*[ib.ContFuture(symbol=x[0], multiplier=x[1], exchange=x[2], currency='USD') for x in
                [('ES', '50', 'CME'), ('NQ', '20', 'CME'), ('RTY', '50', 'CME'),
                 ('VXM', '100', 'CFE'), ('ZB', '1000', 'CBOT'), ('ZC', '5000', 'CBOT'), ('ZF', '1000', 'CBOT'),
                 ('ZN', '1000', 'CBOT'), ('ZT', '2000', 'CBOT'), ('ZW', '5000', 'CBOT'),
                 ('SI', '5000', 'COMEX'), ('GC', '100', 'COMEX'), ('CL', '1000', 'NYMEX'), ('NG', '10000', 'NYMEX')]]]
#
futures = [*eu_futures, *us_futures]

## %%
us_etf_symbols = [*utils.underlyings.market_etf_symbols, *utils.underlyings.sectors_etf_symbols,
                  *utils.underlyings.world_etf_symbols, *utils.underlyings.crypto_etf_symbols,
                  *utils.underlyings.forex_etf_symbols, *utils.underlyings.metals_etf_symbols,
                  *utils.underlyings.energy_etf_symbols, *utils.underlyings.agriculture_etf_symbols]

us_etfs = [ib.Stock(symbol=x, exchange='SMART', currency='USD') for x in us_etf_symbols]

us_stocks = [ib.Stock(symbol=x, exchange='SMART', currency='USD') for x in utils.underlyings.us_stock_symbols]

contracts = [*indices, *us_etfs, *forex, *futures, *us_stocks]

for contract in contracts:
  ib_con.qualifyContracts(contract)
  details = ib_con.reqContractDetails(contract)
  print(details[0].longName)

## %%
available_types_of_data = ['TRADES', 'MIDPOINT', 'BID', 'ASK', 'BID_ASK', 'HISTORICAL_VOLATILITY',
                           'OPTION_IMPLIED_VOLATILITY']
types_of_data = {'IND': ['TRADES', 'HISTORICAL_VOLATILITY', 'OPTION_IMPLIED_VOLATILITY'], 'CFD': ['MIDPOINT'],
                 'FUT': ['TRADES'],
                 'STK': ['TRADES', 'HISTORICAL_VOLATILITY', 'OPTION_IMPLIED_VOLATILITY'], 'CONTFUT': ['TRADES'],
                 'CASH': ['MIDPOINT'], 'CMDTY': ['MIDPOINT']}
rth = True
##%%
duration = '365 D'
offset_days = 360
barSize = '1 hour'

startDateTime = dateutil.parser.parse('1993-01-01')
endDateTime = datetime.now()


def contract_to_fieldname(contract):
  if contract.secType == 'IND' or contract.secType == 'STK' or contract.secType == 'CMDTY':
    return contract.symbol
  if 'FUT' in contract.secType:
    return f'F{contract.symbol}'
  if contract.secType == 'CASH':
    return contract.symbol + contract.currency


field_name_lu = {
  'TRADES': {'open': 'o', 'high': 'h', 'low': 'l', 'close': 'c', 'volume': 'v', 'average': 'a', 'barCount': 'bc'},
  'MIDPOINT': {'open': 'o', 'high': 'h', 'low': 'l', 'close': 'c', 'volume': 'v', 'average': 'a', 'barCount': 'bc'},
  'HISTORICAL_VOLATILITY': {'open': 'hvo', 'high': 'hvh', 'low': 'hvl', 'close': 'hvc', 'volume': 'hvv',
                            'average': 'hva', 'barCount': 'hvbc'},
  'OPTION_IMPLIED_VOLATILITY': {'open': 'ivo', 'high': 'ivh', 'low': 'ivl', 'close': 'ivc', 'volume': 'ivv',
                                'average': 'iva', 'barCount': 'ivbc'},
}
# %%
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
                                    database=DB_DAILY)
    if c_last:
      current_date = pd.Timestamp(c_last[contract_to_fieldname(contract)].index.values[0]).to_pydatetime()
      if current_date.date() > (datetime.now() - timedelta(days=3)).date():
        continue
    else:
      # current_date = startDateTime
      endDateTimeString = datetime.now().strftime('%Y%m%d %H:%M:%S') if not contract.secType == 'CONTFUT' else ""
      data = ib_con.reqHistoricalData(contract, endDateTime=endDateTimeString, durationStr='20 Y',
                                    barSizeSetting='1 month', whatToShow=typ, useRTH=rth)
      current_date = datetime.combine(data[0].date, datetime.min.time())
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
                                    database=DB_DAILY)
