# %%
from datetime import datetime

import ib_async as ib
import pandas as pd

import finance.utils as utils

%load_ext autoreload
%autoreload 2

##%%
tws_instance = 'real'
ib_con = utils.ibkr.connect(tws_instance, 7, 2)

# %%
underlying_market_data = {}
MAX_TRIES = 10

df_options = pd.read_csv('finance/ibkr/predefined_options.csv', parse_dates=['expiry'])

##%%
contracts = [ib.Contract(conId=o['contract_id']) for o in df_options.to_dict('records')]

ib_con.qualifyContracts(*contracts)

for contract in contracts:
  print(f'{contract.symbol} {contract.secType} {contract.lastTradeDateOrContractMonth} {contract.strike} {contract.right}')

## %%
SEP = ','
plain = f'Symbol {SEP} IV {SEP} Right {SEP} Underlying {SEP} Date {SEP} Exp {SEP} Strike {SEP} Price {SEP} TimeValue {SEP} Δ {SEP} Θ {SEP} Γ {SEP} ν {SEP} Action\n'
for contract in contracts:
  #%%
  ib_con.reqMarketDataType(2)
  market_data = ib_con.reqMktData(contract, "", True, False)
  tries = 0
  while (market_data.modelGreeks is None or ((utils.math.isnan(utils.ibkr.get_options_price(market_data, 'bid')) or
                                              utils.math.isnan(utils.ibkr.get_options_price(market_data, 'ask')))  and
                                             utils.math.isnan(utils.ibkr.get_options_price(market_data, 'last')))) and tries < MAX_TRIES:
    print(f"Waiting {tries} / {MAX_TRIES}  for option frozen data...")
    tries += 1
    ib_con.sleep(5)
  contract_details = ib_con.reqContractDetails(contract)[0]

  if contract_details.underConId in underlying_market_data:
    umd = underlying_market_data[contract_details.underConId]
  else:
    underlying = ib.Contract(symbol=contract_details.underSymbol, secType=contract_details.underSecType,
                             conId=contract_details.underConId)
    ib_con.qualifyContracts(underlying)
    ib_con.sleep(1)

    umd = ib_con.reqHistoricalData(underlying, "", durationStr='1 D', barSizeSetting='1 day',
                                   whatToShow='TRADES', useRTH=True)[0]
    underlying_market_data[underlying.conId] = umd
  greeks =  ib.OptionComputation(-1, None, None, None, None, None, None, None, None)
  if market_data.lastGreeks is not None and market_data.lastGreeks.delta is not None \
      and market_data.lastGreeks.gamma is not None and market_data.lastGreeks.vega is not None \
      and market_data.lastGreeks.theta is not None and market_data.lastGreeks.impliedVol is not None:
    greeks = market_data.lastGreeks
  elif market_data.modelGreeks is not None:
    greeks = market_data.modelGreeks

  bid = utils.ibkr.get_options_price(market_data, 'bid')
  ask = utils.ibkr.get_options_price(market_data, 'ask')
  last = utils.ibkr.get_options_price(market_data, 'last')

  price = last
  if utils.math.isnan(price) and not utils.math.isnan(bid) and not utils.math.isnan(ask):
    price = (bid + ask)/2
  iv = greeks.impliedVol if greeks.impliedVol is not None else -1
  daily_iv = utils.ibkr.yearly_to_daily_iv(iv)
  ##%% Greeks sometimes return None
  greeks_to_str = lambda x: f'{1000*x:.0f}' if x is not None else 'NaN'
  exp = contract.lastTradeDateOrContractMonth
  exp_str = f'{exp[:4]}-{exp[4:6]}-{exp[6:8]}'
  moneyness =  contract.strike - umd.close if contract.right == 'P' else umd.close - contract.strike
  time_value = price - moneyness if moneyness > 0 else price
  #%%
  plain += f'{contract.symbol}{SEP} {100*iv:.2f}{SEP} {contract.right}{SEP} {datetime.now().strftime("%Y-%m-%d %H:%M")}'
  plain += f'{SEP} {umd.close}{SEP} {exp_str}{SEP} {contract.strike}{SEP} {price:.2f}{SEP} {time_value:.2f}{SEP}'
  plain += f' {greeks_to_str(greeks.delta)}{SEP} {greeks_to_str(greeks.theta)}{SEP}'
  plain += f' {greeks_to_str(greeks.gamma)}{SEP} {greeks_to_str(greeks.vega)}{SEP} Check\n\n'

print(plain)
