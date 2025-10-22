# %%
import os
from datetime import datetime

import ib_async as ib
import pandas as pd
import time
from zoneinfo import ZoneInfo

import finance.utils as utils

%load_ext autoreload
%autoreload 2

##%%
tws_instance = 'real'
ib_con = utils.ibkr.connect(tws_instance, 11, 2)

## %%
MAX_TRIES = 10
TRIGGER = "PortfolioUpdate"
KEY = "fjJheEoJRyGZ8IAWRzP2jvZfLgtWEj6PJs2fwbUd1Dz"

##%%
## %% PNL PCT
SEP = ','
header = f'Date {SEP} ContractId {SEP} Underlying {SEP} IV {SEP} Price {SEP} TimeValue {SEP} Δ {SEP} Θ {SEP} Γ {SEP} ν {SEP} Multiplier\n'
header_portfolio = f'Symbol {SEP} ContractId {SEP} Expiry {SEP} Pos {SEP} Right {SEP} Strike {SEP} Cost {SEP} Price {SEP} PnL {SEP} PnL%'

underlying_market_data = {}
last_market_data = {}
closing_order_states = {}
##%%
def print_and_notify(option_portfolio_position):
  #%%
  pnl = option_portfolio_position.unrealizedPNL
  pnl_pct = pnl * 100 / (abs(option_portfolio_position.position) * option_portfolio_position.averageCost)
  exp = option_portfolio_position.contract.lastTradeDateOrContractMonth
  exp_str = f'{exp[:4]}-{exp[4:6]}-{exp[6:8]}'
  line = f'{option_portfolio_position.contract.symbol}{SEP} {option_portfolio_position.contract.conId}{SEP}'
  line += f'{SEP} {exp_str}{SEP}{SEP}{option_portfolio_position.position:5}{SEP}'
  line += f'{option_portfolio_position.contract.right:2}{SEP} {option_portfolio_position.contract.strike}'
  line += f'{SEP}{option_portfolio_position.averageCost/int(option_portfolio_position.contract.multiplier):10.5f}{SEP}{option_portfolio_position.marketPrice:10.5f}'
  line += f'{SEP}{pnl:8.2f}{SEP}{pnl_pct:8.2f}{SEP}'
  color = utils.colors.Colors.BRIGHT_GREEN if pnl > 0 else utils.colors.Colors.BRIGHT_RED
  # Short PUT attention
  if option_portfolio_position.position < 0 and pnl_pct < -100:
    print(utils.colors.Colors.BG_RED + utils.colors.Colors.BRIGHT_WHITE + line + utils.colors.Colors.RESET)
    utils.ifttt.send_ifttt_webhook(TRIGGER, KEY, [line])
  else:
    print(color+line+utils.colors.Colors.RESET)

def isnan(x):
  return x != x

def get_price(market_data, type):
  return getattr(market_data, type) if getattr(market_data, type) > 0 else getattr(market_data, 'prev'+type.capitalize())


def log_position(position):
  #%%
  contract = position.contract
  ib_con.reqMarketDataType(2)
  market_data = ib_con.reqMktData(contract, "", True, False)
  ib_con.sleep(1)
  tries = 0
  while (market_data.modelGreeks is None or ((isnan(get_price(market_data, 'bid')) or
         isnan(get_price(market_data, 'ask')))  and
         isnan(get_price(market_data, 'last')))) and tries < MAX_TRIES:
    #print(f"Waiting {tries} / {MAX_TRIES} for option frozen data...")
    tries += 1
    ib_con.sleep(5)
  contract_details = ib_con.reqContractDetails(contract)[0]
  last_market_data[contract.conId] = market_data

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
  bid = get_price(market_data, 'bid')
  ask = get_price(market_data, 'ask')
  last = get_price(market_data, 'last')

  price = last
  if isnan(price) and not isnan(bid) and not isnan(ask):
    price = (bid + ask)/2
  iv = greeks.impliedVol if greeks.impliedVol is not None else -1
  ##%% Greeks sometimes return None
  greeks_to_str = lambda x: f'{x:.8f}' if x is not None else 'NaN'

  moneyness =  contract.strike - umd.close if contract.right == 'P' else umd.close - contract.strike
  time_value = price - moneyness if moneyness > 0 else price
  order = ib.MarketOrder("SELL" if position.position > 0 else "BUY", abs(position.position))
  state = ib_con.whatIfOrder(contract, order)
  closing_order_states[contract.conId] = [state, order]
  maintenance_margin = abs(float(state.maintMarginChange)) if hasattr(state, 'maintMarginChange') else 0

  #%%
  plain = f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}{SEP} {contract.conId}'
  plain += f'{SEP} {umd.close}{SEP} {iv:.2f}{SEP} {price:.5f}{SEP} {time_value:.5f}'
  plain += f'{SEP} {greeks_to_str(greeks.delta)}{SEP} {greeks_to_str(greeks.theta)}{SEP} {greeks_to_str(greeks.gamma)}{SEP} {greeks_to_str(greeks.vega)}'
  plain += f'{SEP} {maintenance_margin:.2f}'
  print(plain)
  with open(file, 'a', encoding='utf8') as f:
    f.write(plain+'\n')


file = f'N:/My Drive/Trading/portfolio_{tws_instance}.csv'
if not os.path.exists(file):
  with open(file, 'w', encoding='utf8') as f:
    f.write(header)
##%%
# while True:
##%%
summary = ib_con.accountSummary()
net_liq = [x for x in summary if x.account == 'U16408041' and x.tag == 'NetLiquidation'][0]
bpr = [x for x in summary if x.account == 'U16408041' and x.tag == 'BuyingPower'][0]
maint = [x for x in summary if x.account == 'U16408041' and x.tag == 'MaintMarginReq'][0]
excess = [x for x in summary if x.account == 'U16408041' and x.tag == 'AvailableFunds'][0]

capital = f'Net Liq{SEP} Maintenance{SEP} Excess{SEP} BPR\n'
capital += f'{datetime.now().strftime("%Y-%m-%d")}{SEP} {float(net_liq.value):.0f}{SEP} {float(maint.value):.0f}'
capital += f'{SEP} {float(excess.value):.0f}{SEP} {float(bpr.value):.0f}'
print(capital)
##%%
values = ib_con.accountValues()
portfolio = sorted(ib_con.portfolio(), key=lambda x: x.contract.conId)
positions = ib_con.positions()
option_portfolio_positions = [position for position in portfolio if position.contract.secType in ['OPT', 'FOP']  ]
option_portfolio_contracts = [position.contract for position in option_portfolio_positions  ]
ib_con.qualifyContracts(*option_portfolio_contracts)

print(f"---------------------{datetime.now().strftime("%Y-%m-%d %H:%M")}--------------------------------")
print(header_portfolio)
for option_portfolio_position in option_portfolio_positions:
  print_and_notify(option_portfolio_position)
print("\n Writing positions to file... \n")
print(header)
##%%
for option_portfolio_position in option_portfolio_positions:
  log_position(option_portfolio_position)



  # #%%
  # print(f"---------------------------------------------------------------------\n\n")
  # time.sleep(1800)

#%%
for option_portfolio_contract in option_portfolio_contracts:
  print(f'{option_portfolio_contract.symbol} => {option_portfolio_contract.multiplier}')

