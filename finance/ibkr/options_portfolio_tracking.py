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
ib.util.startLoop()
ib_con = ib.IB()
tws_real_port = 7497
tws_paper_port = 7498
api_real_port = 4001
api_paper_port = 4002
# ib_con.connect('127.0.0.1', tws_real_port, clientId=11, readonly=True)
# ib_con.connect('127.0.0.1', api_paper_port, clientId=11, readonly=True)
ib_con.connect('127.0.0.1', tws_paper_port, clientId=11, readonly=True)
# ib_con.connect('127.0.0.1', api_real_port, clientId=11, readonly=True)
ib_con.reqMarketDataType(2)

## %%
MAX_TRIES = 10
TRIGGER = "PortfolioUpdate"
KEY = "fjJheEoJRyGZ8IAWRzP2jvZfLgtWEj6PJs2fwbUd1Dz"


#%%
underlying_market_data = {}
def print_and_notify(option_portfolio_position):

  pnl = option_portfolio_position.unrealizedPNL
  pnl_pct = pnl * 100 / (abs(option_portfolio_position.position) * option_portfolio_position.averageCost)
  sym_strike = f"{option_portfolio_position.contract.symbol}@{option_portfolio_position.contract.strike}"
  line = f'{option_portfolio_position.contract.lastTradeDateOrContractMonth} {option_portfolio_position.position:5} {option_portfolio_position.contract.right:1} {sym_strike:15} pnl {pnl:8.2f} ( {pnl_pct:6.2f} % )'
  color = utils.colors.Colors.BRIGHT_GREEN if pnl > 0 else utils.colors.Colors.BRIGHT_RED
  # Short PUT attention
  if option_portfolio_position.position < 0 and pnl_pct < -100:
    print(utils.colors.Colors.BG_RED + utils.colors.Colors.BRIGHT_WHITE + line + utils.colors.Colors.RESET)
    utils.ifttt.send_ifttt_webhook(TRIGGER, KEY, [line])
  else:
    print(color+line+utils.colors.Colors.RESET)


def log_position(contract):
  #%%
  ib_con.reqMarketDataType(2)
  market_data = ib_con.reqMktData(contract, "", True, False)
  tries = 0
  while market_data.modelGreeks is None and tries < MAX_TRIES:
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
  if market_data.lastGreeks is not None:
    greeks = market_data.lastGreeks
  elif market_data.modelGreeks is not None:
    greeks = market_data.modelGreeks

  bid = market_data.bid if market_data.bid > 0 else market_data.prevBid
  ask = market_data.ask if market_data.ask > 0 else market_data.prevAsk
  iv = greeks.impliedVol if greeks.impliedVol is not None else -1
  ##%% Greeks sometimes return None
  greeks_to_str = lambda x: f'{1000*x:.0f}' if x is not None else 'NaN'
  exp = contract.lastTradeDateOrContractMonth
  exp_str = f'{exp[:4]}-{exp[4:6]}-{exp[6:8]}'
  last = (bid + ask)/2
  moneyness =  contract.strike - umd.close if contract.right == 'P' else umd.close - contract.strike
  time_value = last - moneyness if moneyness > 0 else last
  #%%
  plain = f'{datetime.now(ZoneInfo('UTC')).strftime("%Y-%m-%d %H:%M:%SZ")} {SEP} {contract.conId} {SEP} {contract.symbol}{SEP} '
  plain += f'{umd.close}{SEP} {100*iv:.2f}{SEP} {exp_str}{SEP} {contract.strike}{SEP} {contract.right}{SEP} {last:.2f}{SEP} {time_value:.2f}{SEP}'
  plain += f' {greeks_to_str(greeks.delta)}{SEP} {greeks_to_str(greeks.theta)}{SEP}'
  plain += f' {greeks_to_str(greeks.gamma)}{SEP} {greeks_to_str(greeks.vega)}\n'
  with open(file, 'a', encoding='utf8') as f:
    f.write(plain)

# %% PNL PCT
SEP = ','
header = f'Date {SEP} ContractId {SEP} Symbol {SEP} Underlying {SEP} IV {SEP} Exp {SEP} Strike {SEP} Right {SEP} Price {SEP} TimeValue {SEP} Δ {SEP} Θ {SEP} Γ {SEP} ν\n'

file = f'N:/My Drive/Trading/portfolio.csv'
if not os.path.exists(file):
  with open(file, 'w', encoding='utf8') as f:
    f.write(header)

while True:
  summary = ib_con.accountSummary()
  values = ib_con.accountValues()
  portfolio = ib_con.portfolio()
  positions = ib_con.positions()
  option_portfolio_positions = [position for position in portfolio if position.contract.secType in ['OPT', 'FOP']  ]
  option_portfolio_contracts = [position.contract for position in option_portfolio_positions  ]
  ib_con.qualifyContracts(*option_portfolio_contracts)

  print(f"---------------------{datetime.now().strftime("%Y-%m-%d %H:%M")}--------------------------------")
  for option_portfolio_position in option_portfolio_positions:
    print_and_notify(option_portfolio_position)
  print("\n Writing positions to file... \n")
  for option_portfolio_contract in option_portfolio_contracts:
    log_position(option_portfolio_contract)
  print(f"---------------------------------------------------------------------\n\n")
  time.sleep(900)

# #%%
# df_combined_positions = pd.DataFrame(combined_positions)
# pnl_combined = df_combined_positions.groupby(['comb_id']).agg({'pnl': 'sum', 'cost': 'sum', }).assign( percentage=lambda x: x['pnl']*100 / abs(x['cost']) )
#
# pnl_combined['formatted'] = (pnl_combined.apply(lambda row:
#                                     f"{row['pnl']:,.0f} ( {row['percentage']:,.1f}% )", axis=1))

# %%


for contract in contracts:


print(plain)
