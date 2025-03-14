# %%
from datetime import datetime

import ib_async as ib
import finance.ibkr.utils as ibkr_utils

%load_ext autoreload
%autoreload 2

##%%
ib.util.startLoop()
ib_con = ib.IB()
tws_real_port = 7497
tws_paper_port = 7497
api_real_port = 4001
api_paper_port = 4002
# ib_con.connect('127.0.0.1', api_paper_port, clientId=11, readonly=True)
ib_con.connect('127.0.0.1', tws_paper_port, clientId=11, readonly=True)
# ib_con.connect('127.0.0.1', api_real_port, clientId=11, readonly=True)
ib_con.reqMarketDataType(2)

## %%
summary = ib_con.accountSummary()
values = ib_con.accountValues()
positions = ib_con.positions()
portfolio = ib_con.portfolio()
pnl = ib_con.pnl()
underlying_market_data = {}

## %%
option_positions = [position for position in positions if position.contract.secType == 'OPT']
SEP = ';'

actions = 'Actions: None/Roll/BuySold/TakeProfit/TakeLoss'
plain = f'Date {SEP} Symbol {SEP} Date {SEP} Right {SEP} Strike {SEP} Pos {SEP} P/L {SEP} Last {SEP} IV {SEP} Δ {SEP} Θ {SEP} Γ {SEP} ν {SEP} Action {SEP} Comment\n'
html = '''<table class="table table-bordered"><tbody>
<tr><th>Date</th><th>Symbol</th><th>Pos</th><th>P/L</th><th>Last</th><th>IV</th><th>Δ</th><th>Θ</th><th>Γ</th><th>ν</th><th>Action</th><th>Comment</th></tr>'''
# position = option_positions[3]
for position in option_positions:
  print(f'Processing {position.contract.symbol}...')
  ib_con.qualifyContracts(position.contract)

  ib_con.reqMarketDataType(2)
  market_data = ib_con.reqMktData(position.contract, "", True, False)
  while market_data.modelGreeks is None:
    print("Waiting for option frozen data...")
    ib_con.sleep(1)
  contract_details = ib_con.reqContractDetails(position.contract)[0]

  if contract_details.underConId in underlying_market_data:
    umd = underlying_market_data[contract_details.underConId]
  else:
    underlying = ib.Contract(symbol=contract_details.underSymbol, secType=contract_details.underSecType,
                             conId=contract_details.underConId)
    ib_con.qualifyContracts(underlying)
    ib_con.sleep(1)

    umd = ib_con.reqMktData(underlying, "", True, False)
    while ib.util.isNan(umd.last):
      print("Waiting for underlying frozen data...")
      ib_con.sleep(1)
    underlying_market_data[underlying.conId] = umd

  greeks = market_data.modelGreeks
  iv = greeks.impliedVol if greeks.impliedVol is not None else -1
  daily_iv = ibkr_utils.yearly_to_daily_iv(iv)
  ##%% Greeks sometimes return None
  greeks_to_str = lambda x: f'{x:.2}' if x is not None else 'NaN'
  mkt_price = market_data.last * float(market_data.contract.multiplier)
  pnl = position.position * (mkt_price - position.avgCost)
  plain += f'{datetime.now().strftime("%Y-%m-%d %H:%M")} {SEP} {position.contract.symbol} {SEP} {position.contract.lastTradeDateOrContractMonth} {SEP} {position.contract.right} {SEP} {position.contract.strike} {SEP} '
  plain += f'{position.position} {SEP} {pnl:.2f} {SEP} {umd.last} {SEP}'
  plain += f'{iv * 100:.2f}, {daily_iv * 100:.2f} {SEP} {greeks_to_str(greeks.delta)} {SEP} {greeks_to_str(greeks.theta)} {SEP} '
  plain += f'{greeks_to_str(greeks.gamma)} {SEP} {greeks_to_str(greeks.vega)} {SEP}\n'

  html += f'<tr><td>{datetime.now().strftime("%Y-%m-%d %H:%M")}</td><td>{position.contract.symbol} {position.contract.lastTradeDateOrContractMonth} {position.contract.right} {position.contract.strike}</td><td>'
  html += f'{position.position}</td><td>{pnl:.2f}</td><td>{umd.last}</td><td>'
  html += f'{iv * 100:.2f}, {daily_iv * 100:.2f}</td><td>{greeks_to_str(greeks.delta)}</td><td>{greeks_to_str(greeks.theta)}</td><td>'
  html += f'{greeks_to_str(greeks.gamma)}</td><td>{greeks_to_str(greeks.vega)}</td><td></td><td></td></tr>\n'

html += '</tbody></table>'
print(plain)
print(html)
