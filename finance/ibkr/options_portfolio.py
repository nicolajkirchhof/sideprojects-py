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
ib_con.connect('127.0.0.1', tws_paper_port, clientId=10, readonly=True)
# ib_con.connect('127.0.0.1', api_real_port, clientId=12, readonly=True)
ib_con.reqMarketDataType(2)

## %%
summary = ib_con.accountSummary()
values = ib_con.accountValues()
positions = ib_con.positions()
portfolio = ib_con.portfolio()
pnl = ib_con.pnl()
underlying_market_data = {}
# %%
option_positions = [position for position in positions if position.contract.secType == 'OPT']

actions = 'Actions: None/Roll/BuySold/TakeProfit/TakeLoss'
plain = f'Date | Symbol | Pos | P/L | Last | IV | Δ | Θ | Γ | ν | Action | Comment\n'
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

  daily_iv = ibkr_utils.yearly_to_daily_iv(market_data.modelGreeks.impliedVol)
  greeks = market_data.modelGreeks
  mkt_price = market_data.last * float(market_data.contract.multiplier)
  pnl = position.position * (mkt_price - position.avgCost)
  plain += f'{datetime.now().strftime("%Y-%m-%d %H:%M")} | {position.contract.symbol} {position.contract.lastTradeDateOrContractMonth} {position.contract.right} {position.contract.strike} | '
  plain += f'{position.position} | {pnl:.2f} | {umd.last} | '
  plain += f'{greeks.impliedVol * 100:.2f}, {daily_iv * 100:.2f} | {greeks.delta:.2f} | {greeks.theta:.2f} | '
  plain += f'{greeks.gamma:.2f} | {greeks.vega:.2f} |\n'

  html += f'<tr><td>{datetime.now().strftime("%Y-%m-%d %H:%M")}</td><td>{position.contract.symbol} {position.contract.lastTradeDateOrContractMonth} {position.contract.right} {position.contract.strike}</td><td>'
  html += f'{position.position}</td><td>{pnl:.2f}</td><td>{umd.last}</td><td>'
  html += f'{greeks.impliedVol * 100:.2f}, {daily_iv * 100:.2f}</td><td>{greeks.delta:.2f}</td><td>{greeks.theta:.2f}</td><td>'
  html += f'{greeks.gamma:.2f}</td><td>{greeks.vega:.2f}</td><td></td><td></td></tr>\n'

html += '</tbody></table>'
print(plain)
print(html)
