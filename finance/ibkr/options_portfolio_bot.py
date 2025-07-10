# %%
from datetime import datetime

import ib_async as ib
import pandas as pd
import time

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
summary = ib_con.accountSummary()
values = ib_con.accountValues()
portfolio = ib_con.portfolio()
underlying_market_data = {}
MAX_TRIES = 10
TRIGGER = "PortfolioUpdate"
KEY = "fjJheEoJRyGZ8IAWRzP2jvZfLgtWEj6PJs2fwbUd1Dz"

# %% PNL PCT

while True:
  positions = ib_con.positions()
  option_portfolio_positions = [position for position in portfolio if position.contract.secType in ['OPT', 'FOP']  ]
  SEP = ','

  combined_positions = []

  print("-----------------------------------------------------------")
  print(datetime.now().strftime("%Y-%m-%d %H:%M"))

  for option_portfolio_position in option_portfolio_positions:
    pnl = option_portfolio_position.unrealizedPNL
    pnl_pct = pnl * 100 / (abs(option_portfolio_position.position) * option_portfolio_position.averageCost)
    combined_positions.append({"comb_id": f"{option_portfolio_position.contract.lastTradeDateOrContractMonth} {option_portfolio_position.contract.symbol}", "pnl": pnl, "cost":option_portfolio_position.position*option_portfolio_position.averageCost})
    sym_strike = f"{option_portfolio_position.contract.symbol}@{option_portfolio_position.contract.strike}"
    line = f'{option_portfolio_position.contract.lastTradeDateOrContractMonth} {option_portfolio_position.position:5} {option_portfolio_position.contract.right:1} {sym_strike:15} pnl {pnl:8.2f} ( {pnl_pct:6.2f} % )'
    color = utils.colors.Colors.BRIGHT_GREEN if pnl > 0 else utils.colors.Colors.BRIGHT_RED
    # Short PUT attention
    if option_portfolio_position.position < 0 and pnl_pct < -100:
      print(utils.colors.Colors.BG_RED + utils.colors.Colors.BRIGHT_WHITE + line + utils.colors.Colors.RESET)
      utils.ifttt.send_ifttt_webhook(TRIGGER, KEY, [line])
    else:
      print(color+line+utils.colors.Colors.RESET)

  time.sleep(900)

# #%%
# df_combined_positions = pd.DataFrame(combined_positions)
# pnl_combined = df_combined_positions.groupby(['comb_id']).agg({'pnl': 'sum', 'cost': 'sum', }).assign( percentage=lambda x: x['pnl']*100 / abs(x['cost']) )
#
# pnl_combined['formatted'] = (pnl_combined.apply(lambda row:
#                                     f"{row['pnl']:,.0f} ( {row['percentage']:,.1f}% )", axis=1))

# %%

