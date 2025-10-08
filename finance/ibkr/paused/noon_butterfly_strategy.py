# %%
from datetime import datetime, time, timedelta, timezone

import ib_async as ib
import finance.utils as utils

%load_ext autoreload
%autoreload 2

##%%
ib.util.startLoop()
ib_con = ib.IB()
tws_real_port = 7497
tws_paper_port = 7497
api_real_port = 4001
api_paper_port = 4002
# ib_con.connect('127.0.0.1', api_paper_port, clientId=5, readonly=True)
ib_con.connect('127.0.0.1', tws_paper_port, clientId=5, readonly=False)
# ib_con.connect('127.0.0.1', api_real_port, clientId=5, readonly=True)
ib_con.reqMarketDataType(2)

## %%
eu_indices = [utils.ibkr.get_and_qualify_contract_details(ib_con,ib.Index(x, 'EUREX', 'EUR')) for x in ['DAX', 'ESTX50']]
us_indices = [utils.ibkr.get_and_qualify_contract_details(ib_con, ib.Index(symbol=x[0], exchange=x[1], currency='USD')) for x in [('XSP', 'CBOE')]]
de_tz = utils.exchanges.DE_EXCHANGE['TZ']
# us_tz = utils.exchanges.US_EXCHANGE['TZ']
us_tz = utils.exchanges.US_NY_EXCHANGE['TZ']


trading_class_map = {'DAX': 'ODAP', 'ESTX50': 'OEXP', 'XSP': 'XSP'}

min_noon = datetime.combine(datetime.now().date(), time(11, 45, 00))
noon = datetime.combine(datetime.now().date(), time(12, 00, 00))
max_noon = datetime.combine(datetime.now().date(), time(12, 30, 00))

target_times = {'DAX': de_tz, 'ESTX50': de_tz, 'XSP':us_tz}
indices = [*eu_indices, *us_indices]
##%%
contracts_detail = us_indices[0]
# contracts_detail = eu_indices[1]
#%%
for contracts_detail in indices[1:]:
  if not target_times[contracts_detail.contract.symbol].localize(min_noon) <= datetime.now(timezone.utc) <= target_times[contracts_detail.contract.symbol].localize(max_noon):
    print(f'Skipping {contracts_detail.contract.symbol} ...')
    continue

##%%
  contract = contracts_detail.contract
  chains = ib_con.reqSecDefOptParams(futFopExchange='', underlyingSymbol=contract.symbol,
                                     underlyingSecType=contract.secType, underlyingConId=contract.conId)
  ##%%
  chain = [chain for chain in chains if chain.tradingClass == trading_class_map[contract.symbol]][0]
  next_expiration = chain.expirations[0]
  # option_contract = ib.Option(symbol=contract.symbol, exchange="SMART", multiplier=chain.multiplier, strike=0.0, lastTradeDateOrContractMonth=next_expiration, right="", currency=contract.currency)
  option_contract = ib.Option(symbol=contract.symbol, exchange=contract.exchange, multiplier=chain.multiplier, strike=0.0,
                              lastTradeDateOrContractMonth=next_expiration, right="", currency=contract.currency)
  ##%%
  option_contract_details = ib_con.reqContractDetails(option_contract)

  ##%%
  contract_ticker = utils.ibkr.get_options_data(ib_con, contract, signalParameterLive="impliedVolatility")[0]
  sigma_move, max_value, min_value = utils.ibkr.get_sigma_move(contract_ticker, 2.5, 1)

  ##%%
  relevant_option_contract_details = [od.contract for od in option_contract_details if
                                      min_value <= od.contract.strike <= max_value]
  # relevant_option_contract_details = [od.contract for od in option_contract_details]

  ##%%
  option_contract_details_ticker = utils.ibkr.get_options_data(ib_con, relevant_option_contract_details,
                                                               signalParameterFrozen="modelGreeks", max_waittime=600)
  valid_option_contract_details_ticker = [ocdt for ocdt in option_contract_details_ticker if ocdt.modelGreeks is not None]

  ##%%
  puts = [ticker for ticker in valid_option_contract_details_ticker if ticker.contract.right == 'P']
  calls = [ticker for ticker in valid_option_contract_details_ticker if ticker.contract.right == 'C']
  sorted_calls = sorted(calls, key=lambda x: x.contract.strike)
  sorted_puts = sorted(puts, key=lambda x: x.contract.strike)

  ##%%
  low_put_wing = [ticker for ticker in sorted_puts if ticker.modelGreeks.delta <= -0.2][0]
  atm_put = [ticker for ticker in sorted_puts if ticker.contract.strike > contract_ticker.last][0]
  high_call_wing = [ticker for ticker in sorted_calls if ticker.modelGreeks.delta >= 0.2][-1]
  atm_call = [ticker for ticker in sorted_calls if ticker.contract.strike < contract_ticker.last][-1]


  ## %%
  # Subscribe to live prices
  # low_put_wing = ib_con.reqMktData(low_put_wing.contract, "", False, False)
  # atm_put = ib_con.reqMktData(atm_put.contract, "", False, False)
  # high_call_wing = ib_con.reqMktData(high_call_wing.contract, "", False, False)
  # atm_call = ib_con.reqMktData(atm_call.contract, "", False, False)

  last_av = utils.ibkr.get_last_available

  def combo_contract_bid_price():
    ib_con.sleep(1)
    return (last_av(low_put_wing, 'ask') +
            last_av(high_call_wing, 'ask') -
            last_av(atm_put, 'bid') -
            last_av(atm_call, 'bid'))

  def combo_contract_ask_price():
    ib_con.sleep(1)
    return (last_av(low_put_wing, 'bid') +
            last_av(high_call_wing, 'bid') -
            last_av(atm_put, 'ask') -
            last_av(atm_call, 'ask'))

  ## %%
  print(f'''
  Created butterfly {next_expiration}
  Last: {contract_ticker.last} 
  \t\t\tWingPut: Buy {low_put_wing.contract.strike} @ {last_av(low_put_wing,"ask")} 
  AtmCall:  Sell {atm_call.contract.strike} @ -{last_av(atm_call,"bid")}
  \t\t\tAtmPut:  Sell {atm_put.contract.strike} @ -{last_av(atm_put,"bid")}
  WingCall: Buy {high_call_wing.contract.strike} @ {last_av(high_call_wing,"ask")} 
  
  Bid: {combo_contract_bid_price()} Ask: {combo_contract_ask_price()}
  ''')

  ## %% Create order and submit
  exchange = contract.exchange
  combo_contract = ib.Bag(
    symbol=contract.symbol,  # Underlying ticker symbol
    currency=contract.currency,
    exchange=contract.exchange
  )

  leg_low_put_wing = ib.ComboLeg(action='BUY', ratio=1, conId=low_put_wing.contract.conId, exchange=exchange)
  leg_atm_put = ib.ComboLeg(action='SELL', ratio=1, conId=atm_put.contract.conId, exchange=exchange)
  leg_high_call_wing = ib.ComboLeg(action='BUY', ratio=1, conId=high_call_wing.contract.conId, exchange=exchange)
  leg_atm_call = ib.ComboLeg(action='SELL', ratio=1, conId=atm_call.contract.conId, exchange=exchange)

  combo_contract.comboLegs = [leg_low_put_wing, leg_atm_put, leg_high_call_wing, leg_atm_call]

## %%
  # Specify the order
  order = ib.LimitOrder(
    action="BUY",  # Action for the entire combo
    totalQuantity=1,  # Quantity of combos
    lmtPrice=combo_contract_ask_price(),  # Specify your limit price
    transmit=False
  )

  trade = ib_con.placeOrder(combo_contract, order)
  ib_con.sleep(5)
  bid_ask_spread = combo_contract_ask_price() - combo_contract_bid_price()
  bid_ask_spread_pct = bid_ask_spread * 0.1 if bid_ask_spread * 0.1 >= 0.1 else 0.1

##%%
  order.transmit = True
  while trade.orderStatus.status != 'Filled':
    order.lmtPrice = round(order.lmtPrice + bid_ask_spread_pct, 1)
    print(f'Adapted order price to {order.lmtPrice} ...')
    ib_con.placeOrder(combo_contract, order)
    ib_con.sleep(5)

  # %%
  # ib_con.cancelMktData(low_put_wing.contract)
  # ib_con.cancelMktData(atm_put.contract)
  # ib_con.cancelMktData(high_call_wing.contract)
  # ib_con.cancelMktData(atm_call.contract)
