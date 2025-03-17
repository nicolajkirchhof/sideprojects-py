# %%
from datetime import datetime, time

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

# %%
eu_indices = [utils.ibkr.get_and_qualify_contract_details(ib_con,ib.Index(x, 'EUREX', 'EUR')) for x in ['DAX', 'ESTX50']]
us_etfs = [utils.ibkr.get_and_qualify_contract_details(ib_con,ib.Stock(symbol=x, exchange='SMART', currency='USD')) for x in ['SPY', 'QQQ']]
de_tz = utils.exchanges.DE_EXCHANGE['TZ']
# us_tz = utils.exchanges.US_EXCHANGE['TZ']
us_tz = utils.exchanges.US_NY_EXCHANGE['TZ']

trading_class_map = {'DAX': 'ODAP', 'ESTX50': 'OEXP'}
noon = datetime.combine(datetime.now().date(), time(12, 00, 00))

target_times = [(de_tz.localize(noon), eu_indices),(us_tz.localize(noon), us_etfs)]
#%%
for contracts_detail in eu_indices:
  contract = contracts_detail.contract

  # for contract_detail in contracts_details:

  chains = ib_con.reqSecDefOptParams(futFopExchange='', underlyingSymbol=contract.symbol,
                                     underlyingSecType=contract.secType, underlyingConId=contract.conId)
  ##%%
  chain = [chain for chain in chains if contract.secType == 'STK' or chain.tradingClass == trading_class_map[contract.symbol]][0]
  next_expiration = chain.expirations[0]
  # option_contract = ib.Option(symbol=contract.symbol, exchange="SMART", multiplier=chain.multiplier, strike=0.0, lastTradeDateOrContractMonth=next_expiration, right="", currency=contract.currency)
  option_contract = ib.Option(symbol=contract.symbol, exchange=contract.exchange, multiplier=chain.multiplier, strike=0.0,
                              lastTradeDateOrContractMonth=next_expiration, right="", currency=contract.currency)
  ##%%
  option_contract_details = ib_con.reqContractDetails(option_contract)

  ##%%
  contract_ticker = utils.ibkr.get_options_data(ib_con, contract, signalParameterLive="impliedVolatility")[0]
  sigma_move, max_value, min_value = utils.ibkr.get_sigma_move(contract_ticker, 2.5, 0.5)

  ##%%
  relevant_option_contract_details = [od.contract for od in option_contract_details if
                                      min_value <= od.contract.strike <= max_value]

  ##%%
  option_contract_details_ticker = utils.ibkr.get_options_data(ib_con, relevant_option_contract_details,
                                                               signalParameterFrozen="modelGreeks", max_waittime=600)
  valid_option_contract_details_ticker = [ocdt for ocdt in option_contract_details_ticker if ocdt.modelGreeks is not None]


  ##%%
  low_put_wing = [ticker for ticker in valid_option_contract_details_ticker if
                  ticker.contract.right == 'P' and ticker.modelGreeks.delta <= -0.2][0]
  atm_put = [ticker for ticker in valid_option_contract_details_ticker if
             ticker.contract.right == 'P' and ticker.contract.strike > contract_ticker.last][0]
  high_call_wing = [ticker for ticker in valid_option_contract_details_ticker if
                    ticker.contract.right == 'C' and ticker.modelGreeks.delta >= 0.2][-1]
  atm_call = [ticker for ticker in valid_option_contract_details_ticker if
              ticker.contract.right == 'C' and ticker.contract.strike < contract_ticker.last][-1]


  # %%
  # Subscribe to live prices
  # low_put_wing = ib_con.reqMktData(low_put_wing.contract, "", False, False)
  # atm_put = ib_con.reqMktData(atm_put.contract, "", False, False)
  # high_call_wing = ib_con.reqMktData(high_call_wing.contract, "", False, False)
  # atm_call = ib_con.reqMktData(atm_call.contract, "", False, False)

  last_av = utils.ibkr.get_last_available

  def combo_contract_price():
    ib_con.sleep(1)
    return (last_av(low_put_wing, 'ask') +
            last_av(high_call_wing, 'ask') -
            last_av(atm_put, 'bid') -
            last_av(atm_call, 'bid'))


  ## %%
  print(f'''
  Created butterfly {next_expiration}
  Last: {contract_ticker.last} 
  Puts: Buy {low_put_wing.contract.strike} @ {last_av(low_put_wing,"ask")} Sell {atm_put.contract.strike} @ -{last_av(atm_put,"bid")}
  Calls: Buy {high_call_wing.contract.strike} @ {last_av(high_call_wing,"ask")} Sell {atm_call.contract.strike} @ -{last_av(atm_call,"bid")}
  Price: {combo_contract_price()}
  ''')

  # %% Create order and submit
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

  # %%
  # Specify the order
  order = ib.LimitOrder(
    action="BUY",  # Action for the entire combo
    totalQuantity=1,  # Quantity of combos
    lmtPrice=combo_contract_price(),  # Specify your limit price
    transmit=False
  )
  trade = ib_con.placeOrder(combo_contract, order)
  ib_con.sleep(1)

  # %%
  # ib_con.cancelMktData(low_put_wing.contract)
  # ib_con.cancelMktData(atm_put.contract)
  # ib_con.cancelMktData(high_call_wing.contract)
  # ib_con.cancelMktData(atm_call.contract)
