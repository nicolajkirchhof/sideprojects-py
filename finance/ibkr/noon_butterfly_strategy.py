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
# ib_con.connect('127.0.0.1', api_paper_port, clientId=5, readonly=True)
ib_con.connect('127.0.0.1', tws_paper_port, clientId=5, readonly=True)
# ib_con.connect('127.0.0.1', api_real_port, clientId=5, readonly=True)
ib_con.reqMarketDataType(2)

## %%
eu_indices = [ib.Index(x, 'EUREX', 'EUR') for x in ['DAX', 'ESTX50']]
us_futures =[*[ib.ContFuture(symbol=x[0], multiplier=x[1], exchange='CME',currency='USD', includeExpired=True) for x in [('MES', '5'), ('MNQ', '2'), ('RTY', '50')]]]

contracts_details = [ibkr_utils.get_and_qualify_contract_details(ib_con, x) for x in eu_indices + us_futures]

trading_class_map = {'DAX': 'ODAP'}
#%%
contracts_detail = contracts_details[0]
contract = contracts_detail.contract

# for contract_detail in contracts_details:

chains = ib_con.reqSecDefOptParams(futFopExchange='',underlyingSymbol=contract.symbol, underlyingSecType=contract.secType,underlyingConId=contract.conId)
chain = [chain for chain in chains if chain.tradingClass == trading_class_map[contract.symbol]][0]
next_expiration = chain.expirations[1]
# option_contract = ib.Option(symbol=contract.symbol, exchange="SMART", multiplier=chain.multiplier, strike=0.0, lastTradeDateOrContractMonth=next_expiration, right="", currency=contract.currency)
option_contract = ib.Option(symbol=contract.symbol, exchange=contract.exchange, multiplier=chain.multiplier, strike=0.0, lastTradeDateOrContractMonth=next_expiration, right="", currency=contract.currency)
#%%
option_contract_details = ib_con.reqContractDetails(option_contract)

#%%
contract_ticker = ibkr_utils.get_options_data(ib_con, contract, signalParameterLive="impliedVolatility")[0]
sigma_move, max_value, min_value = ibkr_utils.get_sigma_move(contract_ticker, 1.5, 0.5)

#%%
relevant_option_contract_details = [od.contract for od in option_contract_details if min_value <= od.contract.strike <= max_value]

#%%
option_contract_details_ticker = ibkr_utils.get_options_data(ib_con, relevant_option_contract_details, signalParameterFrozen="modelGreeks", max_waittime=600)
valid_option_contract_details_ticker  = [ocdt for ocdt in option_contract_details_ticker if ocdt.modelGreeks is not None]

#%%
low_put_wing = [ticker for ticker in valid_option_contract_details_ticker if ticker.contract.right == 'P' and  ticker.modelGreeks.delta <= -0.2][0]
atm_put = [ticker for ticker in valid_option_contract_details_ticker if ticker.contract.right == 'P' and  ticker.contract.strike > contract_ticker.last][0]
high_call_wing = [ticker for ticker in valid_option_contract_details_ticker if ticker.contract.right == 'C' and  ticker.modelGreeks.delta >= 0.2][-1]
atm_call = [ticker for ticker in valid_option_contract_details_ticker if ticker.contract.right == 'C' and  ticker.contract.strike < contract_ticker.last][-1]

#%% Create order and submit
