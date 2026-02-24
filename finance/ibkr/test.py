from datetime import datetime

from finance import utils

api = 'api_paper'
ib_con = utils.ibkr.connect(api, 17, 1)
symbol = 'NFLX'
import ib_async as ib
contract = ib.Stock(symbol=symbol, exchange='SMART', currency='USD')
ib_con.qualifyContracts(contract)
details = ib_con.reqContractDetails(contract)

#%%
end_date_time_str = datetime.now().strftime('%Y%m%d %H:%M:%S')
adj = ib_con.reqHistoricalData( contract, endDateTime="", durationStr='365 D', barSizeSetting='1 day', whatToShow='ADJUSTED_LAST', useRTH=True)

trades = ib_con.reqHistoricalData( contract, endDateTime=end_date_time_str, durationStr='90 D', barSizeSetting='1 day', whatToShow='TRADES', useRTH=True)
