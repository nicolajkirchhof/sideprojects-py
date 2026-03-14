from datetime import datetime

import numpy as np
import pandas as pd
from finance import utils

%load_ext autoreload
%autoreload 2

#%%
api = 'api_paper'
ib_con = utils.ibkr.connect(api, 5, 1)
symbol = 'NVA'
import ib_async as ib

#%%
index_cfd_euro = ['IBGB100', 'IBEU50', 'IBDE40', 'IBFR40', 'IBES35', 'IBNL25', 'IBCH20']
index_cfd_us = ['IBUS500', 'IBUS30', 'IBUST100']
index_cfd_asia = ['IBHK50', 'IBJP225', 'IBAU200']
index_cfds = [*index_cfd_euro, *index_cfd_us, *index_cfd_asia]
symbols = ['$ESTX50', '$V2TX', '$V1X', '$VIX', '$VXN', '$RVX', '$VXSLV', '$GVZ', '$OVX', '$SPX', '$NDX', '$RUT', '$INDU', '$CAC40', '$N225', '$HSI']
#%%

symbols = ['$ESTX50', '$V2TX', '$V1X', '$VIX', '$VXN', '$RVX', '$VXSLV', '$GVZ', '$OVX', '$SPX', '$NDX', '$RUT', '$INDU', '$CAC40', '$N225', '$HSI']
for symbol in symbols:
  df_day = utils.ibkr.daily_w_volatility(symbol, offline=False, ib_con=ib_con)


#%%

#%%
contract = ib.Stock(symbol=symbol, exchange='SMART', currency='USD')
ib_con.qualifyContracts(contract)
details = ib_con.reqContractDetails(contract)

#%%
end_date_time_str = datetime.now().strftime('%Y%m%d %H:%M:%S')
adj = ib_con.reqHistoricalData( contract, endDateTime="", durationStr='365 D', barSizeSetting='1 day', whatToShow='ADJUSTED_LAST', useRTH=True)

trades = ib_con.reqHistoricalData( contract, endDateTime=end_date_time_str, durationStr='90 D', barSizeSetting='1 day', whatToShow='TRADES', useRTH=True)

#%%
# Test divide by zero AIPT KPLTW
df_stk = utils.ibkr.daily_w_volatility('SYTAW', offline=True)
df_stk = utils.dolt_data.daily_w_volatility('SYTAW', offline=True)

SYTAW
#%%
df_stk = utils.ibkr.daily_w_volatility('CVSA', offline=False)
df_stk = utils.dolt_data.daily_w_volatility('CVSA')
data = utils.SwingTradingData('AIDX', datasource='offline')
