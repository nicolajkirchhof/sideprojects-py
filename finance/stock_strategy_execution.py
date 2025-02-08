#%%
import numpy as np
import scipy
from ib_async import ib, util, IB, Forex, Stock
import backtrader as bt

from finance.strategies.avg_sig_change_strategy import AverageSignificantChangeStrategy
from finance.strategies.fixed_time_strategy import FixedTimeStrategy
from finance.strategies.bull_flag_dip_strategy import BullFlagDipStrategy
from finance.strategies.ema_strategy import EmaStrategy
from finance.strategies.in_out_strategy import InOutStrategy
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

from finance.utils import percentage_change

mpl.use('TkAgg')
mpl.use('QtAgg')

%load_ext autoreload
%autoreload 2


#%%
# stock_name = 'TSLA'
stock_name = 'SPY'
df_contract = pd.read_pickle(f'finance/ibkr_finance_data/{stock_name}.pkl')

p1_p0 = pd.DataFrame(df_contract['average'] - df_contract['average'].shift(1))
p1_p0['x'] = 1
p1_p0['angle'] = np.arctan2(p1_p0['x'], p1_p0['average'])

p1_p2 = pd.DataFrame(df_contract['average'].shift(2) - df_contract['average'].shift(1))
p1_p2['x'] = -1
p1_p2['angle'] = np.arctan2(p1_p2['x'], p1_p2['average'])

df_contract['deg_change'] = (p1_p0['angle'] - p1_p2['angle'] ) *(180/np.pi)
df_contract['avg_change'] =  100 * (df_contract['average'] - df_contract['average'].shift(1)) / df_contract['average'].shift(1)

##%%
class IbkrPandasData(bt.feeds.PandasData):
  lines = ('average','deg_change')
  params = (
    ('datetime', 'timestamp'),
    ('open', -1),
    ('high', -1),
    ('low', -1),
    ('close', -1),
    ('volume', -1),
    ('openinterest',None),
    ('average', -1),
    ('barCount', -1),
    ('ta_bid', -1),
    ('ta_ask', -1),
    ('max_ask', -1),
    ('min_bid', -1),
    ('historical_volatility', -1),
    ('option_implied_volatility', -1),
    ('deg_change', -1),
  )


#%%
plt.close()
print('=============================== NEW RUN ======================================')
cerebro = bt.Cerebro(preload=True)
data = IbkrPandasData(dataname=df_contract, datetime='date') #, todate=pd.Timestamp('2024-03-01')) #fromdate=pd.Timestamp('2024-03-01'), todate=pd.Timestamp('2024-07-01'))
# data = IbkrPandasData(dataname=df_contract, datetime='date', fromdate=pd.Timestamp('2024-02-28'), todate=pd.Timestamp('2024-06-01'))
cerebro.adddata(data)
cerebro.broker.setcash(100000)
cerebro.broker.setcommission(commission=1, name='us', margin=True)
cerebro.broker.set_coc(True)

# Add the printer as a strategy
# cerebro.addstrategy(InOutStrategy)
# cerebro.addstrategy(BullFlagDipStrategy)
# cerebro.addstrategy(AverageSignificantChangeStrategy)
for bd in range(0, 5):
  for hold_days in range(0, 50):
    cerebro.addstrategy(FixedTimeStrategy, buyday=bd, intervaldays=hold_days)
# cerebro.addstrategy(EmaStrategy)
# cerebro.addstrategy(MyStrategy)
# cerebro.addstrategy(bt.Strategy)

results = cerebro.run()
#%%
all_bd = [r.params.buyday for r in results]
all_id = [r.params.intervaldays for r in results]
all_pl = [r.allPL for r in results]
all_w = [r.wins for r in results]
all_l = [r.losses for r in results]
for result in results:
  print(f'BuyDate {result.params.buyday} IntervalDays {result.params.intervaldays} AP/L {result.allPL:.0f} AW {result.wins} AL {result.losses}')

df_eval = pd.DataFrame({'bd': all_bd, 'id': all_id, 'pl': all_pl, 'w': all_w, 'l': all_l})

df_eval.to_pickle('finance/_data/spy_options_eval.pkl')
##%%
# cerebro.plot(style='bar', iplot=False)
