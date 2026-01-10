#%%
import re

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import finance.utils as utils
import scipy.stats as stats


# ... existing code ...
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import permutation_importance
from sklearn.inspection import PartialDependenceDisplay

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2


#%%
# load all xes
xs = []
for symbol in ['XLK', 'XLV', 'XLY', 'XLU', 'XLRE', 'XLP', 'XLI', 'XLF', 'XLE', 'XLB', 'XBI', 'XLC']:
  xs.append(utils.swing_trading_data.SwingTradingData(symbol, offline=True))

# %% [1] Consolidate Data
sector_prices = pd.DataFrame({x.symbol: x.df_month['c'] for x in xs}).sort_index()
monthly_actual_returns = sector_prices.pct_change()

# %% [2] Individual Timeframe Evaluation
lookbacks = {'1 Month': 1, '3 Month': 3, '6 Month': 6, '1 Year': 12}
n_pos = 3 # Number of top/bottom sectors to trade

plt.figure(figsize=(14, 8))

for label, window in lookbacks.items():
    # 1. Calculate the signal based on the specific window
    lookback_returns = sector_prices.pct_change(window)
    
    # 2. Rank sectors (1 = Leader, 12 = Laggard)
    ranks = lookback_returns.rank(axis=1, ascending=False)
    
    # 3. Create Position Matrix (1 for Long, -1 for Short, 0 for Neutral)
    positions = pd.DataFrame(0, index=sector_prices.index, columns=sector_prices.columns)
    positions[ranks <= n_pos] = 1
    positions[ranks > (len(xs) - n_pos)] = -1
    
    # 4. Calculate realized returns
    # We shift(1) positions because we buy at the END of month T based on signals
    # and realize the return of month T+1
    strategy_returns = (positions.shift(1) * monthly_actual_returns).mean(axis=1)
    
    # 5. Plot the equity curve for this specific lookback
    equity_curve = (1 + strategy_returns.fillna(0)).cumprod()
    equity_curve.plot(label=f'{label} Momentum', lw=2)

# 6. Add Benchmark (Equal weight all sectors)
benchmark = (1 + monthly_actual_returns.mean(axis=1).fillna(0)).cumprod()
benchmark.plot(label='Equal Weighted (Bench)', color='black', ls='--', alpha=0.6)

plt.title(f'Sector Rotation Comparison: Long Top {n_pos} / Short Bottom {n_pos} by Period')
plt.ylabel('Cumulative Performance')
plt.legend(loc='upper left')
plt.grid(alpha=0.3)
plt.yscale('log') # Use log scale for long-term growth comparison
plt.show()

# %% [3] Determine Current Leaders/Laggards per Timeframe
print("\n--- Current Leaders (Top 3) by Timeframe ---")
for label, window in lookbacks.items():
    current_rets = sector_prices.pct_change(window).iloc[-1].sort_values(ascending=False)
    leaders = current_rets.head(n_pos).index.tolist()
    laggards = current_rets.tail(n_pos).index.tolist()
    print(f"{label:10} | Leaders: {leaders} | Laggards: {laggards}")
