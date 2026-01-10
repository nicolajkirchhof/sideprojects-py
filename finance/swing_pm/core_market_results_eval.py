# %%
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns
from glob import glob
import re
from collections import Counter

pd.set_option("display.max_columns", None)  # None means "no limit"
pd.set_option("display.max_rows", None)  # None means "no limit"
pd.set_option("display.width", 140)  # Set the width to 80 characters

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2

# %%
symbol = 'SPY'
# symbol = 'QQQ'
# symbol = 'IWM'
strategies = ['NP', 'C', 'NC', 'P']
features = '2'

# Create a figure with 2x2 subplots - removed sharex=True to allow independent ordering
fig, axes = plt.subplots(2, 2, figsize=(20, 14))
axes = axes.flatten()

for i, strategy in enumerate(strategies):
    eval_files = glob(f'finance/analysis/core/{symbol}/{strategy}_Strategy/[{features}]*.csv')
    
    if not eval_files:
        axes[i].set_title(f"No data for {strategy}")
        continue
        
    df_results = pd.concat([pd.read_csv(r) for r in eval_files])

    # Process regimes
    df_results.sort_values('score', ascending=False, inplace=True)
    combined_string = " ".join(df_results['regimes'][0:15].astype(str))
    cleaned_string = re.sub(r"[\[\]'|,]", " ", combined_string)
    word_counts = Counter(cleaned_string.split())
    
    # Sort by index (the regime names) instead of values
    counts_series = pd.Series(word_counts).sort_index()

    # Plot using the alphabetically sorted index
    sns.barplot(x=counts_series.index, y=counts_series.values, ax=axes[i], 
                palette="magma", order=counts_series.index)
    
    axes[i].set_title(f"{strategy} Strategy - Regime Frequencies")
    axes[i].set_ylabel("Count")
    axes[i].set_xlabel("") # Clear x label to save space
    
    # Rotate labels for readability
    for label in axes[i].get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment('right')
        
    axes[i].grid(axis='y', linestyle='--', alpha=0.5)

plt.suptitle(f"Regime Analysis for {symbol} (Features: {features})", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
