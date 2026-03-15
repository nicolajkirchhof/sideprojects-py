from finance import utils

%load_ext autoreload
%autoreload 2

# %%
# Load SPY in offline mode by default
# Start interactive app
spy = utils.SwingTradingData('SPY', datasource='offline')
utils.swing_plot.interactive()
