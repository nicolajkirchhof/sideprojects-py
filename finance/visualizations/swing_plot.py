from finance import utils
%load_ext autoreload
%autoreload 2


spy_data = utils.swing_trading_data.SwingTradingData('SPY', offline=True, metainfo=False)

#%%
utils.swing_plot.interactive(spy_data.df_day)
#%%
utils.swing_plot.export(spy_data.df_day, 'swing_plot.png')
