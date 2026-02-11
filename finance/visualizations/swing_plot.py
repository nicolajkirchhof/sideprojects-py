from finance import utils
%load_ext autoreload
%autoreload 2


spy_data = utils.swing_trading_data.SwingTradingData('SPY', offline=True, metainfo=False)

#%%
utils.swing_plot.interactive_swing_plot(spy_data.df_day)
#%%
utils.swing_plot.export_swing_plot(spy_data.df_day, 'swing_plot.png')
