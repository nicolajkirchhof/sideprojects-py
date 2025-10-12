# %%
fplt.right_margin_candles = 0
fplt.side_margin = 0
# fplt.winx = 0
# fplt.winy = 0
# fplt.winh = 2160
# fplt.winw = 3840
fplt.timestamp_format = '%H:%M'
fplt.time_splits = [('years', 2*365*24*60*60,    'YS',  4), ('months', 3*30*24*60*60,   'MS', 10), ('weeks',   3*7*24*60*60, 'W-MON', 10),
                    ('days',      3*24*60*60,     'D', 10), ('hours',        9*60*60,   'h', 16), ('hours',        3*60*60,     'h', 16),
                    ('minutes',        45*60, '15min', 16), ('minutes',        15*60, '5min', 16), ('minutes',         3*60,   'min', 16),
                    ('seconds',           45,   '15s', 19), ('seconds',           15,   '5s', 19), ('seconds',            3,     's', 19),
                    ('milliseconds',       0,    'ms', 23)]
#%%
ax, ax2, ax3 = fplt.create_plot('DAX', rows=3)
# ax.decouple()
# ax2.decouple()
# ax3.decouple()

create_interactive_plot(ax, '2m', df_2m)
# create_interactive_plot(ax2, '5m', df_5m)
create_interactive_plot(ax2, '10m', df_10m)

# create_interactive_plot(ax4, '15m', df_15m)
create_interactive_plot(ax3, '60m', df_60m)
# create_interactive_plot(ax3, '50m', df_50m)
fplt.windows[0].ci.layout.setRowStretchFactor(0, 1)
fplt.windows[0].ci.layout.setRowStretchFactor(1, 1)
fplt.windows[0].ci.layout.setRowStretchFactor(2, 1)

fplt.show()

#%%
# ax, ax2, ax3 = fplt.create_plot('DAX', rows=3)
fplt.create_plot('DAX')
fplt.candlestick_ochl(df_60m[['o', 'c', 'h', 'l']], candle_width=30)
fplt.candlestick_ochl(df_10m[['o', 'c', 'h', 'l']], candle_width=5)
fplt.candlestick_ochl(df_2m[['o', 'c', 'h', 'l']])
fplt.show()

#%%
# no margins
fplt.right_margin_candles = 0
fplt.side_margin = 0
ax= fplt.create_plot('DAX', rows=1)
fplt.candlestick_ochl(df[['o', 'c', 'h', 'l']], ax=ax)
# Aligning ticks to each hour
# ax_date_range = pd.date_range(df.index.min(), df.index.max(), freq='h')
# num_ticks = int((df.index.max()- df.index.min()).total_seconds()/3600)
# ax_date_range = ax.getAxis('bottom').tickValues(df.index.min(), df.index.max(), num_ticks)
# ax.getAxis('bottom').setTicks([
# ax.getAxis('bottom').setTicks([
#   [(date.timestamp(), date.strftime('%H:%M')) for date in ax_date_range]
# ])
fplt.show()

#%%
