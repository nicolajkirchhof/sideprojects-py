from datetime import timedelta, datetime

from dateutil import parser
import numpy as np
import scipy

import pandas as pd
import os

import influxdb as idb

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import mplfinance as mpf

import finplot as fplt
import pytz

import finance.utils as utils
from finance.utils.influx import DB_CFD, DB_INDEX, DB_FOREX

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2


#%% get influx data
influx_client_df, influx_client = utils.influx.get_influx_clients()


de_exchange = {'TZ':pytz.timezone('Europe/Berlin'), 'Open': timedelta(hours=9), 'Close': timedelta(hours=17, minutes=30), 'PostClose': timedelta(hours=22), 'PreOpen': timedelta(hours=8)}
gb_exchange = {'TZ':pytz.timezone('Europe/London'), 'Open': timedelta(hours=8), 'Close': timedelta(hours=16, minutes=30), 'PostClose': timedelta(hours=17), 'PreOpen': timedelta(hours=4, minutes=30)}
us_exchange = {'TZ':pytz.timezone('America/Chicago'), 'Open': timedelta(hours=8), 'Close': timedelta(hours=16, minutes=30), 'PostClose': timedelta(hours=17), 'PreOpen': timedelta(hours=4, minutes=30)}
jp_exchange = {'TZ':pytz.timezone('Asia/Tokyo'), 'Open': timedelta(hours=8, minutes=45), 'Close': timedelta(hours=15, minutes=45), 'PostClose': timedelta(hours=30), 'PreOpen': timedelta(hours=8, minutes=45)}
hk_exchange = {'TZ':pytz.timezone('Asia/Hong_Kong'), 'Open': timedelta(hours=9, minutes=30), 'Close': timedelta(hours=16, minutes=0), 'PostClose': timedelta(hours=26), 'PreOpen': timedelta(hours=9, minutes=30)}
au_exchange = {'TZ':pytz.timezone('Australia/Sydney'), 'Open': timedelta(hours=10, minutes=0), 'Close': timedelta(hours=16, minutes=0), 'PostClose': timedelta(hours=19), 'PreOpen': timedelta(hours=7, minutes=0)}
exchanges = {'DE': de_exchange, 'GB': gb_exchange, 'US': us_exchange, 'JP': jp_exchange, 'HK': hk_exchange}

exchange_mapping = {'IBDE40': 'DE', 'IBNL25':'DE', 'IBCH20': 'DE', 'IBES35':'DE', 'IBEU50':'DE', 'IBFR40':'DE',
                    'IBGB100':'GB', 'IBUS30':'US', 'IBUS500':'US', 'IBUST100':'US', 'IBJP225': 'JP', 'IBHK50': 'HK',
                    'IBAU200': 'AU'}
#%%
tz = pytz.utc
# Create a directory
# tz = pytz.timezone('EST')
symbol = 'EURGBP'
# db = DB_CFD
# db = DB_INDEX
db = DB_FOREX

dfs_ref_range = []
dfs_closing = []

test_day_feb = tz.localize(parser.parse('2024-02-01T00:00:00'))
test_day_aug = tz.localize(parser.parse('2024-05-08T00:00:00'))
test_day_oct = tz.localize(parser.parse('2024-10-08T00:00:00'))
test_day_nov = tz.localize(parser.parse('2024-11-08T00:00:00'))
##%%

df_feb = influx_client_df.query(utils.influx.get_candles_range_aggregate_query(test_day_feb, test_day_feb+timedelta(days=1), symbol, '1h'), database=db)[symbol]
df_aug = influx_client_df.query(utils.influx.get_candles_range_aggregate_query(test_day_aug, test_day_aug+timedelta(days=1), symbol, '1h'), database=db)[symbol]
# df_oct = influx_client_df.query(utils.influx.get_candles_range_aggregate_query(test_day_oct, test_day_oct+timedelta(days=1), symbol, '1h'), database=db)[symbol]
# df_nov = influx_client_df.query(utils.influx.get_candles_range_aggregate_query(test_day_nov, test_day_nov+timedelta(days=1), symbol, '1h'), database=db)[symbol]

#%%
plt.close()
# tz = pytz.timezone('Asia/Tokyo')
# tz = pytz.timezone('Asia/Hong_Kong')
# tz = pytz.timezone('Australia/Sydney')
# tz = pytz.timezone('Europe/Berlin')
# tz = pytz.timezone('Europe/London')
# tz = pytz.timezone('America/Chicago')
tz = pytz.timezone('America/New_York')
# df = df_feb
# df = df_aug
# df = df_feb.tz_convert(tz)
df = df_aug.tz_convert(tz)

# Initialize the candlestick chart
fig, ax = mpf.plot(
  df,
  type="candle",
  style="charles",
  returnfig=True,
  title=f"{symbol}",
  ylabel="Price ($)",
  columns=utils.influx.MPF_COLUMN_MAPPING,
  volume=False,
  tight_layout=True,
  figsize=(24, 13),
)  # ax[0] is the main candlestick axis

main_ax = ax[0]  # Main chart axis
# volume_ax = ax[2] if len(ax) > 1 else None  # Volume axis (if present)

# Display box for showing OHLC
info_box = plt.figtext(0.15, 0.9, "", fontsize=10, bbox={"facecolor": "lightgrey", "alpha": 0.5, "pad": 4})


# Define a function to handle mouse hovering over the chart
def on_hover(event):
  """
    Display OHLC information when hovering over the candlestick plot.
    """
  if event.inaxes == main_ax:  # Check if the pointer is within the price axis
    # Find the closest candle to the x coordinate (event.xdata is the x-value in data coordinates)
    if event.xdata is None:  # Check for invalid event data
      return

    # Convert event.xdata to the closest integer index within the DataFrame
    x_index = int(round(event.xdata))

    # Ensure x_index is within valid bounds
    if 0 <= x_index < len(df):
      # Extract the OHLC data for the closest candle
      ohlc = df.iloc[x_index]
      ohlc_info = (
        f"Date: {df.index[x_index].strftime('%Y-%m-%d %H:%M')} \n"
        f"Open: {ohlc['o']} \n"
        f"High: {ohlc['h']} \n"
        f"Low: {ohlc['l']} \n"
        f"Close: {ohlc['h']} \n"
      )

      # Update the information box with OHLC data
      info_box.set_text(ohlc_info)
      plt.draw()


# Connect the hover event listener
fig.canvas.mpl_connect("motion_notify_event", on_hover)

# Show the plot
plt.show()

