import os
import re
from datetime import timedelta, datetime

import mplfinance as mpf
import numpy as np
import pandas as pd
from matplotlib import gridspec, pyplot as plt
from matplotlib.pyplot import tight_layout

from finance import utils
from finance.utils.trading_day_data import TradingDayData
import matplotlib.ticker as mticker


def daily_change_plot(day_data: TradingDayData, alines=None, title_add='', atr_vlines=dict(vlines=[], colors=[])):
  # |-------------------------|
  # |           5m            |
  # | ------------------------|
  # |   D   |       30m       |
  # | ------------------------|

  fig = mpf.figure(style='yahoo', figsize=(16,9), tight_layout=True)
  gs = gridspec.GridSpec(3, 2, height_ratios=[2, 0.25, 1], width_ratios=[1, 2])

  date_str = day_data.day_start.strftime('%Y-%m-%d')
  ax1 = fig.add_subplot(gs[0, :])
  ax2 = fig.add_subplot(gs[2, 0])
  ax3 = fig.add_subplot(gs[2, 1])
  ax4 = fig.add_subplot(gs[1, :])

  indicator_hlines = [day_data.cdl, day_data.cdh, day_data.cdc, day_data.pdc, day_data.pdh, day_data.pdl,
                      day_data.onh, day_data.onl, day_data.cwh, day_data.cwl, day_data.pwh, day_data.pwl]
  fig.suptitle(f'{day_data.symbol} {date_str} || O {day_data.cdo:.2f} H {day_data.cdh:.2f} C {day_data.cdc:.2f} L {day_data.cdl:.2f} || On: H {day_data.onh:.2f} L {day_data.onl:.2f} \n' +
               f'PD: H {day_data.pdh:.2f} C {day_data.pdc:.2f} L {day_data.pdl:.2f} || ' +
               f'CW: H {day_data.cwh:.2f} L {day_data.cwl:.2f} || PW: H {day_data.pwh:.2f} L {day_data.pwl:.2f} || {title_add}')

  hlines=dict(hlines=indicator_hlines, colors= ['deeppink']*3+['#bf42f5']*5+['#3179f5']*4, linewidths=[0.4]*3+[0.6]*3+[0.4]*6, linestyle=['--']*4+['-']*(len(indicator_hlines)-1))
  hlines_day=dict(hlines=indicator_hlines[3:], colors= ['#bf42f5']*5+['#3179f5']*4, linewidths=[0.6]*3+[0.4]*6, linestyle=['--']+['-']*(len(indicator_hlines)-1))
  vlines=dict(vlines=[day_data.day_open, day_data.day_close], colors= ['deeppink']*2, linewidths=[0.4], linestyle=['--'])

  ind_5m_ema20_plot = mpf.make_addplot(day_data.df_5m['20EMA'], ax=ax1, width=0.6, color="#FF9900", linestyle='--')
  ind_5m_ema240_plot = mpf.make_addplot(day_data.df_5m['200EMA'], ax=ax1, width=0.6, color='#0099FF', linestyle='--')
  ind_vwap3_plot = mpf.make_addplot(day_data.df_5m['VWAP3'], ax=ax1, width=2, color='turquoise')

  ind_30m_ema20_plot = mpf.make_addplot(day_data.df_30m['20EMA'], ax=ax3, width=0.6, color="#FF9900", linestyle='--')

  ind_day_ema20_plot = mpf.make_addplot(day_data.df_day['20EMA'], ax=ax2, width=0.6, color="#FF9900", linestyle='--')

  mpf.plot(day_data.df_5m, type='candle', ax=ax1, columns=utils.influx.MPF_COLUMN_MAPPING, xrotation=0, datetime_format='%H:%M', tight_layout=True,
           scale_width_adjustment=dict(candle=1.35), hlines=hlines, alines=alines, vlines=vlines, addplot=[ind_5m_ema20_plot, ind_5m_ema240_plot, ind_vwap3_plot])
  mpf.plot(day_data.df_5m, type='line', ax=ax4, columns=['lh']*5, xrotation=0, datetime_format='%H:%M', vlines=atr_vlines, tight_layout=True)

  mpf.plot(day_data.df_day, type='candle', ax=ax2, columns=utils.influx.MPF_COLUMN_MAPPING,  xrotation=0, datetime_format='%m-%d', tight_layout=True,
           hlines=hlines_day, warn_too_much_data=700, addplot=[ind_day_ema20_plot])
  mpf.plot(day_data.df_30m, type='candle', ax=ax3, columns=utils.influx.MPF_COLUMN_MAPPING, xrotation=0, datetime_format='%H:%M', tight_layout=True,
           scale_width_adjustment=dict(candle=1.35), hlines=hlines, addplot=[ind_30m_ema20_plot])


  # Use MaxNLocator to increase the number of ticks
  ax1.xaxis.set_major_locator(mticker.MaxNLocator(nbins=20))  # Increase number of ticks on x-axis
  ax4.xaxis.set_major_locator(mticker.MaxNLocator(nbins=20))  # Increase number of ticks on x-axis
  ax1.yaxis.set_major_locator(mticker.MaxNLocator(nbins=10))  # Increase number of ticks on y-axis
  plt.tight_layout(h_pad=0.1)

def last_date_from_files(directory):
  files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
  # Sort files by name in descending order
  files_sorted = sorted(files, reverse=True)
  first_file = files_sorted[0] if files_sorted else None
  if first_file is not None:
    # Define a regex pattern to extract the date (format: YYYY-MM-DD)
    date_pattern = r'\d{4}-\d{2}-\d{2}'  # Adjust the pattern to match your date format

    # Find the date in the filename
    match = re.search(date_pattern, first_file)
    if match:
      date_str = match.group()  # Extract the date string
      parsed_date = datetime.strptime(date_str, "%Y-%m-%d")  # Parse into a datetime object
      print(f"Date string: {date_str}")
      print(f"Parsed date: {parsed_date}")
      return parsed_date
    else:
      print("No date found in filename.")
  return None
