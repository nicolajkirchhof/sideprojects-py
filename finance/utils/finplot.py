# import finplot as fplt
import pandas as pd

def create_interactive_plot(ax, symbol, interval, df):
  fplt.candlestick_ochl(df[['o', 'c', 'h', 'l']], ax=ax)
  hover_label = fplt.add_legend(interval, ax=ax)
  hover_label.opts['color']='#000'

  # #######################################################
  # ## update crosshair and legend when moving the mouse ##
  #
  def update_legend_text(x, y):
    print(interval)
    row = df.loc[pd.to_datetime(x, unit='ns', utc=True)]
    # format html with the candle and set legend
    fmt = '<span style="font-size:15px;color:#%s;background-color:#fff">%%.2f</span>' % ('0d0' if (row.o<row.c).all() else 'd00')
    rawtxt = '<span style="font-size:14px">%%s %%s</span> &nbsp; O %s C %s H %s L %s' % (fmt, fmt, fmt, fmt)
    values = [row.o, row.c, row.h, row.l]
    hover_label.setText(rawtxt % tuple([symbol, interval.upper()] + values))

  def update_crosshair_text(x, y, xtext, ytext):
    row =  df.iloc[x]
    ytext = f'{y:.2f} O {row.c:.2f} H {row.h:.2f} L {row.l:.2f} C {row.c:.2f}'
    return xtext, ytext

  fplt.set_mouse_callback(update_legend_text, ax=ax, when='hover')
  fplt.add_crosshair_info(update_crosshair_text, ax=ax)


# Function to save screenshot of the chart
def save_screenshot(filename:str):
  # Grab the Finplot window as a QPixmap
  pixmap = fplt.app.activeWindow().grab()
  # Save the pixmap to an image file (e.g., PNG)
  pixmap.save(filename)
  print(f"Screenshot saved as '{filename}'")
  fplt.close()
