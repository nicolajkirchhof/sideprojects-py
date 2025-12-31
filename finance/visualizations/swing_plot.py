# %%
import numpy as np

import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui

from finance import utils

#%%
symbol = 'SPY'
# spy_data = utils.swing_trading_data.SwingTradingData('SPY')
df =  utils.indicators.swing_indicators(utils.barchart_data.daily_w_volatility(symbol)) #spy_data.df_day

#%%

class DateAxis(pg.AxisItem):
  def __init__(self, dates, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.dates = dates

  def tickStrings(self, values, scale, spacing):
    return [self.dates[int(v)].strftime('%y-%m-%d') if 0 <= int(v) < len(self.dates) else "" for v in values]

class OHLCItem(pg.GraphicsObject):
  def __init__(self, data):
    pg.GraphicsObject.__init__(self)
    self.data = data # [time, open, high, low, close]
    self.generatePicture()

  def generatePicture(self):
    self.picture = QtGui.QPicture()
    p = QtGui.QPainter(self.picture)
    p.setPen(pg.mkPen('w'))
    for t, open, high, low, close in self.data:
      color = pg.mkPen('g') if close >= open else pg.mkPen('r')
      p.setPen(color)
      p.drawLine(QtCore.QPointF(t, low), QtCore.QPointF(t, high))
      # Open tick (left) and Close tick (right)
      p.drawLine(QtCore.QPointF(t-0.3, open), QtCore.QPointF(t, open))
      p.drawLine(QtCore.QPointF(t, close), QtCore.QPointF(t+0.3, close))
    p.end()

  def paint(self, p, *args):
    p.drawPicture(0, 0, self.picture)

  def boundingRect(self):
    return QtCore.QRectF(self.picture.boundingRect())

def plot_pyqtgraph(df):
    app = pg.mkQApp()
    win = pg.GraphicsLayoutWidget(show=True, title="SPY Multi-Pane Analysis")
    win.resize(1400, 1000)

    x_dates = df.index
    x_range = np.arange(len(df))

    # 1. Setup 3-Row Layout with Shared Date Axis
    p1 = win.addPlot(row=0, col=0, axisItems={'bottom': DateAxis(dates=x_dates, orientation='bottom')}, title="Price & EMAs")
    p2 = win.addPlot(row=1, col=0, axisItems={'bottom': DateAxis(dates=x_dates, orientation='bottom')})
    p3 = win.addPlot(row=2, col=0, axisItems={'bottom': DateAxis(dates=x_dates, orientation='bottom')})
    
    # Synchronize Panes
    plots = [p1, p2, p3]
    for p in plots:
        p.showGrid(x=True, y=True, alpha=0.3)
        # ALIGN AXES: Force the left axis to a fixed width so charts line up
        p.getAxis('left').setWidth(70)
        if p != p1:
            p.setXLink(p1)

    p2.setMaximumHeight(150)
    p3.setMaximumHeight(150)
    p3.setLabels(left='ATR %')

    # 2. Hover Text Overlay
    hover_label = pg.TextItem(anchor=(0, 0), color='#ccc', fill='#222b')
    p1.addItem(hover_label, ignoreBounds=True)

    # 3. Add Content to Panes
    p1.addItem(OHLCItem([(i, df.o.iloc[i], df.h.iloc[i], df.l.iloc[i], df.c.iloc[i]) for i in x_range]))
    
    ema_colors = {'ema10': '#1f77b4', 'ema20': '#ff7f0e', 'ema50': '#2ca02c', 'ema100': '#d62728', 'ema200': '#9467bd'}
    for col, color in ema_colors.items():
        if col in df.columns:
            p1.plot(x=x_range, y=df[col].values, pen=pg.mkPen(color, width=1.5), connect='finite')

    brushes = [pg.mkBrush('#26a69a' if c >= o else '#ef5350') for o, c in zip(df.o, df.c)]
    p2.addItem(pg.BarGraphItem(x=x_range, height=df.v.values, width=0.5, brushes=brushes, pen=None))

    atr_colors = {'atrp14': '#55f', 'atrp20': '#f5f'}
    for col, color in atr_colors.items():
        if col in df.columns:
            p3.plot(x=x_range, y=df[col].values, pen=pg.mkPen(color, width=1.5), connect='finite')

    # 4. VERTICAL INDICATORS (Crosshair) for all panes
    v_lines = []
    for p in plots:
        v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#666', width=1, style=QtCore.Qt.PenStyle.DashLine))
        p.addItem(v_line, ignoreBounds=True)
        v_lines.append(v_line)

    def update_hover(evt):
        pos = evt[0]
        # Check all plots to see which one the mouse is currently over
        for active_p in plots:
            if active_p.sceneBoundingRect().contains(pos):
                # Map coordinates using the ViewBox of the plot currently hovered
                mousePoint = active_p.vb.mapSceneToView(pos)
                idx = int(mousePoint.x() + 0.5)
                
                if 0 <= idx < len(df):
                    row = df.iloc[idx]
                    # Update all vertical indicators across all panes
                    for line in v_lines:
                        line.setPos(idx)
                    
                    # Update Hover Text (Format string logic)
                    txt = f"<span style='font-size: 11pt; color: white;'>{x_dates[idx].strftime('%Y-%m-%d')}</span><br>"
                    txt += f"O:{row.o:.2f} H:{row.h:.2f} L:{row.l:.2f} C:{row.c:.2f} V:{row.v:,.0f}<br>"
                    emas = " | ".join([f"<span style='color:{ema_colors[c]};'>{c.upper()}:{row[c]:.2f}</span>" for c in ema_colors if c in df.columns])
                    atrs = " | ".join([f"<span style='color:{atr_colors[c]};'>{c.upper()}:{row[c]:.2f}%</span>" for c in atr_colors if c in df.columns])
                    
                    hover_label.setHtml(txt + emas + "<br>" + atrs)
                    
                    # Ensure the hover label stays at the top-left of the main pane
                    vb_range = p1.vb.viewRange()
                    hover_label.setPos(vb_range[0][0], vb_range[1][1])
                break # Found the active plot, stop searching

    # The signal proxy on the scene catches movements across the entire GraphicsLayoutWidget
    proxy = pg.SignalProxy(p1.scene().sigMouseMoved, rateLimit=60, slot=update_hover)

    # 5. Multi-Pane Y-scaling
    def update_y_views():
        vr = p1.viewRange()[0]
        s, e = max(0, int(vr[0])), min(len(df), int(vr[1]))
        if s < e:
            chunk = df.iloc[s:e]
            p1.setYRange(chunk.l.min()*0.99, chunk.h.max()*1.01, padding=0)
            p2.setYRange(0, chunk.v.max()*1.1, padding=0)
            atr_cols = [c for c in ['atrp14', 'atrp20'] if c in df.columns]
            if atr_cols:
                p3.setYRange(chunk[atr_cols].min().min()*0.95, chunk[atr_cols].max().max()*1.05, padding=0)

    p1.sigXRangeChanged.connect(update_y_views)
    p1.setXRange(len(df)-250, len(df))
    update_y_views()

    pg.exec()
    return win, proxy


#%%
plot_pyqtgraph(df)
