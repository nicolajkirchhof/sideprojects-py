# %%
import numpy as np

import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui

from finance import utils

#%%
symbol = 'SPY'
# spy_data = utils.swing_trading_data.SwingTradingData('SPY')
df =  utils.indicators.swing_indicators(utils.barchart_data.daily_w_volatility(symbol)) #spy_data.df_day

#%% # Global Plot Configurations
EMA_CONFIGS = {
  'ema10':  {'color': '#f5deb3', 'width': 1.0}, # Wheat (Lightest)
  'ema20':  {'color': '#e2b46d', 'width': 1.2},
  'ema50':  {'color': '#c68e17', 'width': 1.5},
  'ema100': {'color': '#8b5a2b', 'width': 1.8},
  'ema200': {'color': '#4b3621', 'width': 2.2}  # Darkest
}

ATR_CONFIGS = {
  'atrp9':  {'color': '#b0c4de', 'width': 1.0}, # Light Steel Blue
  'atrp14': {'color': '#4682b4', 'width': 1.5}, # Steel Blue
  'atrp20': {'color': '#000080', 'width': 2.0}  # Navy
}

AC_CONFIGS = {
  'ac_lag_1':  {'color': '#e0ffff', 'width': 1.0}, # Light Cyan
  'ac_lag_5':  {'color': '#00ced1', 'width': 1.5}, # Dark Turquoise
  'ac_lag_21': {'color': '#008080', 'width': 2.0}  # Teal
}

SLOPE_CONFIGS = {
  'ema10_slope':  {'color': '#ffccbc', 'width': 1.0}, # Deep Orange Light
  'ema20_slope':  {'color': '#ff8a65', 'width': 1.2},
  'ema50_slope':  {'color': '#ff5722', 'width': 1.5},
  'ema100_slope': {'color': '#e64a19', 'width': 1.8},
  'ema200_slope': {'color': '#bf360c', 'width': 2.2}  # Deep Orange Dark
}

HURST_COLOR = '#ffeb3b' # Dimmed Yellow for Hurst

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
    win = pg.GraphicsLayoutWidget(show=True, title=f"{df.symbol[0]} Multi-Pane Indicator Analysis")
    win.resize(1400, 1200)

    x_dates = df.index
    x_range = np.arange(len(df))

    # 1. Setup 4-Row Layout
    p1 = win.addPlot(row=0, col=0, axisItems={'bottom': DateAxis(dates=x_dates, orientation='bottom')})
    p2 = win.addPlot(row=1, col=0, axisItems={'bottom': DateAxis(dates=x_dates, orientation='bottom')})
    p3 = win.addPlot(row=2, col=0, axisItems={'bottom': DateAxis(dates=x_dates, orientation='bottom')})
    p4 = win.addPlot(row=3, col=0, axisItems={'bottom': DateAxis(dates=x_dates, orientation='bottom')})
    p5 = win.addPlot(row=4, col=0, axisItems={'bottom': DateAxis(dates=x_dates, orientation='bottom')})
    p6 = win.addPlot(row=5, col=0, axisItems={'bottom': DateAxis(dates=x_dates, orientation='bottom')})

    plots = [p1, p2, p3, p4, p5, p6]
    for p in plots:
        p.showGrid(x=True, y=True, alpha=0.3)
        p.getAxis('left').setWidth(70)
        if p != p1:
            p.setXLink(p1)
            p.setMaximumHeight(130)

    p3.setLabels(left='ATR %')
    p4.setLabels(left='AutoCorr')
    p5.setLabels(left='EMA Slope')
    p6.setLabels(left='Hurst')

    # 2. Hover Overlay - Darkened background
    hover_label = pg.TextItem(anchor=(0, 0), color='#ccc', fill='#000e')
    p1.addItem(hover_label, ignoreBounds=True)

    # 3. Pane 1: Price
    p1.addItem(OHLCItem([(i, df.o.iloc[i], df.h.iloc[i], df.l.iloc[i], df.c.iloc[i]) for i in x_range]))
    # Logic: Shorter = Lighter/Thinner, Longer = Darker/Thicker
    for col, cfg in EMA_CONFIGS.items():
      if col in df.columns:
        p1.plot(x=x_range, y=df[col].values, pen=pg.mkPen(cfg['color'], width=cfg['width']), connect='finite')

    # 4. Pane 2: Volume
    brushes = [pg.mkBrush('#26a69a' if c >= o else '#ef5350') for o, c in zip(df.o, df.c)]
    p2.addItem(pg.BarGraphItem(x=x_range, height=df.v.values, width=0.5, brushes=brushes, pen=None))

    # 5. Pane 3: ATRP (Blue Scale)
    for col, cfg in ATR_CONFIGS.items():
      if col in df.columns:
        p3.plot(x=x_range, y=df[col].values, pen=pg.mkPen(cfg['color'], width=cfg['width']), connect='finite')
    p3.addLine(y=1.2, pen=pg.mkPen('#666', width=1))

    # 6. Pane 4: Autocorrelation Lags (Teal/Cyan Scale)
    for col, cfg in AC_CONFIGS.items():
        if col in df.columns:
            p4.plot(x=x_range, y=df[col].values, pen=pg.mkPen(cfg['color'], width=cfg['width']), connect='finite')
    p4.addLine(y=0, pen=pg.mkPen('#666', width=1)) 

    # 7. Pane 5: EMA Slopes (Orange/Red Scale)
    for col, cfg in SLOPE_CONFIGS.items():
        if col in df.columns:
            p5.plot(x=x_range, y=df[col].values, pen=pg.mkPen(cfg['color'], width=cfg['width']), connect='finite')
    p5.addLine(y=0, pen=pg.mkPen('#666', width=1))

    # 8. Pane 6: Hurst Exponent
    if 'hurst' in df.columns:
      p6.plot(x=x_range, y=df['hurst'].values, pen=pg.mkPen(HURST_COLOR, width=1.5), connect='finite')
      p6.addLine(y=0.5, pen=pg.mkPen('#666', style=QtCore.Qt.PenStyle.DashLine)) # Baseline

    # 9. CROSSHAIRS & Interaction Logic
    v_lines = []
    h_lines = []
    for p in plots:
        v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#666', style=QtCore.Qt.PenStyle.DashLine))
        h_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('#666', style=QtCore.Qt.PenStyle.DashLine))
        p.addItem(v_line, ignoreBounds=True)
        p.addItem(h_line, ignoreBounds=True)
        v_lines.append(v_line)
        h_lines.append(h_line)
        # Hide h_lines initially until mouse movement
        h_line.hide()

    def update_hover(evt):
        pos = evt[0]
        # Hide all horizontal lines first; we only show the one in the active pane
        for hl in h_lines: hl.hide()

        for i, active_p in enumerate(plots):
            if active_p.sceneBoundingRect().contains(pos):
                mousePoint = active_p.vb.mapSceneToView(pos)
                idx = int(mousePoint.x() + 0.5)

                if 0 <= idx < len(df):
                    row = df.iloc[idx]
                    for line in v_lines: line.setPos(idx)
                    
                    # 1. Update Vertical Indicators (All panes)
                    for line in v_lines:
                        line.setPos(idx)

                    # 2. Update Horizontal Indicator (Active pane only)
                    h_lines[i].setPos(mousePoint.y())
                    h_lines[i].show()
                    # Update Hover Text with matched colors
                    txt = f"<span style='font-size: 11pt; color: white;'>{x_dates[idx].strftime('%Y-%m-%d')}</span><br>"
                    txt += f"O:{row.o:.2f} H:{row.h:.2f} L:{row.l:.2f} C:{row.c:.2f} V:{row.v:,.0f}<br>"
                    
                    emas = " | ".join([f"<span style='color:{EMA_CONFIGS[c]['color']};'>{c.upper()}:{row[c]:.2f}</span>" for c in EMA_CONFIGS if c in df.columns])
                    atrs = " | ".join([f"<span style='color:{ATR_CONFIGS[c]['color']};'>{c.upper()}:{row[c]:.2f}%</span>" for c in ATR_CONFIGS if c in df.columns])
                    acs = " | ".join([f"<span style='color:{AC_CONFIGS[c]['color']};'>{c.split('_')[-1]}:{row[c]:.2f}</span>" for c in AC_CONFIGS if c in df.columns])
                    slopes = " | ".join([f"<span style='color:{SLOPE_CONFIGS[c]['color']};'>{c.split('_')[0]}:{row[c]:.2f}</span>" for c in SLOPE_CONFIGS if c in df.columns])
                    hurst_val = f" | <span style='color:{HURST_COLOR};'>Hurst: {row.hurst:.2f}</span>" if 'hurst' in df.columns else ""
                    
                    hover_label.setHtml(txt + emas + "<br>" + atrs + "<br>" + slopes + "<br>" + acs + "<br>" + hurst_val)
                    vb_range = p1.vb.viewRange()

                    # Offset slightly from the top-left corner of the viewport
                    hover_label.setPos(vb_range[0][0] + (vb_range[0][1]-vb_range[0][0])*0.01,
                                       vb_range[1][1] - (vb_range[1][1]-vb_range[1][0])*0.01)

    proxy = pg.SignalProxy(p1.scene().sigMouseMoved, rateLimit=60, slot=update_hover)

    # 9. Unified Scaling
    def update_y_views():
        vr = p1.viewRange()[0]
        s, e = max(0, int(vr[0])), min(len(df), int(vr[1]))
        if s < e:
            chunk = df.iloc[s:e]
            p1.setYRange(chunk.l.min()*0.99, chunk.h.max()*1.01, padding=0)
            p2.setYRange(0, chunk.v.max()*1.1, padding=0)
            
            # Scale Indicator Panes
            scale_map = [
                (p3, ['atrp9', 'atrp14', 'atrp20']), 
                (p4, ['ac_lag_1', 'ac_lag_5', 'ac_lag_21']),
                (p5, ['ema10_slope', 'ema20_slope', 'ema50_slope', 'ema100_slope', 'ema200_slope'])
            ]
            for p, cols in scale_map:
                valid_cols = [c for c in cols if c in df.columns]
                if valid_cols:
                    p.setYRange(chunk[valid_cols].min().min()*0.95, chunk[valid_cols].max().max()*1.05, padding=0)

    p1.sigXRangeChanged.connect(update_y_views)
    p1.setXRange(len(df)-250, len(df))
    update_y_views()

    pg.exec()
    return win, proxy


#%%
plot_pyqtgraph(df)
