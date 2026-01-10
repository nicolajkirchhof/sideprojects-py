# %%
import numpy as np

import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui
from pyqtgraph.Qt import QtWidgets

from finance import utils

import pandas as pd

%load_ext autoreload
%autoreload 2


#%% # Global Plot Configurations
EMA_CONFIGS = {
  'ema10':  {'color': '#f5deb3', 'width': 1.0}, # Wheat (Lightest)
  'ema20':  {'color': '#e2b46d', 'width': 1.0},
  'ema50':  {'color': '#c68e17', 'width': 1.0},
  'ema100': {'color': '#8b5a2b', 'width': 1.0},
  'ema200': {'color': '#4b3621', 'width': 1.0},  # Darkest
  'vwap3': {'color': '#00bfff', 'width': 1.0}  # Darkest
}

ATR_CONFIGS = {
  'atrp9':  {'color': '#b0c4de', 'width': 1.0}, # Light Steel Blue
  'atrp14': {'color': '#4682b4', 'width': 1.0}, # Steel Blue
  'atrp20': {'color': '#000080', 'width': 1.5}  # Navy
}

AC_CONFIGS = {
  'ac100_lag_1':  {'color': '#e0ffff', 'width': 1.0}, # Light Cyan
  'ac100_lag_5':  {'color': '#00ced1', 'width': 1.0}, # Dark Turquoise
  'ac100_lag_10': {'color': '#00bfff', 'width': 1.0}, # Deep Sky Blue
  'ac100_lag_20': {'color': '#008080', 'width': 1.5}  # Teal
}

# New configuration for the Autocorrelation Regime pane
AC_REGIME_CONFIGS = {
    'ac_mom':  {'color': '#e1bee7', 'width': 1.0}, # Light Purple (Momentum)
    'ac_mr':   {'color': '#ba68c8', 'width': 1.5}, # Medium Purple (Mean Reversion)
    'ac_comp': {'color': '#4a148c', 'width': 2.2}  # Indigo/Darkest (Composite)
}

SLOPE_CONFIGS = {
  'ema10_slope':  {'color': '#ffccbc', 'width': 1.0}, # Deep Orange Light
  'ema20_slope':  {'color': '#ff8a65', 'width': 1.0},
  'ema50_slope':  {'color': '#ff5722', 'width': 1.0},
  'ema100_slope': {'color': '#e64a19', 'width': 1.0},
  'ema200_slope': {'color': '#bf360c', 'width': 2.0}  # Deep Orange Dark
}

HURST_CONFIGS = {
  'hurst50':  {'color': '#fff59d', 'width': 1.0}, # Light Yellow
  'hurst100': {'color': '#fbc02d', 'width': 1.0}  # Dimmed Gold/Brownish
}

HV_CONFIGS = {
  'hv9':  {'color': '#a5d6a7', 'width': 1.0}, # Light Green
  'hv14': {'color': '#66bb6a', 'width': 1.0}, # Medium Green
  'hv30': {'color': '#2e7d32', 'width': 1.5},  # Dark Green
  'iv':   {'color': '#ff00ff', 'width': 1.0}  # Magenta (Standout)
}

##%%

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

def plot_pyqtgraph(full_df, initial_max_date=None):
    app = pg.mkQApp()
    
    # Main Window Setup
    main_win = QtWidgets.QMainWindow()
    main_win.setWindowTitle(f"{full_df.symbol[0]} Multi-Pane Analysis")
    central_widget = QtWidgets.QWidget()
    main_win.setCentralWidget(central_widget)
    layout = QtWidgets.QVBoxLayout(central_widget)

    # 1. Date Selector Toolbar
    toolbar = QtWidgets.QHBoxLayout()
    
    years = [str(y) for y in sorted(full_df.index.year.unique(), reverse=True)]
    months = [f"{m:02d}" for m in range(1, 13)]
    days = [f"{d:02d}" for d in range(1, 32)]

    year_cb = QtWidgets.QComboBox(); year_cb.addItems(years)
    month_cb = QtWidgets.QComboBox(); month_cb.addItems(months)
    day_cb = QtWidgets.QComboBox(); day_cb.addItems(days)

    # Set initial values from initial_max_date or last date in df
    start_date = pd.to_datetime(initial_max_date) if initial_max_date else full_df.index[-1]
    year_cb.setCurrentText(str(start_date.year))
    month_cb.setCurrentText(f"{start_date.month:02d}")
    day_cb.setCurrentText(f"{start_date.day:02d}")

    toolbar.addWidget(QtWidgets.QLabel("Max Date Filter:"))
    toolbar.addWidget(year_cb); toolbar.addWidget(month_cb); toolbar.addWidget(day_cb)
    toolbar.addStretch()
    layout.addLayout(toolbar)

    # 2. Graphics Layout
    win = pg.GraphicsLayoutWidget()
    layout.addWidget(win)

    # Store proxy in a list or as a property to prevent garbage collection
    state = {'proxy': None}
    main_win.resize(1400, 1300)
    main_win.show()

    def update_plot():
        win.clear()
        target_str = f"{year_cb.currentText()}-{month_cb.currentText()}-{day_cb.currentText()}"
        df = full_df[full_df.index <= target_str]
        
        if df.empty: return

        x_dates = df.index
        x_range = np.arange(len(df))

        # --- Plot Setup ---
        p1 = win.addPlot(row=0, col=0, axisItems={'bottom': DateAxis(dates=x_dates, orientation='bottom')})
        p2 = win.addPlot(row=1, col=0, axisItems={'bottom': DateAxis(dates=x_dates, orientation='bottom')})
        p3 = win.addPlot(row=2, col=0, axisItems={'bottom': DateAxis(dates=x_dates, orientation='bottom')})
        p4 = win.addPlot(row=3, col=0, axisItems={'bottom': DateAxis(dates=x_dates, orientation='bottom')})
        p5 = win.addPlot(row=4, col=0, axisItems={'bottom': DateAxis(dates=x_dates, orientation='bottom')})
        p6 = win.addPlot(row=5, col=0, axisItems={'bottom': DateAxis(dates=x_dates, orientation='bottom')})
        p7 = win.addPlot(row=6, col=0, axisItems={'bottom': DateAxis(dates=x_dates, orientation='bottom')})
        p8 = win.addPlot(row=7, col=0, axisItems={'bottom': DateAxis(dates=x_dates, orientation='bottom')})

        plots = [p1, p2, p3, p4, p5, p6, p7, p8]
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
        p7.setLabels(left='AC Regimes')
        p7.setLabels(left='IV/HV')

        # 2. Hover Overlay - Darkened background
        hover_label = pg.TextItem(anchor=(0, 0), color='#ccc', fill='#000e')
        p1.addItem(hover_label, ignoreBounds=True)

        # --- Plotting ---
        # 3. Pane 1: Price
        p1.addItem(OHLCItem([(i, df.o.iloc[i], df.h.iloc[i], df.l.iloc[i], df.c.iloc[i]) for i in x_range]))
        for col, cfg in EMA_CONFIGS.items():
            if col in df.columns:
                p1.plot(x=x_range, y=df[col].values, pen=pg.mkPen(cfg['color'], width=cfg['width']), connect='finite')

        # 4. Pane 2: Volume
        brushes = [pg.mkBrush('#26a69a' if c >= o else '#ef5350') for o, c in zip(df.o, df.c)]
        p2.addItem(pg.BarGraphItem(x=x_range, height=df.v.values, width=0.5, brushes=brushes, pen=None))

        # 5. Pane 3: ATR (Blue Scale)
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
        for col, cfg in HURST_CONFIGS.items():
            if col in df.columns:
                p6.plot(x=x_range, y=df[col].values, pen=pg.mkPen(cfg['color'], width=cfg['width']), connect='finite')
        p6.addLine(y=0.5, pen=pg.mkPen('#666', style=QtCore.Qt.PenStyle.DashLine))

        # 10. Pane 7: AC Regimes (Purple Scale)
        for col, cfg in AC_REGIME_CONFIGS.items():
            if col in df.columns:
                p7.plot(x=x_range, y=df[col].values, pen=pg.mkPen(cfg['color'], width=cfg['width']), connect='finite')
        p7.addLine(y=0, pen=pg.mkPen('#666', width=1))

        # 11. Pane 8: Historical Volatility (Green Scale)
        for col, cfg in HV_CONFIGS.items():
          if col in df.columns:
            p8.plot(x=x_range, y=df[col].values, pen=pg.mkPen(cfg['color'], width=cfg['width']), connect='finite')

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
            h_line.hide()

        def update_hover(evt):
            pos = evt[0]
            for hl in h_lines: hl.hide()
            for i, active_p in enumerate(plots):
                if active_p.sceneBoundingRect().contains(pos):
                    mousePoint = active_p.vb.mapSceneToView(pos)
                    idx = int(mousePoint.x() + 0.5)

                    if 0 <= idx < len(df):
                        row = df.iloc[idx]

                        # 1. Update Vertical Indicators (All panes)
                        for line in v_lines:
                            line.setPos(idx)

                        # 2. Update Horizontal Indicator (Active pane only)
                        h_lines[i].setPos(mousePoint.y())
                        h_lines[i].show()

                        # Formatted Hover Text
                        # Adding %a for short weekday (e.g., Mon)
                        txt = f"<span style='font-size: 11pt; color: white; font-weight: bold;'>{x_dates[idx].strftime('%a %Y-%m-%d')}</span><br>"
                        txt += f"O:{row.o:.2f} H:{row.h:.2f} L:{row.l:.2f} C:{row.c:.2f} V:{row.v:,.0f}<br>"

                        emas = " | ".join([f"<span style='color:{EMA_CONFIGS[c]['color']};'>{c.upper()}:{row[c]:.2f}</span>" for c in EMA_CONFIGS if c in df.columns])
                        atrs = " | ".join([f"<span style='color:{ATR_CONFIGS[c]['color']};'>{c.upper()}:{row[c]:.2f}%</span>" for c in ATR_CONFIGS if c in df.columns])
                        hursts = " | ".join([f"<span style='color:{HURST_CONFIGS[c]['color']};'>{c}:{row[c]:.2f}</span>" for c in HURST_CONFIGS if c in df.columns])
                        slopes = " | ".join([f"<span style='color:{SLOPE_CONFIGS[c]['color']};'>{c.split('_')[0]}:{row[c]:.2f}</span>" for c in SLOPE_CONFIGS if c in df.columns])
                        acs = " | ".join([f"<span style='color:{AC_CONFIGS[c]['color']};'>ac{c.split('_')[-1]}:{row[c]:.2f}</span>" for c in AC_CONFIGS if c in df.columns])
                        ac_regs = " | ".join([f"<span style='color:{AC_REGIME_CONFIGS[c]['color']};'>{c.split('_')[-1]}:{row[c]:.2f}</span>" for c in AC_REGIME_CONFIGS if c in df.columns])
                        hvs = " | ".join([f"<span style='color:{HV_CONFIGS[c]['color']};'>{c.upper()}:{row[c]:.2f}</span>" for c in HV_CONFIGS if c in df.columns])

                        hover_label.setHtml(txt + emas + "<br>" + atrs + "<br>" + hursts + "<br>" + slopes + "<br>" + acs + "<br>" + ac_regs + "<br>" + hvs)

                        vb_range = p1.vb.viewRange()
                        hover_label.setPos(vb_range[0][0] + (vb_range[0][1]-vb_range[0][0])*0.01,
                                           vb_range[1][1] - (vb_range[1][1]-vb_range[1][0])*0.01)


        # IMPORTANT: Create a new proxy for the new scene and store it to keep it alive
        state['proxy'] = pg.SignalProxy(p1.scene().sigMouseMoved, rateLimit=60, slot=update_hover)

        # 11. Unified Scaling
        def update_y_views():
            vr = p1.viewRange()[0]
            s, e = max(0, int(vr[0])), min(len(df), int(vr[1]))
            if s < e:
                chunk = df.iloc[s:e]
                p1.setYRange(chunk.l.min()*0.99, chunk.h.max()*1.01, padding=0)
                p2.setYRange(0, chunk.v.max()*1.1, padding=0)

                # Scale Indicator Panes
                scale_map = [
                    (p3, list(ATR_CONFIGS.keys())), (p4, list(AC_CONFIGS.keys())),
                    (p5, list(SLOPE_CONFIGS.keys())), (p6, ['hurst']),
                    (p7, list(AC_REGIME_CONFIGS.keys())), (p8, list(HV_CONFIGS.keys()))]

                for p, cols in scale_map:
                    valid_cols = [c for c in cols if c in df.columns]
                    if valid_cols:
                        p.setYRange(chunk[valid_cols].min().min()*0.95, chunk[valid_cols].max().max()*1.05, padding=0)

        p1.sigXRangeChanged.connect(update_y_views)
        p1.setXRange(max(0, len(df)-250), len(df))
        update_y_views()

    # Connect signals and run initial draw
    for cb in [year_cb, month_cb, day_cb]:
        cb.currentIndexChanged.connect(update_plot)
    
    update_plot()
    main_win.show()
    pg.exec()

# %%
symbol = 'SPY'
df = utils.indicators.swing_indicators(utils.barchart_data.daily_w_volatility(symbol))

# Start with a specific date
plot_pyqtgraph(df, initial_max_date='2022-01-24')
