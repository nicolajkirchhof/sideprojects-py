# %%
import numpy as np

import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui
from pyqtgraph.Qt import QtWidgets
import pyqtgraph.exporters

from finance import utils

import pandas as pd

%load_ext autoreload
%autoreload 2


#%% # Global Plot Configurations
EMA_CONFIGS = {
  'ema10':  {'color': '#9b7f6c', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine}, # Wheat (Lightest)
  'ema20':  {'color': '#b26529', 'width': 2.0, 'style': QtCore.Qt.PenStyle.SolidLine},
  'ema50':  {'color': '#7b4326', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},
  'ema100': {'color': '#703b24', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},
  'ema200': {'color': '#5a2c27', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},  # Darkest
  'vwap3': {'color': '#47a3b9', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine}  # Darkest
}

ATR_CONFIGS = {
  'atrp1': {'color': '#f5a1df', 'width': 1.0, 'style': QtCore.Qt.PenStyle.DashLine}, # Steel Blue
  'atrp9':  {'color': '#f81cfc', 'width': 1.0, 'style': QtCore.Qt.PenStyle.DashLine}, # Light Steel Blue
  'atrp20': {'color': '#b72494', 'width': 2.0, 'style': QtCore.Qt.PenStyle.SolidLine},  # Navy
  'atrp50': {'color': '#6b1255', 'width': 1.0, 'style': QtCore.Qt.PenStyle.DashLine}  # Navy
}

STD_CONFIGS = {
  'std' : {'color':  '#ba68c8', 'width': 1.5, 'style': QtCore.Qt.PenStyle.SolidLine}  # Blue
}

AC_CONFIGS = {
  'ac100_lag_1':  {'color': '#e0ffff', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine}, # Light Cyan
  'ac100_lag_5':  {'color': '#00ced1', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine}, # Dark Turquoise
  'ac100_lag_10': {'color': '#00bfff', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine}, # Deep Sky Blue
  'ac100_lag_20': {'color': '#008080', 'width': 1.5, 'style': QtCore.Qt.PenStyle.SolidLine}  # Teal
}

# New configuration for the Autocorrelation Regime pane
AC_REGIME_CONFIGS = {
    'ac_mom':  {'color': '#e1bee7', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine}, # Light Purple (Momentum)
    'ac_mr':   {'color': '#ba68c8', 'width': 1.5, 'style': QtCore.Qt.PenStyle.SolidLine}, # Medium Purple (Mean Reversion)
    'ac_comp': {'color': '#4a148c', 'width': 2.2, 'style': QtCore.Qt.PenStyle.SolidLine}  # Indigo/Darkest (Composite)
}

SLOPE_CONFIGS = {
  'ema10_slope':  {'color': '#ffccbc', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine}, # Deep Orange Light
  'ema20_slope':  {'color': '#ff8a65', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},
  'ema50_slope':  {'color': '#ff5722', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},
  'ema100_slope': {'color': '#e64a19', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},
  'ema200_slope': {'color': '#bf360c', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine}  # Deep Orange Dark
}

VOL_CONFIGS = {
  'v':  {'color': '#49bdd9', 'width': 1.0, 'style': QtCore.Qt.PenStyle.DashLine}, # Deep Orange Light
  'v9':  {'color': '#fcec98', 'width': 1.0, 'style': QtCore.Qt.PenStyle.DashLine},
  'v20':  {'color': '#f3cb21', 'width': 1.5, 'style': QtCore.Qt.PenStyle.SolidLine},
  'v50': {'color': '#dab312', 'width': 1.5, 'style': QtCore.Qt.PenStyle.DashLine}
}

DIST_CONFIGS = {
  'ema10_dist':  {'color': '#9b7f6c', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine}, # Deep Orange Light
  'ema20_dist':  {'color': '#b26529', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},
  'ema50_dist':  {'color': '#7b4326', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},
  'ema100_dist': {'color': '#703b24', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},
  'ema200_dist': {'color': '#5a2c27', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine}  # Deep Orange Dark
}

HURST_CONFIGS = {
  'hurst50':  {'color': '#fff59d', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine}, # Light Yellow
  'hurst100': {'color': '#fbc02d', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine}  # Dimmed Gold/Brownish
}

HV_CONFIGS = {
  'hv9':  {'color': '#b7a3db', 'width': 1.0, 'style': QtCore.Qt.PenStyle.DashLine}, # Light Green
  'hv20': {'color': '#6539b4', 'width': 2.0, 'style': QtCore.Qt.PenStyle.SolidLine}, # Medium Green
  'hv50': {'color': '#583098', 'width': 1.0, 'style': QtCore.Qt.PenStyle.DashLine},  # Dark Green
  'iv':   {'color': '#49bcd8', 'width': 2.0, 'style': QtCore.Qt.PenStyle.SolidLine}  # Magenta (Standout)
}

IVPCT_CONFIGS = {
  'iv_pct':  {'color': '#b72494', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine}, # Light Green
}

# TTM Squeeze Colors
TTM_COLORS = {
    'pos_up': '#00ff00',   # Bright Green (Bullish rising)
    'pos_down': '#006400', # Dark Green (Bullish falling)
    'neg_down': '#ff0000', # Red (Bearish falling)
    'neg_up': '#8b0000',   # Dark Red (Bearish rising)
    'sq_on': '#ff0000',    # Red dot (Squeeze ON)
    'sq_off': '#00ff00'    # Green dot (Squeeze OFF)
}

##%%

class DateAxis(pg.AxisItem):
  def __init__(self, dates, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.dates = dates

  def tickStrings(self, values, scale, spacing):
    return [self.dates[int(v)].strftime('%y-%m-%d') if 0 <= int(v) < len(self.dates) else "" for v in values]

  def tickValues(self, minVal, maxVal, size):
    # Increase the number of ticks by reducing the required spacing
    return super().tickValues(minVal, maxVal, size * 0.5)

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

class TTMSqueezeItem(pg.GraphicsObject):
    def __init__(self, data):
        pg.GraphicsObject.__init__(self)
        self.data = data # list of (x, mom, squeeze_on)
        self.generatePicture()

    def generatePicture(self):
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        
        # Determine a dynamic radius based on the momentum range to keep dots visible
        mom_values = [d[1] for d in self.data if not np.isnan(d[1])]
        max_abs_mom = max(abs(min(mom_values)), abs(max(mom_values))) if mom_values else 1.0
        dot_radius_y = max_abs_mom * 0.05  # 5% of the max range
        dot_radius_x = 0.2                 # Fixed width in x-units (bars)

        for i in range(len(self.data)):
            x, mom, sq_on = self.data[i]
            if np.isnan(mom): continue
            
            prev_mom = self.data[i-1][1] if i > 0 else mom
            
            # 1. Determine Histogram Color
            if mom >= 0:
                color = TTM_COLORS['pos_up'] if mom >= prev_mom else TTM_COLORS['pos_down']
            else:
                color = TTM_COLORS['neg_down'] if mom <= prev_mom else TTM_COLORS['neg_up']
            
            p.setPen(pg.mkPen(None))
            p.setBrush(pg.mkBrush(color))
            # drawRect(x, y, w, h) -> y is top-left, so for positive mom, y=0 is bottom
            # For negative mom, y=0 is top.
            p.drawRect(QtCore.QRectF(x - 0.4, 0, 0.8, -mom)) # Negating mom handles the y-direction correctly
            
            # 2. Draw Squeeze Dot
            dot_color = TTM_COLORS['sq_on'] if sq_on else TTM_COLORS['sq_off']
            p.setPen(pg.mkPen(None))
            p.setBrush(pg.mkBrush(dot_color))
            # Use specific X and Y radii to account for axis scaling
            p.drawEllipse(QtCore.QPointF(x, 0), dot_radius_x, dot_radius_y)
            
        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())

def plot_pyqtgraph(full_df, initial_max_date=None, export_path=None, vlines=None, display_range=250, 
                   export_width=1920, export_height=1080):
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
    state = {'proxy': None, 'df': None, 'x_range': None, 'x_dates': None}
    p1 = p3 = p4 = p5 = p6 = p7 = p8 = p1_vol = None
    
    # Resize window to match export proportions initially to help layout calculation
    main_win.resize(export_width, export_height)
    main_win.show()
    if export_path:
        main_win.hide()

    def update_y_views():
        if state['df'] is None or p1 is None: return
        
        vr = p1.viewRange()[0]
        s, e = max(0, int(vr[0])), min(len(state['df']), int(vr[1]))
        if s < e:
            chunk = state['df'].iloc[s:e]
            p1.setYRange(chunk.l.min()*0.99, chunk.h.max()*1.01, padding=0)
            
            # Scale Indicator Panes
            scale_map = [
                (p7, list(VOL_CONFIGS.keys())), (p3, list(DIST_CONFIGS.keys())), 
                (p4, list(ATR_CONFIGS.keys())), (p5, list(HV_CONFIGS.keys())), 
                (p6, list(IVPCT_CONFIGS.keys())), (p8, ['ttm_mom'])]

            for p, cols in scale_map:
                valid_cols = [c for c in cols if c in state['df'].columns]
                if valid_cols:
                    chunk_data = chunk[valid_cols]
                    if p == p7: # Scale Volume MAs to thousands
                        chunk_data = chunk_data / 1000
                    p.setYRange(chunk_data.min().min()*1.1, chunk_data.max().max()*1.1, padding=0)

    def update_plot():
        nonlocal p1, p3, p4, p5, p6, p7, p8
        win.clear()
        target_str = f"{year_cb.currentText()}-{month_cb.currentText()}-{day_cb.currentText()}"
        df = full_df[full_df.index <= target_str]
        if df.empty: return
        
        state['df'] = df
        state['x_dates'] = df.index
        state['x_range'] = np.arange(len(df))
        x_range = state['x_range']

        # --- Plot Setup ---
        p1 = win.addPlot(row=0, col=0)
        
        # Indicator Panes - Reordered: Vol MA is now first (row 1)
        p7 = win.addPlot(row=1, col=0)
        p3 = win.addPlot(row=2, col=0)
        p4 = win.addPlot(row=3, col=0)
        p5 = win.addPlot(row=4, col=0)
        p6 = win.addPlot(row=5, col=0)
        # Bottom pane gets the DateAxis
        p8 = win.addPlot(row=6, col=0, axisItems={'bottom': DateAxis(dates=state['x_dates'], orientation='bottom')})

        # Apply proportional heights
        win.ci.layout.setRowStretchFactor(0, 50) # Main Chart
        win.ci.layout.setRowStretchFactor(1, 8)  # p7 (Vol MA)
        win.ci.layout.setRowStretchFactor(2, 2)  # p3 (EMA Dist)
        win.ci.layout.setRowStretchFactor(3, 15) # p4 (ATR)
        win.ci.layout.setRowStretchFactor(4, 15) # p5 (IV/HV)
        win.ci.layout.setRowStretchFactor(5, 2)  # p6 (IVPct)
        win.ci.layout.setRowStretchFactor(6, 8)  # p8 (TTM Squeeze)

        plots = [p1, p7, p3, p4, p5, p6, p8]
        for p in plots:
            p.showGrid(x=True, y=True, alpha=0.3)
            p.getAxis('left').setWidth(70)
            p.getAxis('right').setWidth(70)
            p.showAxis('right')
            if p != p8:
                p.getAxis('bottom').hide() # Hide x-axis for all but bottom pane
            if p != p1:
                p.setXLink(p1)
                p.setMaximumHeight(16777215)

        p7.setLabels(left='Vol MA')
        p3.setLabels(left='EMA Dist')
        p4.setLabels(left='ATR %')
        p5.setLabels(left='IV/HV')
        p6.setLabels(left='IVPct')
        p8.setLabels(left='TTM Squeeze')

        hover_label = pg.TextItem(anchor=(0, 0), color='#ccc', fill='#000e')
        p1.addItem(hover_label, ignoreBounds=True)

        # --- Plotting Content ---
        p1.addItem(OHLCItem([(i, df.o.iloc[i], df.h.iloc[i], df.l.iloc[i], df.c.iloc[i]) for i in x_range]))
        for col, cfg in EMA_CONFIGS.items():
            if col in df.columns:
                p1.plot(x=x_range, y=df[col].values, pen=pg.mkPen(cfg['color'], width=cfg['width'], style=cfg.get('style', QtCore.Qt.PenStyle.SolidLine)), connect='finite')

        # Indicators
        for col, cfg in VOL_CONFIGS.items():
            if col in df.columns:
                p7.plot(x=x_range, y=df[col].values / 1000, pen=pg.mkPen(cfg['color'], width=cfg['width'], style=cfg.get('style', QtCore.Qt.PenStyle.SolidLine)), connect='finite')
        p7.addLine(y=0, pen=pg.mkPen('#666', width=1))

        for col, cfg in DIST_CONFIGS.items():
            if col in df.columns:
                p3.plot(x=x_range, y=df[col].values, pen=pg.mkPen(cfg['color'], width=cfg['width'], style=cfg.get('style', QtCore.Qt.PenStyle.SolidLine)), connect='finite')
        p3.addLine(y=1.2, pen=pg.mkPen('#666', width=1))

        for col, cfg in ATR_CONFIGS.items():
            if col in df.columns:
                p4.plot(x=x_range, y=df[col].values, pen=pg.mkPen(cfg['color'], width=cfg['width'], style=cfg.get('style', QtCore.Qt.PenStyle.SolidLine)), connect='finite')
        p4.addLine(y=0, pen=pg.mkPen('#666', width=1))

        for col, cfg in HV_CONFIGS.items():
            if col in df.columns:
                p5.plot(x=x_range, y=df[col].values, pen=pg.mkPen(cfg['color'], width=cfg['width'], style=cfg.get('style', QtCore.Qt.PenStyle.SolidLine)), connect='finite')
        p5.addLine(y=0, pen=pg.mkPen('#666', width=1))

        for col, cfg in IVPCT_CONFIGS.items():
            if col in df.columns:
                p6.plot(x=x_range, y=df[col].values, pen=pg.mkPen(cfg['color'], width=cfg['width'], style=cfg.get('style', QtCore.Qt.PenStyle.SolidLine)), connect='finite')
        p6.addLine(y=0.5, pen=pg.mkPen('#666', style=QtCore.Qt.PenStyle.DashLine))

        if 'ttm_mom' in df.columns and 'squeeze_on' in df.columns:
            p8.addItem(TTMSqueezeItem([(i, df.ttm_mom.iloc[i], df.squeeze_on.iloc[i]) for i in x_range]))
        p8.addLine(y=0, pen=pg.mkPen('#666', width=1))

        # --- Vertical Marker Lines ---
        if vlines:
            for v_date in vlines:
                v_dt = pd.to_datetime(v_date)
                if v_dt in df.index:
                    idx = df.index.get_loc(v_dt)
                    for p in plots:
                        marker_line = pg.InfiniteLine(pos=idx, angle=90, pen=pg.mkPen('darkviolet', width=0.8, style=QtCore.Qt.PenStyle.DashLine))
                        p.addItem(marker_line)

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
                        for line in v_lines: line.setPos(idx)
                        h_lines[i].setPos(mousePoint.y())
                        h_lines[i].show()

                        # Formatted Hover Text
                        txt = f"<span style='font-size: 11pt; color: white; font-weight: bold;'>{state['x_dates'][idx].strftime('%a %Y-%m-%d')}</span><br>"
                        txt += f"O:{row.o:.2f} H:{row.h:.2f} L:{row.l:.2f} C:{row.c:.2f} V:{row.v/1000:,.0f}k<br>"
                        dists = " | ".join([f"<span style='color:{DIST_CONFIGS[c]['color']};'>{c.split('_')[0]}:{row[c]:.2f}</span>" for c in DIST_CONFIGS if c in df.columns])
                        emas = " | ".join([f"<span style='color:{EMA_CONFIGS[c]['color']};'>{c.upper()}:{row[c]:.2f}</span>" for c in EMA_CONFIGS if c in df.columns])
                        atrs = " | ".join([f"<span style='color:{ATR_CONFIGS[c]['color']};'>{c.upper()}:{row[c]:.2f}%</span>" for c in ATR_CONFIGS if c in df.columns])
                        hvs = " | ".join([f"<span style='color:{HV_CONFIGS[c]['color']};'>{c.upper()}:{row[c]:.2f}</span>" for c in HV_CONFIGS if c in df.columns])
                        ivpct = " | ".join([f"<span style='color:{IVPCT_CONFIGS[c]['color']};'>{c}:{row[c]:.2f}</span>" for c in IVPCT_CONFIGS if c in df.columns])
                        vols = " | ".join([f"<span style='color:{VOL_CONFIGS[c]['color']};'>{c.upper()}:{row[c]/1000:.2f}k</span>" for c in VOL_CONFIGS if c in df.columns])

                        hover_label.setHtml(txt + dists + "<br>" + atrs + "<br>" + hvs + "<br>" + ivpct + "<br>" + vols )
                        vb_range = p1.vb.viewRange()
                        hover_label.setPos(vb_range[0][0] + (vb_range[0][1]-vb_range[0][0])*0.01,
                                           vb_range[1][1] - (vb_range[1][1]-vb_range[1][0])*0.01)

        state['proxy'] = pg.SignalProxy(p1.scene().sigMouseMoved, rateLimit=60, slot=update_hover)
        p1.sigXRangeChanged.connect(update_y_views)
        
        # Use the display_range parameter to set the initial view
        p1.setXRange(max(0, len(df) - display_range), len(df))
        update_y_views()

    # Connect signals and run initial draw
    for cb in [year_cb, month_cb, day_cb]:
        cb.currentIndexChanged.connect(update_plot)
    
    update_plot()
    
    if export_path:
        win.ci.layout.activate()
        app.processEvents()
        update_y_views()
        app.processEvents()
    
        exporter = pg.exporters.ImageExporter(win.scene())
        exporter.parameters()['width'] = export_width
        exporter.parameters()['height'] = export_height
        exporter.export(export_path)
        main_win.close()
    else:
        pg.exec()
# %%
symbol = 'SPY'
df = utils.indicators.swing_indicators(utils.barchart_data.daily_w_volatility(symbol))

#%%
# Example with vertical lines
plot_pyqtgraph(df,display_range=50,  vlines=['2025-12-01'], export_path=f'swing_{symbol}.png', export_width=1920, export_height=1080)

plot_pyqtgraph(df)

utils.plots.plot_pyqtgraph(df)
