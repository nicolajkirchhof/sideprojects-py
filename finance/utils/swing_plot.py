import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui
from pyqtgraph.Qt import QtWidgets
import pyqtgraph.exporters
from datetime import datetime
import sys

# %% Global Plot Configurations
EMA_CONFIGS = {
    'ema5': {'color': '#9b7f6c', 'width': 1.0, 'style': QtCore.Qt.PenStyle.DashLine},  # Wheat (Lightest)
    'ema10': {'color': '#9b7f6c', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},  # Wheat (Lightest)
    'ema20': {'color': '#b26529', 'width': 2.0, 'style': QtCore.Qt.PenStyle.SolidLine},
    'ema50': {'color': '#7b4326', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},
    'ema100': {'color': '#703b24', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},
    'ema200': {'color': '#5a2c27', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},  # Darkest
    'vwap3': {'color': '#47a3b9', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine}  # Darkest
}

ATR_CONFIGS = {
    'atrp1': {'color': '#f5a1df', 'width': 1.0, 'style': QtCore.Qt.PenStyle.DashLine},  # Steel Blue
    'atrp9': {'color': '#f81cfc', 'width': 1.0, 'style': QtCore.Qt.PenStyle.DashLine},  # Light Steel Blue
    'atrp20': {'color': '#b72494', 'width': 2.0, 'style': QtCore.Qt.PenStyle.SolidLine},  # Navy
    'atrp50': {'color': '#6b1255', 'width': 1.0, 'style': QtCore.Qt.PenStyle.DashLine}  # Navy
}

STD_CONFIGS = {
    'std': {'color': '#ba68c8', 'width': 1.5, 'style': QtCore.Qt.PenStyle.SolidLine}  # Blue
}

AC_CONFIGS = {
    'ac100_lag_1': {'color': '#e0ffff', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},  # Light Cyan
    'ac100_lag_5': {'color': '#00ced1', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},  # Dark Turquoise
    'ac100_lag_10': {'color': '#00bfff', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},  # Deep Sky Blue
    'ac100_lag_20': {'color': '#008080', 'width': 1.5, 'style': QtCore.Qt.PenStyle.SolidLine}  # Teal
}

# New configuration for the Autocorrelation Regime pane
AC_REGIME_CONFIGS = {
    'ac_mom': {'color': '#e1bee7', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},  # Light Purple (Momentum)
    'ac_mr': {'color': '#ba68c8', 'width': 1.5, 'style': QtCore.Qt.PenStyle.SolidLine},
    # Medium Purple (Mean Reversion)
    'ac_comp': {'color': '#4a148c', 'width': 2.2, 'style': QtCore.Qt.PenStyle.SolidLine}
    # Indigo/Darkest (Composite)
}

SLOPE_CONFIGS = {
    'ema10_slope': {'color': '#ffccbc', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},  # Deep Orange Light
    'ema20_slope': {'color': '#ff8a65', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},
    'ema50_slope': {'color': '#ff5722', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},
    'ema100_slope': {'color': '#e64a19', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},
    'ema200_slope': {'color': '#bf360c', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine}  # Deep Orange Dark
}

VOL_CONFIGS = {
    'v': {'color': '#49bdd9', 'width': 1.0, 'style': QtCore.Qt.PenStyle.DashLine},  # Deep Orange Light
    'v9': {'color': '#fcec98', 'width': 1.0, 'style': QtCore.Qt.PenStyle.DashLine},
    'v20': {'color': '#f3cb21', 'width': 1.5, 'style': QtCore.Qt.PenStyle.SolidLine},
    'v50': {'color': '#dab312', 'width': 1.5, 'style': QtCore.Qt.PenStyle.DashLine}
}

DIST_CONFIGS = {
    'ema10_dist': {'color': '#9b7f6c', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},  # Deep Orange Light
    'ema20_dist': {'color': '#b26529', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},
    'ema50_dist': {'color': '#7b4326', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},
    'ema100_dist': {'color': '#703b24', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},
    'ema200_dist': {'color': '#5a2c27', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine}  # Deep Orange Dark
}

HURST_CONFIGS = {
    'hurst50': {'color': '#fff59d', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},  # Light Yellow
    'hurst100': {'color': '#fbc02d', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine}  # Dimmed Gold/Brownish
}

HV_CONFIGS = {
    'hv9': {'color': '#b7a3db', 'width': 1.0, 'style': QtCore.Qt.PenStyle.DashLine},  # Light Green
    'hv20': {'color': '#6539b4', 'width': 2.0, 'style': QtCore.Qt.PenStyle.SolidLine},  # Medium Green
    'hv50': {'color': '#583098', 'width': 1.0, 'style': QtCore.Qt.PenStyle.DashLine},  # Dark Green
    'iv': {'color': '#49bcd8', 'width': 2.0, 'style': QtCore.Qt.PenStyle.SolidLine}  # Magenta (Standout)
}

IVPCT_CONFIGS = {
    'iv_pct': {'color': '#b72494', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},  # Light Green
}

BB_CONFIGS = {
    'bb_upper': {'color': '#9075d6', 'width': 1.0, 'style': QtCore.Qt.PenStyle.DotLine},
    'bb_lower': {'color': '#00aaab', 'width': 1.0, 'style': QtCore.Qt.PenStyle.DotLine}
}


# TTM Squeeze Colors
TTM_COLORS = {
    'pos_up': '#00ff00',  # Bright Green (Bullish rising)
    'pos_down': '#006400',  # Dark Green (Bullish falling)
    'neg_down': '#ff0000',  # Red (Bearish falling)
    'neg_up': '#8b0000',  # Dark Red (Bearish rising)
    'sq_on': '#ff0000',  # Red dot (Squeeze ON)
    'sq_off': '#00ff00'  # Green dot (Squeeze OFF)
}


# %% Custom Graphics Items
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
        self.data = data  # [time, open, high, low, close]
        self._bounds = QtCore.QRectF()
        self._recomputeBounds()

    def setData(self, data):
        """Update OHLC data without reallocating the GraphicsObject."""
        self.data = data
        self._recomputeBounds()
        self.update()

    def _recomputeBounds(self):
        if not self.data:
            self._bounds = QtCore.QRectF()
            self.prepareGeometryChange()
            return

        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')

        for t, op, high, low, close in self.data:
            t, low, high = float(t), float(low), float(high)
            if t < min_x: min_x = t
            if t > max_x: max_x = t
            if low < min_y: min_y = low
            if high > max_y: max_y = high

        self._bounds = QtCore.QRectF(min_x - 0.5, min_y, (max_x - min_x) + 1.0, max_y - min_y)
        self.prepareGeometryChange()

    def paint(self, p, *args):
        # Direct painting to avoid QPicture issues
        # Explicit widths and colors
        w_pen = pg.mkPen('#ffffff', width=3) # White
        g_pen = pg.mkPen('#00ff00', width=3) # Green
        r_pen = pg.mkPen('#ff0000', width=3) # Red
        b_pen = pg.mkPen('#0000ff', width=3) # Blue

        p.setPen(w_pen)

        for t, op, high, low, close in self.data:
            t, op, high, low, close = float(t), float(op), float(high), float(low), float(close)

            if close > op:
                p.setPen(g_pen)
            elif close < op:
                p.setPen(r_pen)
            else:
                p.setPen(b_pen)

            if low != high:
                p.drawLine(QtCore.QPointF(t, low), QtCore.QPointF(t, high))
            p.drawLine(QtCore.QPointF(t - 0.3, op), QtCore.QPointF(t, op))
            p.drawLine(QtCore.QPointF(t, close), QtCore.QPointF(t + 0.3, close))

    def boundingRect(self):
        return self._bounds


class TTMSqueezeItem(pg.GraphicsObject):
    def __init__(self, data):
        pg.GraphicsObject.__init__(self)
        self.data = data  # list of (x, mom, squeeze_on)
        self.picture = QtGui.QPicture()
        self.generatePicture()

    def setData(self, data):
        """Update TTM squeeze histogram without reallocating the GraphicsObject."""
        self.data = data
        self.generatePicture()
        self.update()

    def generatePicture(self):
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)

        for i in range(len(self.data)):
            x, mom, sq_on = self.data[i]
            if np.isnan(mom):
                continue

            # Ensure floats
            x, mom = float(x), float(mom)

            prev_mom = self.data[i - 1][1] if i > 0 else mom

            if mom >= 0:
                color = TTM_COLORS['pos_up'] if mom >= prev_mom else TTM_COLORS['pos_down']
                rect = QtCore.QRectF(x - 0.4, 0, 0.8, -mom)
            else:
                color = TTM_COLORS['neg_down'] if mom <= prev_mom else TTM_COLORS['neg_up']
                rect = QtCore.QRectF(x - 0.4, 0, 0.8, -mom)

            p.setPen(pg.mkPen(None))
            p.setBrush(pg.mkBrush(color))
            p.drawRect(rect)

        p.end()
        self.prepareGeometryChange()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())


# %% Global/Shared State
# Global cache for reusing the window and app across multiple plot calls
_GLOBAL_QT_APP = None
_GLOBAL_MAIN_WIN = None
_GLOBAL_LAYOUT_WIDGET = None
_ACTIVE_PLOTS = []

# Persistent context for exports to prevent memory fragmentation
_EXPORT_WIN = None
_EXPORT_PLOTS = []
_EXPORT_TITLE_ITEM = None
_EXPORT_LAYOUT_VERSION = 2  # bump this when you change export layout structure


# %% Helpers
def _finite_min_max(values):
    """Return (mn, mx) from finite values or (None, None) if none exist."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, None
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if mn == mx:
        # Avoid zero-height range; expand a tiny bit (relative if possible)
        eps = abs(mn) * 1e-6 + 1e-12
        mn -= eps
        mx += eps
    return mn, mx

def _auto_scale_panes(plots, df, x_min, x_max):
    """
    Auto-scale Y-axes for all plots based on the data visible in [x_min, x_max].
    Shared logic between interactive and export modes.
    """
    if not plots:
        return

    s, e = max(0, int(x_min)), min(len(df), int(x_max))
    if s >= e:
        return

    chunk = df.iloc[s:e]
    p1 = plots[0]

    # --- Main OHLC pane scaling (guard against all-NaN) ---
    mn, mx = _finite_min_max(np.r_[chunk.get('l', pd.Series(dtype=float)).to_numpy(),
    chunk.get('h', pd.Series(dtype=float)).to_numpy()])
    if mn is not None:
        p1.setYRange(mn * 0.99, mx * 1.01, padding=0)

    # --- Re-scale indicator panes based on view (guard against all-NaN) ---
    scale_map = [
        (plots[1], VOL_CONFIGS),
        (plots[2], DIST_CONFIGS),
        (plots[3], ATR_CONFIGS),
        (plots[4], HV_CONFIGS),
        (plots[5], IVPCT_CONFIGS),
        (plots[6], ['ttm_mom'])
    ]

    for p, cfg in scale_map:
        cols = list(cfg.keys()) if isinstance(cfg, dict) else cfg
        valid_cols = [c for c in cols if c in df.columns]
        if not valid_cols:
            continue

        c_data = chunk[valid_cols]

        # For volume pane, keep the same units you plot (thousands)
        if p == plots[1]:
            c_data = c_data / 1000.0

        mn2, mx2 = _finite_min_max(c_data.to_numpy().ravel())
        if mn2 is None:
            continue

        # Give a little breathing room
        p.setYRange(mn2 * 0.9, mx2 * 1.1, padding=0)


def _setup_plot_panes(win, x_dates, row_offset=0):
    """Internal helper to create the standard 7-pane layout."""
    p1 = win.addPlot(row=row_offset + 0, col=0)
    p7 = win.addPlot(row=row_offset + 1, col=0)
    p3 = win.addPlot(row=row_offset + 2, col=0)
    p4 = win.addPlot(row=row_offset + 3, col=0)
    p5 = win.addPlot(row=row_offset + 4, col=0)
    p6 = win.addPlot(row=row_offset + 5, col=0)
    p8 = win.addPlot(row=row_offset + 6, col=0, axisItems={'bottom': DateAxis(dates=x_dates, orientation='bottom')})

    win.ci.layout.setRowStretchFactor(row_offset + 0, 50)
    win.ci.layout.setRowStretchFactor(row_offset + 1, 8)
    win.ci.layout.setRowStretchFactor(row_offset + 2, 2)
    win.ci.layout.setRowStretchFactor(row_offset + 3, 15)
    win.ci.layout.setRowStretchFactor(row_offset + 4, 15)
    win.ci.layout.setRowStretchFactor(row_offset + 5, 2)
    win.ci.layout.setRowStretchFactor(row_offset + 6, 8)

    plots = [p1, p7, p3, p4, p5, p6, p8]
    for p in plots:
        p.showGrid(x=True, y=True, alpha=0.3)
        p.getAxis('left').setWidth(70)
        p.getAxis('right').setWidth(70)
        p.showAxis('right')
        if p != p8: p.getAxis('bottom').hide()
        if p != p1:
            p.setXLink(p1)
            p.setMaximumHeight(16777215)

    p7.setLabels(left='Vol MA')
    p3.setLabels(left='EMA Dist')
    p4.setLabels(left='ATR %')
    p5.setLabels(left='IV/HV')
    p6.setLabels(left='IVPct')
    p8.setLabels(left='TTM Squeeze')
    return plots


def _force_export_layout_sync(glw: pg.GraphicsLayoutWidget, width: int, height: int):
    """
    Faster export-only layout sync.
    Assumes widget is off-screen; avoids show() and minimizes event processing.
    """
    last_size = getattr(glw, "_last_export_size", None)
    if last_size != (width, height):
        glw.setFixedSize(width, height)
        glw.setGeometry(0, 0, width, height)
        glw._last_export_size = (width, height)

    glw.ci.layout.activate()
    _GLOBAL_QT_APP.processEvents(QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

    # Make scene rect match viewport for the exporter
    vp_rect = glw.viewport().rect()
    glw.scene().setSceneRect(QtCore.QRectF(vp_rect))
    glw.update()


def _force_layout_and_scene_sync(glw: pg.GraphicsLayoutWidget, width: int | None = None, height: int | None = None):
    """
    Force GraphicsLayoutWidget to honor a target size and update its QGraphicsScene/viewport geometry.
    This fixes the 'tiny plot in top-left' issue caused by missing resize/layout passes.
    """
    if width is not None and height is not None:
        glw.setMinimumSize(width, height)
        glw.setMaximumSize(width, height)
        glw.resize(width, height)
        glw.setGeometry(0, 0, width, height)

    # Ensure layout is recalculated
    glw.ci.layout.activate()

    # Ensure a viewport exists with the correct rect (resize events happen via show/paint)
    glw.show()
    QtWidgets.QApplication.processEvents()

    # Sync scene rect to the viewport; exporter depends on correct scene geometry
    vp_rect = glw.viewport().rect()
    glw.scene().setSceneRect(QtCore.QRectF(vp_rect))
    glw.update()
    glw.repaint()
    QtWidgets.QApplication.processEvents()


def _get_export_context(width, height):
    """Singleton-style helper to maintain one hidden export window."""
    global _EXPORT_WIN, _EXPORT_PLOTS, _GLOBAL_QT_APP, _EXPORT_TITLE_ITEM, _EXPORT_LAYOUT_VERSION
    if _GLOBAL_QT_APP is None:
        _GLOBAL_QT_APP = pg.mkQApp()

    # If the export widget was created under an older layout, rebuild it.
    if _EXPORT_WIN is not None:
        current_ver = getattr(_EXPORT_WIN, "_layout_version", None)
        if current_ver != _EXPORT_LAYOUT_VERSION:
            try:
                _EXPORT_WIN.close()
            except:
                pass
            _EXPORT_WIN = None
            _EXPORT_PLOTS = []
            _EXPORT_TITLE_ITEM = None

    if _EXPORT_WIN is None:
        _EXPORT_WIN = pg.GraphicsLayoutWidget()
        _EXPORT_WIN._layout_version = _EXPORT_LAYOUT_VERSION
        _EXPORT_WIN.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen)

        _EXPORT_WIN.setFixedSize(width, height)
        _EXPORT_WIN.setGeometry(0, 0, width, height)  # Force initial geometry

        _EXPORT_TITLE_ITEM = pg.LabelItem(justify='center', size='14pt')
        _EXPORT_WIN.addItem(_EXPORT_TITLE_ITEM, row=0, col=0)
        _EXPORT_WIN.ci.layout.setRowStretchFactor(0, 3)

        _EXPORT_PLOTS = _setup_plot_panes(_EXPORT_WIN, [datetime.now()], row_offset=1)

        p1, p7, p3, p4, p5, p6, p8 = _EXPORT_PLOTS

        ohlc_item = OHLCItem([])
        p1.addItem(ohlc_item)

        # Persistent TTM items
        ttm_item = TTMSqueezeItem([])
        p8.addItem(ttm_item)

        ttm_dots = pg.ScatterPlotItem(pxMode=True)
        p8.addItem(ttm_dots)

        def _mk_series_items(plot, cfg_dict):
            items = {}
            for col, cfg in cfg_dict.items():
                items[col] = plot.plot(
                    x=[],
                    y=[],
                    pen=pg.mkPen(cfg['color'], width=cfg['width'], style=cfg['style']),
                    connect='finite'
                )
            return items

        export_state = {
            'ohlc': ohlc_item,
            'ema': _mk_series_items(p1, EMA_CONFIGS),
            'bb': _mk_series_items(p1, BB_CONFIGS),  # Add Bollinger Bands
            'vol': _mk_series_items(p7, VOL_CONFIGS),
            'dist': _mk_series_items(p3, DIST_CONFIGS),
            'atr': _mk_series_items(p4, ATR_CONFIGS),
            'hv': _mk_series_items(p5, HV_CONFIGS),
            'ivpct': _mk_series_items(p6, IVPCT_CONFIGS),
            'ttm': ttm_item,
            'ttm_dots': ttm_dots,
            'vlines': [],
        }

        p7.addLine(y=0, pen=pg.mkPen('#666', width=1))
        p3.addLine(y=1.2, pen=pg.mkPen('#666', width=1))
        p4.addLine(y=0, pen=pg.mkPen('#666', width=1))
        p5.addLine(y=0, pen=pg.mkPen('#666', width=1))
        p6.addLine(y=0.5, pen=pg.mkPen('#666', style=QtCore.Qt.PenStyle.DashLine))
        p8.addLine(y=0, pen=pg.mkPen('#666', width=1))

        _EXPORT_WIN._export_state = export_state

        _force_layout_and_scene_sync(_EXPORT_WIN, width, height)

    return _EXPORT_WIN, _EXPORT_PLOTS, _EXPORT_TITLE_ITEM


def _update_export_content(plots, df, vlines):
    """Update persistent export plot items in-place (no clears, no reallocation)."""
    export_state = getattr(_EXPORT_WIN, "_export_state", None)
    if export_state is None:
        return

    x = np.arange(len(df), dtype=float)

    # OHLC
    if len(df) > 0 and all(c in df.columns for c in ('o', 'h', 'l', 'c')):
        o = df['o'].to_numpy(dtype=float, copy=False)
        h = df['h'].to_numpy(dtype=float, copy=False)
        l = df['l'].to_numpy(dtype=float, copy=False)
        c = df['c'].to_numpy(dtype=float, copy=False)
        export_state['ohlc'].setData(
            [(float(i), float(o[i]), float(h[i]), float(l[i]), float(c[i])) for i in range(len(df))])
    else:
        export_state['ohlc'].setData([])

    def _set_group(group_items, transform=None):
        for col, item in group_items.items():
            if col in df.columns and len(df) > 0:
                y = df[col].to_numpy(dtype=float, copy=False)
                if transform is not None:
                    y = transform(y)
                item.setData(x=x, y=y, connect='finite')
                item.show()
            else:
                item.setData(x=[], y=[])
                item.hide()

    _set_group(export_state['ema'])
    _set_group(export_state['bb'])  # Update BB
    _set_group(export_state['vol'], transform=lambda y: y / 1000.0)
    _set_group(export_state['dist'])
    _set_group(export_state['atr'])
    _set_group(export_state['hv'])
    _set_group(export_state['ivpct'])

    # TTM squeeze (histogram + dots)
    if 'ttm_mom' in df.columns and 'squeeze_on' in df.columns and len(df) > 0:
        mom = df['ttm_mom'].to_numpy(dtype=float, copy=False)
        sq = df['squeeze_on'].astype(bool).to_numpy(copy=False)

        export_state['ttm'].setData([(float(i), float(mom[i]), bool(sq[i])) for i in range(len(df))])

        spots = []
        for i in range(len(df)):
            if not np.isfinite(mom[i]):
                continue
            spots.append({
                'pos': (float(i), 0.0),
                'brush': pg.mkBrush(TTM_COLORS['sq_on'] if sq[i] else TTM_COLORS['sq_off']),
                'pen': pg.mkPen(None),
                'size': 7,
            })

        export_state['ttm'].show()
        export_state['ttm_dots'].setData(spots=spots)
        export_state['ttm_dots'].show()
    else:
        export_state['ttm'].setData([])
        export_state['ttm'].hide()
        export_state['ttm_dots'].setData(spots=[])
        export_state['ttm_dots'].hide()

    # Vertical marker lines: reuse existing InfiniteLines, hide extras
    idxs = []
    if vlines:
        for v_date in vlines:
            v_dt = pd.to_datetime(v_date)
            if v_dt in df.index:
                idxs.append(int(df.index.get_loc(v_dt)))

    needed = len(idxs)
    existing = export_state['vlines']

    while len(existing) < needed:
        bundle = []
        for p in plots:
            line = pg.InfiniteLine(pos=0, angle=90,
                                   pen=pg.mkPen('darkviolet', width=0.8, style=QtCore.Qt.PenStyle.DashLine))
            p.addItem(line)
            bundle.append(line)
        existing.append(bundle)

    for i, idx in enumerate(idxs):
        for line in existing[i]:
            line.setPos(idx)
            line.show()

    for j in range(needed, len(existing)):
        for line in existing[j]:
            line.hide()


def _add_plot_content(plots, df, vlines):
    """Internal helper to populate panes with data series."""
    p1, p7, p3, p4, p5, p6, p8 = plots
    x_range = np.arange(len(df))

    p1.addItem(OHLCItem([(i, df.o.iloc[i], df.h.iloc[i], df.l.iloc[i], df.c.iloc[i]) for i in x_range]))

    for col, cfg in EMA_CONFIGS.items():
        if col in df.columns:
            p1.plot(x=x_range, y=df[col].values, pen=pg.mkPen(cfg['color'], width=cfg['width'], style=cfg['style']))

    for col, cfg in BB_CONFIGS.items():
        if col in df.columns:
            p1.plot(x=x_range, y=df[col].values, pen=pg.mkPen(cfg['color'], width=cfg['width'], style=cfg['style']))

    # Indicators
    if 'v' in df.columns:
        for col, cfg in VOL_CONFIGS.items():
            if col in df.columns:
                val = df[col].values / 1000 if col != 'v' else df[col].values / 1000
                p7.plot(x=x_range, y=val, pen=pg.mkPen(cfg['color'], width=cfg['width'], style=cfg['style']))
        p7.addLine(y=0, pen=pg.mkPen('#666', width=1))

    for col, cfg in DIST_CONFIGS.items():
        if col in df.columns:
            p3.plot(x=x_range, y=df[col].values, pen=pg.mkPen(cfg['color'], width=cfg['width'], style=cfg['style']))
    p3.addLine(y=1.2, pen=pg.mkPen('#666', width=1))

    for col, cfg in ATR_CONFIGS.items():
        if col in df.columns:
            p4.plot(x=x_range, y=df[col].values, pen=pg.mkPen(cfg['color'], width=cfg['width'], style=cfg['style']))
    p4.addLine(y=0, pen=pg.mkPen('#666', width=1))

    for col, cfg in HV_CONFIGS.items():
        if col in df.columns:
            p5.plot(x=x_range, y=df[col].values, pen=pg.mkPen(cfg['color'], width=cfg['width'], style=cfg['style']))
    p5.addLine(y=0, pen=pg.mkPen('#666', width=1))

    for col, cfg in IVPCT_CONFIGS.items():
        if col in df.columns:
            p6.plot(x=x_range, y=df[col].values, pen=pg.mkPen(cfg['color'], width=cfg['width'], style=cfg['style']))
    p6.addLine(y=0.5, pen=pg.mkPen('#666', style=QtCore.Qt.PenStyle.DashLine))

    if 'ttm_mom' in df.columns and 'squeeze_on' in df.columns:
        p8.addItem(TTMSqueezeItem([(i, df.ttm_mom.iloc[i], df.squeeze_on.iloc[i]) for i in x_range]))

        # Add visible squeeze dots (pixel-sized)
        mom = df['ttm_mom'].to_numpy(dtype=float)
        sq = df['squeeze_on'].astype(bool).to_numpy()

        spots = []
        for i in x_range:
            if not np.isfinite(mom[i]):
                continue
            spots.append({
                'pos': (float(i), 0.0),
                'brush': pg.mkBrush(TTM_COLORS['sq_on'] if sq[i] else TTM_COLORS['sq_off']),
                'pen': pg.mkPen(None),
                'size': 7,
            })

        dots = pg.ScatterPlotItem(pxMode=True)
        dots.addPoints(spots)
        p8.addItem(dots)

    p8.addLine(y=0, pen=pg.mkPen('#666', width=1))

    if vlines:
        for v_date in vlines:
            v_dt = pd.to_datetime(v_date)
            if v_dt in df.index:
                idx = df.index.get_loc(v_dt)
                for p in plots:
                    p.addItem(pg.InfiniteLine(pos=idx, angle=90,
                                              pen=pg.mkPen('darkviolet', width=0.8, style=QtCore.Qt.PenStyle.DashLine)))


# %% Main Functions
def export_swing_plot(df, path, vlines=None, display_range=50, width=1920, height=1080, title=None):
    """High-speed version using a persistent hidden window context."""
    global _GLOBAL_QT_APP
    win, plots, title_item = _get_export_context(width, height)

    # 1. Update title + axis
    title_item.setText(title if title else "")
    plots[-1].getAxis('bottom').dates = df.index

    # 2. Update persistent items (avoid clear/recreate churn)
    _update_export_content(plots, df, vlines)

    # 3. Set view range and scale Y
    p1 = plots[0]
    x_start = max(0, len(df) - display_range)
    x_end = len(df)
    p1.setXRange(x_start, x_end)

    # Use shared scaling logic
    _auto_scale_panes(plots, df, x_start, x_end)

    # 4. Fast export-only sync
    _force_export_layout_sync(win, width, height)

    # 5. Export
    exporter = pg.exporters.ImageExporter(win.scene())
    exporter.parameters()['width'] = width
    exporter.parameters()['height'] = height
    exporter.export(path)

    _GLOBAL_QT_APP.processEvents(QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

def interactive_swing_plot(full_df, display_range=250, title: str | None = None):
    """Full-featured interactive version with Toolbar, Crosshairs, and dynamic scaling."""
    global _GLOBAL_QT_APP, _GLOBAL_MAIN_WIN, _GLOBAL_LAYOUT_WIDGET, _ACTIVE_PLOTS

    if _GLOBAL_QT_APP is None:
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
        _GLOBAL_QT_APP = pg.mkQApp()

    if _GLOBAL_MAIN_WIN is None:
        _GLOBAL_MAIN_WIN = QtWidgets.QMainWindow()
        central_widget = QtWidgets.QWidget()
        _GLOBAL_MAIN_WIN.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        toolbar = QtWidgets.QHBoxLayout()
        year_cb, month_cb, day_cb = QtWidgets.QComboBox(), QtWidgets.QComboBox(), QtWidgets.QComboBox()
        toolbar.addWidget(QtWidgets.QLabel("Max Date Filter:"))
        for cb in [year_cb, month_cb, day_cb]: toolbar.addWidget(cb)
        toolbar.addStretch()
        main_layout.addLayout(toolbar)

        _GLOBAL_LAYOUT_WIDGET = pg.GraphicsLayoutWidget()
        main_layout.addWidget(_GLOBAL_LAYOUT_WIDGET)

        # Prevent "min-size" starts
        _GLOBAL_MAIN_WIN.setMinimumSize(1200, 700)
        _GLOBAL_MAIN_WIN.resize(1600, 900)

        _GLOBAL_MAIN_WIN._year_cb, _GLOBAL_MAIN_WIN._month_cb, _GLOBAL_MAIN_WIN._day_cb = year_cb, month_cb, day_cb
        _GLOBAL_MAIN_WIN._proxy = None

    main_win, win = _GLOBAL_MAIN_WIN, _GLOBAL_LAYOUT_WIDGET
    year_cb, month_cb, day_cb = main_win._year_cb, main_win._month_cb, main_win._day_cb

    # Prevent signal duplication on re-entry
    try:
        year_cb.currentIndexChanged.disconnect()
    except:
        pass
    try:
        month_cb.currentIndexChanged.disconnect()
    except:
        pass
    try:
        day_cb.currentIndexChanged.disconnect()
    except:
        pass

    # Prevent ghost hover events from previous runs
    if getattr(main_win, '_proxy', None):
        try:
            main_win._proxy.disconnect()
        except:
            pass
        main_win._proxy = None

    state = {
        'proxy': None,
        'df': None,
        'x_dates': None,
        'hover_label': None,
        'v_lines': None,
        'h_lines': None,
    }
    plots = []

    def update_y_views():
        if state['df'] is None or not plots:
            return
        p1 = plots[0]
        vr = p1.viewRange()[0]
        # Use shared scaling logic
        _auto_scale_panes(plots, state['df'], vr[0], vr[1])

    def update_plot(*args):
        global _ACTIVE_PLOTS
        nonlocal plots

        # Cleanup
        if _ACTIVE_PLOTS:
            try:
                plots[0].sigXRangeChanged.disconnect()
            except:
                pass
            if state['proxy']: state['proxy'].disconnect()
        for p in _ACTIVE_PLOTS: p.deleteLater()
        win.clear()

        # Title in row 0, plots start at row 1
        title_item = pg.LabelItem(justify='center', size='14pt')
        title_item.setText(title if title else "")
        win.addItem(title_item, row=0, col=0)
        win.ci.layout.setRowStretchFactor(0, 3)

        target_str = f"{year_cb.currentText()}-{month_cb.currentText()}-{day_cb.currentText()}"
        df = full_df[full_df.index <= target_str]
        if df.empty: return

        state['df'], state['x_dates'] = df, df.index
        plots = _setup_plot_panes(win, state['x_dates'], row_offset=1)
        _ACTIVE_PLOTS = plots
        _add_plot_content(plots, df, vlines=None)

        # Crosshair Logic (keep strong references!)
        v_lines, h_lines = [], []
        # Discrete grey pen with alpha: #808080 (grey) with alpha 100 (out of 255)
        crosshair_pen = pg.mkPen(color=(128, 128, 128, 100), width=1, style=QtCore.Qt.PenStyle.DashLine)

        for p in plots:
            v = pg.InfiniteLine(angle=90, movable=False, pen=crosshair_pen)
            h = pg.InfiniteLine(angle=0, movable=False, pen=crosshair_pen)
            p.addItem(v, ignoreBounds=True)
            p.addItem(h, ignoreBounds=True)
            v_lines.append(v)
            h_lines.append(h)
            h.hide()

        state['v_lines'] = v_lines
        state['h_lines'] = h_lines

        hover_label = pg.TextItem(anchor=(0, 0), color='#ccc', fill='#000e')
        plots[0].addItem(hover_label, ignoreBounds=True)
        state['hover_label'] = hover_label

        def update_hover(evt):
            pos = evt[0]
            for i, p in enumerate(plots):
                if p.sceneBoundingRect().contains(pos):
                    mousePoint = p.vb.mapSceneToView(pos)
                    idx = int(mousePoint.x() + 0.5)
                    if 0 <= idx < len(df):
                        row = df.iloc[idx]

                        for v in state['v_lines']:
                            v.setPos(idx)
                        for h in state['h_lines']:
                            h.hide()
                        state['h_lines'][i].setPos(mousePoint.y())
                        state['h_lines'][i].show()

                        date_str = state['x_dates'][idx].strftime('%a %Y-%m-%d') if state.get(
                            'x_dates') is not None else str(df.index[idx])

                        txt = (
                            f"<span style='font-size: 11pt; color: white; font-weight: bold;'>{date_str}</span><br>"
                            f"O:{row.o:.2f} H:{row.h:.2f} L:{row.l:.2f} C:{row.c:.2f}"
                        )
                        if 'v' in df.columns and pd.notna(row.get('v', np.nan)):
                            txt += f" V:{row.v / 1000:,.0f}k"
                        txt += "<br>"

                        def _fmt_group(config_dict, value_fmt, col_label_fn=None, transform_fn=None):
                            parts = []
                            for col, cfg in config_dict.items():
                                if col not in df.columns:
                                    continue
                                val = row.get(col, np.nan)
                                if not np.isfinite(val):
                                    continue
                                if transform_fn is not None:
                                    val = transform_fn(val)
                                color = cfg['color'] if isinstance(cfg, dict) and 'color' in cfg else str(cfg)
                                label = col_label_fn(col) if col_label_fn else col
                                parts.append(f"<span style='color:{color};'>{label}:{value_fmt(val)}</span>")
                            return " | ".join(parts)

                        emas = _fmt_group(
                            EMA_CONFIGS,
                            value_fmt=lambda v: f"{v:.2f}",
                            col_label_fn=lambda c: c.upper()
                        )
                        dists = _fmt_group(
                            DIST_CONFIGS,
                            value_fmt=lambda v: f"{v:.2f}",
                            col_label_fn=lambda c: c.replace('_dist', '').upper()
                        )
                        atrs = _fmt_group(
                            ATR_CONFIGS,
                            value_fmt=lambda v: f"{v:.2f}%",
                            col_label_fn=lambda c: c.upper()
                        )
                        hvs = _fmt_group(
                            HV_CONFIGS,
                            value_fmt=lambda v: f"{v:.2f}",
                            col_label_fn=lambda c: c.upper()
                        )
                        ivpct = _fmt_group(
                            IVPCT_CONFIGS,
                            value_fmt=lambda v: f"{v:.2f}",
                            col_label_fn=lambda c: c
                        )
                        vols = _fmt_group(
                            VOL_CONFIGS,
                            value_fmt=lambda v: f"{v:.2f}k",
                            col_label_fn=lambda c: c.upper(),
                            transform_fn=lambda v: v / 1000.0
                        )

                        if emas:
                            txt += emas + "<br>"
                        if dists:
                            txt += dists + "<br>"
                        if atrs:
                            txt += atrs + "<br>"
                        if hvs:
                            txt += hvs + "<br>"
                        if ivpct:
                            txt += ivpct + "<br>"
                        if vols:
                            txt += vols + "<br>"

                        ttm_extra = ""
                        if 'ttm_mom' in df.columns and np.isfinite(row.get('ttm_mom', np.nan)):
                            ttm_extra += f"TTM_MOM:{row.ttm_mom:.2f}"
                        if 'squeeze_on' in df.columns and pd.notna(row.get('squeeze_on', np.nan)):
                            sq = bool(row.squeeze_on)
                            ttm_extra += (" | " if ttm_extra else "") + f"SQUEEZE:{'ON' if sq else 'OFF'}"
                        if ttm_extra:
                            txt += ttm_extra

                        state['hover_label'].setHtml(txt)

                        # Place near top-left of pane 1 view
                        vb_range = plots[0].vb.viewRange()
                        state['hover_label'].setPos(
                            vb_range[0][0] + (vb_range[0][1] - vb_range[0][0]) * 0.01,
                            vb_range[1][1] - (vb_range[1][1] - vb_range[1][0]) * 0.01
                        )

        state['proxy'] = pg.SignalProxy(plots[0].scene().sigMouseMoved, rateLimit=60, slot=update_hover)
        main_win._proxy = state['proxy']
        plots[0].sigXRangeChanged.connect(update_y_views)
        plots[0].setXRange(max(0, len(df) - display_range), len(df))
        update_y_views()

    # Initial UI Setup
    year_cb.blockSignals(True);
    month_cb.blockSignals(True);
    day_cb.blockSignals(True)
    year_cb.clear();
    month_cb.clear();
    day_cb.clear()
    year_cb.addItems([str(y) for y in sorted(full_df.index.year.unique(), reverse=True)])
    month_cb.addItems([f"{m:02d}" for m in range(1, 13)])
    day_cb.addItems([f"{d:02d}" for d in range(1, 32)])

    last_date = full_df.index[-1]
    year_cb.setCurrentText(str(last_date.year))
    month_cb.setCurrentText(f"{last_date.month:02d}")
    day_cb.setCurrentText(f"{last_date.day:02d}")

    for cb in [year_cb, month_cb, day_cb]: cb.currentIndexChanged.connect(update_plot)
    year_cb.blockSignals(False);
    month_cb.blockSignals(False);
    day_cb.blockSignals(False)

    update_plot()
    main_win.showNormal()

    # Force layout AFTER the window is actually shown
    def _after_show():
        main_win.resize(1600, 900)
        _force_layout_and_scene_sync(win)

    QtCore.QTimer.singleShot(0, _after_show)

    # Standard blocking behavior for scripts, while being friendly to interactive sessions
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        pg.exec()
