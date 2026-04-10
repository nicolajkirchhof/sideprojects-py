"""
finance.apps.swing_plot._items
================================
Custom PyQtGraph graphics items for the swing plot dashboard.
"""
import numpy as np
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui

from finance.utils.chart_styles import TTM_COLORS


class DateAxis(pg.AxisItem):
    def __init__(self, dates, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dates = dates

    def tickStrings(self, values, scale, spacing):
        return [
            self.dates[int(v)].strftime('%y-%m-%d')
            if 0 <= int(v) < len(self.dates) else ''
            for v in values
        ]

    def tickValues(self, minVal, maxVal, size):
        return super().tickValues(minVal, maxVal, size * 0.5)


class OHLCItem(pg.GraphicsObject):
    def __init__(self, data):
        pg.GraphicsObject.__init__(self)
        self.data = data  # list of (time, open, high, low, close)
        self._bounds = QtCore.QRectF()
        self._recomputeBounds()

    def setData(self, data):
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
        w_pen = pg.mkPen('#ffffff', width=2)
        g_pen = pg.mkPen('#1b9e44', width=2)
        r_pen = pg.mkPen('#cf3030', width=2)
        b_pen = pg.mkPen('#0000ff', width=2)
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
            x, mom = float(x), float(mom)
            prev_mom = self.data[i - 1][1] if i > 0 else mom
            if mom >= 0:
                color = TTM_COLORS['pos_up'] if mom >= prev_mom else TTM_COLORS['pos_down']
            else:
                color = TTM_COLORS['neg_down'] if mom <= prev_mom else TTM_COLORS['neg_up']
            rect = QtCore.QRectF(x - 0.4, 0, 0.8, mom)
            p.setPen(pg.mkPen(None))
            p.setBrush(pg.mkBrush(color))
            p.drawRect(rect)
        p.end()
        self.prepareGeometryChange()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())


class VolumeProfileItem(pg.GraphicsObject):
    def __init__(self, df, width_fraction=0.75, bins=100):
        pg.GraphicsObject.__init__(self)
        self.df = df
        self.width_fraction = width_fraction
        self.bins = bins
        self.picture = QtGui.QPicture()
        self._bounds = QtCore.QRectF()
        self.current_range = None

    def setViewRange(self, x_min, x_max):
        s, e = max(0, int(x_min)), min(len(self.df), int(x_max))
        if s >= e:
            self.picture = QtGui.QPicture()
            self.update()
            return
        if self.current_range == (s, e):
            return
        self.current_range = (s, e)

        chunk = self.df.iloc[s:e]
        if chunk.empty or 'c' not in chunk or 'v' not in chunk:
            return

        prices  = chunk['c'].values
        volumes = chunk['v'].values
        mask    = np.isfinite(prices) & np.isfinite(volumes)
        prices  = prices[mask]
        volumes = volumes[mask]
        if len(prices) == 0:
            return

        hist, bin_edges = np.histogram(prices, bins=self.bins, weights=volumes)
        if hist.max() == 0:
            return

        view_width    = x_max - x_min
        max_bar_width = view_width * self.width_fraction
        scale         = max_bar_width / hist.max()

        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        color = QtGui.QColor(255, 255, 255, 40)
        p.setPen(pg.mkPen(None))
        p.setBrush(pg.mkBrush(color))
        base_x = x_min
        for i in range(len(hist)):
            if hist[i] == 0:
                continue
            p.drawRect(QtCore.QRectF(base_x, bin_edges[i], hist[i] * scale, bin_edges[i + 1] - bin_edges[i]))
        p.end()
        self._bounds = QtCore.QRectF(base_x, bin_edges[0], max_bar_width, bin_edges[-1] - bin_edges[0])
        self.prepareGeometryChange()
        self.update()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return self._bounds
