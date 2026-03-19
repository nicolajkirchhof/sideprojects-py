"""
finance.visualizations._chart
==============================
Chart layout helpers: pane setup, data binding, and auto-scaling.
All PyQtGraph — no matplotlib.
"""
import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph import QtCore
from pyqtgraph.Qt import QtWidgets

from ._config import (
    MA_CONFIGS, ATR_CONFIGS, VOL_CONFIGS, DIST_CONFIGS,
    HV_CONFIGS, IVPCT_CONFIGS, BB_CONFIGS,
    ATR_RATIO_THRESHOLD, ATR_RATIO_COLOR,
)
from ._items import DateAxis, OHLCItem, TTMSqueezeItem, VolumeProfileItem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _finite_min_max(values):
    """Return (mn, mx) from finite values, or (None, None) if none exist."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, None
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mn == mx:
        eps = abs(mn) * 1e-6 + 1e-12
        mn -= eps
        mx += eps
    return mn, mx


def _force_layout_and_scene_sync(glw, width=None, height=None):
    """
    Force GraphicsLayoutWidget to honour a target size and sync scene geometry.
    Fixes the 'tiny plot in top-left' issue after show().
    """
    if width is not None and height is not None:
        glw.setMinimumSize(width, height)
        glw.setMaximumSize(width, height)
        glw.resize(width, height)
        glw.setGeometry(0, 0, width, height)
    glw.ci.layout.activate()
    glw.show()
    QtWidgets.QApplication.processEvents()
    vp_rect = glw.viewport().rect()
    glw.scene().setSceneRect(QtCore.QRectF(vp_rect))
    glw.update()
    glw.repaint()
    QtWidgets.QApplication.processEvents()


# ---------------------------------------------------------------------------
# Pane layout
# ---------------------------------------------------------------------------
# Pane order (8 panes):
#   p1   — OHLC + MAs + BB + SPY overlay
#   p_vol — Volume MA
#   p_dist — EMA Distance
#   p_atr  — ATR %
#   p_ratio — ATR Impulse Ratio (pct / ATR20), ±1.75 lines   ← NEW
#   p_hv   — IV / HV
#   p_ivpct — IV Percentile
#   p_ttm  — TTM Squeeze

def _setup_plot_panes(win, x_dates, row_offset=0):
    """Create the standard 8-pane layout and return [p1, p_vol, p_dist, p_atr, p_ratio, p_hv, p_ivpct, p_ttm]."""
    p1      = win.addPlot(row=row_offset + 0, col=0)
    p_vol   = win.addPlot(row=row_offset + 1, col=0)
    p_dist  = win.addPlot(row=row_offset + 2, col=0)
    p_atr   = win.addPlot(row=row_offset + 3, col=0)
    p_ratio = win.addPlot(row=row_offset + 4, col=0)
    p_hv    = win.addPlot(row=row_offset + 5, col=0)
    p_ivpct = win.addPlot(row=row_offset + 6, col=0)
    p_ttm   = win.addPlot(row=row_offset + 7, col=0,
                           axisItems={'bottom': DateAxis(dates=x_dates, orientation='bottom')})

    win.ci.layout.setRowStretchFactor(row_offset + 0, 50)
    win.ci.layout.setRowStretchFactor(row_offset + 1, 8)
    win.ci.layout.setRowStretchFactor(row_offset + 2, 2)
    win.ci.layout.setRowStretchFactor(row_offset + 3, 12)
    win.ci.layout.setRowStretchFactor(row_offset + 4, 8)
    win.ci.layout.setRowStretchFactor(row_offset + 5, 12)
    win.ci.layout.setRowStretchFactor(row_offset + 6, 2)
    win.ci.layout.setRowStretchFactor(row_offset + 7, 8)

    plots = [p1, p_vol, p_dist, p_atr, p_ratio, p_hv, p_ivpct, p_ttm]
    for p in plots:
        p.showGrid(x=True, y=True, alpha=0.3)
        p.getAxis('left').setWidth(70)
        p.getAxis('right').setWidth(70)
        p.showAxis('right')
        if p != p_ttm:
            p.getAxis('bottom').hide()
        if p != p1:
            p.setXLink(p1)
            p.setMaximumHeight(16777215)

    p_vol.setLabels(left='Vol MA')
    p_dist.setLabels(left='EMA Dist')
    p_atr.setLabels(left='ATR %')
    p_ratio.setLabels(left='ATR Ratio')
    p_hv.setLabels(left='IV/HV')
    p_ivpct.setLabels(left='IVPct')
    p_ttm.setLabels(left='TTM Squeeze')
    return plots


# ---------------------------------------------------------------------------
# Data binding
# ---------------------------------------------------------------------------

def _add_plot_content(plots, df, vlines, spy_df=None):
    """
    Populate all 8 panes with data series.

    Parameters
    ----------
    plots   : list returned by _setup_plot_panes
    df      : filtered df_day for the active symbol
    vlines  : list of date strings for vertical marker lines
    spy_df  : optional df_day for SPY — draws a normalised overlay on p1
    """
    p1, p_vol, p_dist, p_atr, p_ratio, p_hv, p_ivpct, p_ttm = plots
    x_range = np.arange(len(df))

    # Volume Profile (behind candles)
    vp_item = VolumeProfileItem(df, width_fraction=0.75, bins=100)
    vp_item.setZValue(-10)
    p1.addItem(vp_item)
    vp_item.setViewRange(0, len(df))

    # OHLC candles
    p1.addItem(OHLCItem([(i, df.o.iloc[i], df.h.iloc[i], df.l.iloc[i], df.c.iloc[i]) for i in x_range]))

    # MAs and Bollinger Bands
    for col, cfg in MA_CONFIGS.items():
        if col in df.columns:
            p1.plot(x=x_range, y=df[col].values,
                    pen=pg.mkPen(cfg['color'], width=cfg['width'], style=cfg['style']))
    for col, cfg in BB_CONFIGS.items():
        if col in df.columns:
            p1.plot(x=x_range, y=df[col].values,
                    pen=pg.mkPen(cfg['color'], width=cfg['width'], style=cfg['style']))

    # SPY normalised overlay on the OHLC pane
    if spy_df is not None and not spy_df.empty and 'c' in spy_df.columns:
        spy_aligned = spy_df['c'].reindex(df.index, method='ffill').dropna()
        if len(spy_aligned) >= 2:
            first_spy  = spy_aligned.iloc[0]
            first_self = df['c'].reindex(spy_aligned.index).iloc[0]
            spy_norm   = spy_aligned / first_spy * first_self
            # Map to integer x-positions of df
            spy_x = np.array([df.index.get_loc(idx) for idx in spy_aligned.index])
            p1.plot(x=spy_x, y=spy_norm.values,
                    pen=pg.mkPen(color=(200, 200, 200, 80), width=1.2,
                                 style=QtCore.Qt.PenStyle.SolidLine))

    # Volume pane
    if 'v' in df.columns:
        for col, cfg in VOL_CONFIGS.items():
            if col in df.columns:
                p_vol.plot(x=x_range, y=df[col].values / 1000,
                           pen=pg.mkPen(cfg['color'], width=cfg['width'], style=cfg['style']))
        p_vol.addLine(y=0, pen=pg.mkPen('#666', width=1))

    # EMA Distance pane
    for col, cfg in DIST_CONFIGS.items():
        if col in df.columns:
            p_dist.plot(x=x_range, y=df[col].values,
                        pen=pg.mkPen(cfg['color'], width=cfg['width'], style=cfg['style']))
    p_dist.addLine(y=1.2, pen=pg.mkPen('#666', width=1))

    # ATR% pane
    for col, cfg in ATR_CONFIGS.items():
        if col in df.columns:
            p_atr.plot(x=x_range, y=df[col].values,
                       pen=pg.mkPen(cfg['color'], width=cfg['width'], style=cfg['style']))
    p_atr.addLine(y=0, pen=pg.mkPen('#666', width=1))

    # ATR Impulse Ratio pane  (pct / ATR20)
    if 'pct' in df.columns and 'atrp20' in df.columns:
        atr_ratio = (df['pct'] / df['atrp20'].replace(0, np.nan)).fillna(0)
        p_ratio.plot(x=x_range, y=atr_ratio.values,
                     pen=pg.mkPen('#e0e0e0', width=1.0))
        for sign in (1, -1):
            p_ratio.addLine(y=sign * ATR_RATIO_THRESHOLD,
                            pen=pg.mkPen(ATR_RATIO_COLOR, width=1.2,
                                         style=QtCore.Qt.PenStyle.DashLine))
        p_ratio.addLine(y=0, pen=pg.mkPen('#666', width=1))

    # IV/HV pane
    for col, cfg in HV_CONFIGS.items():
        if col in df.columns:
            p_hv.plot(x=x_range, y=df[col].values,
                      pen=pg.mkPen(cfg['color'], width=cfg['width'], style=cfg['style']))
    p_hv.addLine(y=0, pen=pg.mkPen('#666', width=1))

    # IV Percentile pane
    for col, cfg in IVPCT_CONFIGS.items():
        if col in df.columns:
            p_ivpct.plot(x=x_range, y=df[col].values,
                         pen=pg.mkPen(cfg['color'], width=cfg['width'], style=cfg['style']))
    p_ivpct.addLine(y=0.5, pen=pg.mkPen('#666', style=QtCore.Qt.PenStyle.DashLine))

    # TTM Squeeze pane
    if 'ttm_mom' in df.columns and 'squeeze_on' in df.columns:
        p_ttm.addItem(TTMSqueezeItem([(i, df.ttm_mom.iloc[i], df.squeeze_on.iloc[i]) for i in x_range]))
        mom = df['ttm_mom'].to_numpy(dtype=float)
        sq  = df['squeeze_on'].astype(bool).to_numpy()
        spots = []
        for i in x_range:
            if not np.isfinite(mom[i]):
                continue
            from ._config import TTM_COLORS
            spots.append({
                'pos':   (float(i), 0.0),
                'brush': pg.mkBrush(TTM_COLORS['sq_on'] if sq[i] else TTM_COLORS['sq_off']),
                'pen':   pg.mkPen(None),
                'size':  7,
            })
        dots = pg.ScatterPlotItem(pxMode=True)
        dots.addPoints(spots)
        p_ttm.addItem(dots)
    p_ttm.addLine(y=0, pen=pg.mkPen('#666', width=1))

    # Vertical marker lines
    if vlines:
        for v_date in vlines:
            v_dt = pd.to_datetime(v_date)
            if v_dt in df.index:
                idx = df.index.get_loc(v_dt)
                for p in plots:
                    p.addItem(pg.InfiniteLine(
                        pos=idx, angle=90,
                        pen=pg.mkPen('darkviolet', width=0.8,
                                     style=QtCore.Qt.PenStyle.DashLine)))


# ---------------------------------------------------------------------------
# Auto-scaling
# ---------------------------------------------------------------------------

def _auto_scale_panes(plots, df, x_min, x_max):
    """Auto-scale Y-axes for all 8 panes to the data visible in [x_min, x_max]."""
    if not plots:
        return
    s, e = max(0, int(x_min)), min(len(df), int(x_max))
    if s >= e:
        return
    chunk = df.iloc[s:e]
    p1 = plots[0]

    # Update VolumeProfileItem
    for item in p1.items:
        if isinstance(item, VolumeProfileItem):
            if item.df is None or item.df.empty:
                item.df = df
            item.setViewRange(x_min, x_max)

    # OHLC pane
    mn, mx = _finite_min_max(np.r_[
        chunk.get('l', pd.Series(dtype=float)).to_numpy(),
        chunk.get('h', pd.Series(dtype=float)).to_numpy(),
    ])
    if mn is not None:
        p1.setYRange(mn * 0.99, mx * 1.01, padding=0)

    # Indicator panes  [pane, config_keys]
    scale_map = [
        (plots[1], list(VOL_CONFIGS.keys())),
        (plots[2], list(DIST_CONFIGS.keys())),
        (plots[3], list(ATR_CONFIGS.keys())),
        (plots[4], ['pct', 'atrp20']),   # ATR ratio — need pct and atrp20 to derive range
        (plots[5], list(HV_CONFIGS.keys())),
        (plots[6], list(IVPCT_CONFIGS.keys())),
        (plots[7], ['ttm_mom']),
    ]

    for p, cols in scale_map:
        valid_cols = [c for c in cols if c in df.columns]
        if not valid_cols:
            continue
        c_data = chunk[valid_cols]
        if p == plots[1]:          # volume: already in thousands in the plot
            c_data = c_data / 1000.0
        elif p == plots[4]:        # ATR ratio: derive the series
            if 'pct' in chunk.columns and 'atrp20' in chunk.columns:
                ratio = chunk['pct'] / chunk['atrp20'].replace(0, np.nan)
                mn2, mx2 = _finite_min_max(ratio.to_numpy())
            else:
                continue
            if mn2 is None:
                continue
            bound = max(abs(mn2), abs(mx2), ATR_RATIO_THRESHOLD + 0.5)
            p.setYRange(-bound, bound, padding=0)
            continue

        mn2, mx2 = _finite_min_max(c_data.to_numpy().ravel())
        if mn2 is None:
            continue
        p.setYRange(mn2 * 0.9, mx2 * 1.1, padding=0)
