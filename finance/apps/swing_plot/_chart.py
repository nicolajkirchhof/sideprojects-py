"""
finance.apps.swing_plot._chart
================================
Chart layout helpers: pane setup, data binding, and auto-scaling.
All PyQtGraph — no matplotlib.
"""
import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph import QtCore
from pyqtgraph.Qt import QtWidgets

from finance.utils.chart_styles import (
    MA_CONFIGS, ATR_CONFIGS, VOL_CONFIGS,
    HV_CONFIGS, IVPCT_CONFIGS, BB_CONFIGS,
    ATR_RATIO_THRESHOLD, ATR_RATIO_COLOR, TTM_COLORS,
    qt_pen_cfg,
)

UP_COLOR   = '#195d34'
DOWN_COLOR = '#7d2f37'
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
# Pane order (6 panes):
#   p1     — OHLC + MAs + BB + SPY overlay
#   p_vol  — Volume (coloured bars)
#   p_atr  — ATR% (left axis) + COTR (right axis, twin viewbox)
#   p_hv   — IV / HV (right axis) + IV Percentile (left axis, twin viewbox)
#   p_rs   — Comparative Relative Strength vs SPY (RS + RSMA20)
#   p_ttm  — TTM Squeeze

def _setup_plot_panes(win, x_dates, row_offset=0):
    """Create the 6-pane layout and return [p1, p_vol, p_atr, p_hv, p_rs, p_ttm]."""
    p1      = win.addPlot(row=row_offset + 0, col=0)
    p_vol   = win.addPlot(row=row_offset + 1, col=0)
    p_atr   = win.addPlot(row=row_offset + 2, col=0)
    p_hv    = win.addPlot(row=row_offset + 3, col=0)
    p_rs    = win.addPlot(row=row_offset + 4, col=0)
    p_ttm   = win.addPlot(row=row_offset + 5, col=0,
                           axisItems={'bottom': DateAxis(dates=x_dates, orientation='bottom')})

    win.ci.layout.setRowStretchFactor(row_offset + 0, 55)
    win.ci.layout.setRowStretchFactor(row_offset + 1, 8)
    win.ci.layout.setRowStretchFactor(row_offset + 2, 14)
    win.ci.layout.setRowStretchFactor(row_offset + 3, 14)
    win.ci.layout.setRowStretchFactor(row_offset + 4, 10)
    win.ci.layout.setRowStretchFactor(row_offset + 5, 8)

    plots = [p1, p_vol, p_atr, p_hv, p_rs, p_ttm]
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

    p_vol.setLabels(left='Volume')
    p_atr.setLabels(left='ATR %', right='COTR')
    p_hv.setLabels(left='IVP %', right='IV/HV')
    p_rs.setLabels(left='RS vs SPY')
    p_ttm.setLabels(left='TTM Squeeze')

    def _twin_viewbox(parent_plot):
        vb = pg.ViewBox()
        parent_plot.scene().addItem(vb)
        parent_plot.getAxis('right').linkToView(vb)
        vb.setXLink(p1)

        def _sync():
            vb.setGeometry(parent_plot.vb.sceneBoundingRect())
            vb.linkedViewChanged(parent_plot.vb, vb.XAxis)
        parent_plot.vb.sigResized.connect(_sync)
        return vb

    # Twin viewboxes for right-axis overlays
    p_hv._right_vb  = _twin_viewbox(p_hv)   # IV/HV on right, IVP on left
    p_atr._right_vb = _twin_viewbox(p_atr)  # COTR on right, ATR% on left

    return plots


# ---------------------------------------------------------------------------
# Data binding
# ---------------------------------------------------------------------------

def _add_plot_content(plots, df, vlines, spy_df=None):
    """
    Populate all 6 panes with data series.
    """
    p1, p_vol, p_atr, p_hv, p_rs, p_ttm = plots
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
            p1.plot(x=x_range, y=df[col].values, pen=pg.mkPen(**qt_pen_cfg(cfg)))
    for col, cfg in BB_CONFIGS.items():
        if col in df.columns:
            p1.plot(x=x_range, y=df[col].values, pen=pg.mkPen(**qt_pen_cfg(cfg)))

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

    # Volume pane — coloured bars (green if close>=open, red otherwise)
    if 'v' in df.columns:
        vol_vals = (df['v'] / 1000).to_numpy(dtype=float)
        if 'o' in df.columns and 'c' in df.columns:
            up_mask = (df['c'] >= df['o']).to_numpy()
        else:
            up_mask = np.ones(len(df), dtype=bool)
        brushes = [pg.mkBrush(UP_COLOR if up else DOWN_COLOR) for up in up_mask]
        bars = pg.BarGraphItem(x=x_range, height=vol_vals, width=0.8,
                                brushes=brushes, pens=[pg.mkPen(None)] * len(x_range))
        p_vol.addItem(bars)
        # Volume 20d MA overlay (kept)
        if 'v20' in df.columns:
            p_vol.plot(x=x_range, y=(df['v20'] / 1000).values,
                       pen=pg.mkPen(**qt_pen_cfg(VOL_CONFIGS['v20'])))
        p_vol.addLine(y=0, pen=pg.mkPen('#666', width=1))

    # ATR% pane — ATRs on LEFT axis, COTR on RIGHT axis (twin viewbox)
    for col, cfg in ATR_CONFIGS.items():
        if col in df.columns:
            p_atr.plot(x=x_range, y=df[col].values,
                       pen=pg.mkPen(**qt_pen_cfg(cfg)))
    p_atr.addLine(y=0, pen=pg.mkPen('#666', width=1))

    # COTR (change over true range) on the right-axis twin viewbox
    atr_right_vb = getattr(p_atr, '_right_vb', None)
    if atr_right_vb is not None and 'pct' in df.columns and 'atrp20' in df.columns:
        cotr = (df['pct'] / df['atrp20'].replace(0, np.nan)).fillna(0).to_numpy()
        cotr_curve = pg.PlotDataItem(
            x=x_range, y=cotr,
            pen=pg.mkPen(color='#ffaf1c', width=1.5, style=QtCore.Qt.PenStyle.SolidLine),
        )
        atr_right_vb.addItem(cotr_curve)
        for y in (ATR_RATIO_THRESHOLD, -ATR_RATIO_THRESHOLD):
            href = pg.InfiniteLine(
                pos=y, angle=0,
                pen=pg.mkPen(color='#003b7a', width=1.0, style=QtCore.Qt.PenStyle.SolidLine),
            )
            atr_right_vb.addItem(href, ignoreBounds=True)

    # IV/HV pane — IV/HV on RIGHT viewbox (right axis), IVP on LEFT viewbox (left axis)
    right_vb = getattr(p_hv, '_right_vb', None)
    for col, cfg in HV_CONFIGS.items():
        if col in df.columns and right_vb is not None:
            curve = pg.PlotDataItem(x=x_range, y=df[col].values,
                                    pen=pg.mkPen(**qt_pen_cfg(cfg)))
            right_vb.addItem(curve)
    for col, cfg in IVPCT_CONFIGS.items():
        if col in df.columns:
            p_hv.plot(x=x_range, y=df[col].values,
                      pen=pg.mkPen(**qt_pen_cfg(cfg)))
    p_hv.addLine(y=50, pen=pg.mkPen('#666', style=QtCore.Qt.PenStyle.DashLine))

    # Comparative Relative Strength vs SPY (RS = c / spy_c; RSMA20 = ma20 / spy_ma20)
    if spy_df is not None and not spy_df.empty and 'c' in spy_df.columns and 'c' in df.columns:
        spy_c = spy_df['c'].reindex(df.index, method='ffill')
        rs = (df['c'] / spy_c).to_numpy(dtype=float)
        # Save the raw RS series on the pane for the hover label + auto-scale
        p_rs._rs_series = rs
        p_rs.plot(x=x_range, y=rs,
                  pen=pg.mkPen(color='#dc9b56', width=1.5,
                               style=QtCore.Qt.PenStyle.SolidLine))

        if 'ma20' in df.columns:
            spy_ma20 = spy_df['c'].rolling(20).mean().reindex(df.index, method='ffill')
            rs_ma20 = (df['ma20'] / spy_ma20).to_numpy(dtype=float)
            p_rs._rsma_series = rs_ma20
            p_rs.plot(x=x_range, y=rs_ma20,
                      pen=pg.mkPen(color='#865371', width=1.5,
                                   style=QtCore.Qt.PenStyle.SolidLine))
    else:
        p_rs._rs_series = None
        p_rs._rsma_series = None

    # TTM Squeeze pane
    if 'ttm_mom' in df.columns and 'squeeze_on' in df.columns:
        p_ttm.addItem(TTMSqueezeItem([(i, df.ttm_mom.iloc[i], df.squeeze_on.iloc[i]) for i in x_range]))
        mom = df['ttm_mom'].to_numpy(dtype=float)
        sq  = df['squeeze_on'].astype(bool).to_numpy()
        spots = []
        for i in x_range:
            if not np.isfinite(mom[i]):
                continue
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

def _sync_twin_viewboxes(plots):
    """Force twin viewboxes to match their parent pane geometry. Call after
    layout has settled to avoid the right-axis curves leaking into other panes.
    """
    for p in plots:
        vb = getattr(p, '_right_vb', None)
        if vb is None:
            continue
        try:
            vb.setGeometry(p.vb.sceneBoundingRect())
            vb.linkedViewChanged(p.vb, vb.XAxis)
        except Exception:
            pass


def _auto_scale_panes(plots, df, x_min, x_max):
    """Auto-scale Y-axes for the 6 panes to the data visible in [x_min, x_max]."""
    if not plots:
        return
    s, e = max(0, int(x_min)), min(len(df), int(x_max))
    if s >= e:
        return
    chunk = df.iloc[s:e]
    p1, p_vol, p_atr, p_hv, p_rs, p_ttm = plots

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

    # Volume pane — scale to visible bars (in thousands)
    if 'v' in chunk.columns:
        vols = (chunk['v'] / 1000).to_numpy(dtype=float)
        mn2, mx2 = _finite_min_max(vols)
        if mn2 is not None:
            p_vol.setYRange(0, mx2 * 1.1, padding=0)

    # ATR% pane — left axis
    atr_cols = [c for c in ATR_CONFIGS.keys() if c in chunk.columns]
    if atr_cols:
        mn2, mx2 = _finite_min_max(chunk[atr_cols].to_numpy().ravel())
        if mn2 is not None:
            span = mx2 - mn2
            p_atr.setYRange(mn2 - 0.1 * span, mx2 + 0.1 * span, padding=0)

    # ATR pane — right axis (COTR twin viewbox)
    atr_right_vb = getattr(p_atr, '_right_vb', None)
    if atr_right_vb is not None and 'pct' in chunk.columns and 'atrp20' in chunk.columns:
        ratio = chunk['pct'] / chunk['atrp20'].replace(0, np.nan)
        mn2, mx2 = _finite_min_max(ratio.to_numpy())
        if mn2 is not None:
            bound = max(abs(mn2), abs(mx2), ATR_RATIO_THRESHOLD + 0.5)
            atr_right_vb.setYRange(-bound, bound, padding=0)
        atr_right_vb.setGeometry(p_atr.vb.sceneBoundingRect())

    # HV pane — left axis = IVP, right axis (twin vb) = IV/HV
    ivp_cols = [c for c in IVPCT_CONFIGS.keys() if c in chunk.columns]
    if ivp_cols:
        mn2, mx2 = _finite_min_max(chunk[ivp_cols].to_numpy().ravel())
        if mn2 is not None:
            p_hv.setYRange(max(0, mn2 - 2), min(100, mx2 + 2), padding=0)
        else:
            p_hv.setYRange(0, 100, padding=0)
    else:
        p_hv.setYRange(0, 100, padding=0)

    right_vb = getattr(p_hv, '_right_vb', None)
    if right_vb is not None:
        hv_cols = [c for c in HV_CONFIGS.keys() if c in chunk.columns]
        if hv_cols:
            mn2, mx2 = _finite_min_max(chunk[hv_cols].to_numpy().ravel())
            if mn2 is not None:
                span = mx2 - mn2
                right_vb.setYRange(mn2 - 0.1 * span, mx2 + 0.1 * span, padding=0)
        # Keep the secondary vb aligned to the pane geometry
        right_vb.setGeometry(p_hv.vb.sceneBoundingRect())

    # RS pane
    rs_series   = getattr(p_rs, '_rs_series', None)
    rsma_series = getattr(p_rs, '_rsma_series', None)
    rs_values = []
    if rs_series is not None:
        rs_values.append(np.asarray(rs_series)[s:e])
    if rsma_series is not None:
        rs_values.append(np.asarray(rsma_series)[s:e])
    if rs_values:
        mn2, mx2 = _finite_min_max(np.concatenate(rs_values))
        if mn2 is not None:
            span = mx2 - mn2
            p_rs.setYRange(mn2 - 0.08 * span, mx2 + 0.08 * span, padding=0)

    # TTM pane
    if 'ttm_mom' in chunk.columns:
        mn2, mx2 = _finite_min_max(chunk['ttm_mom'].to_numpy())
        if mn2 is not None:
            span = mx2 - mn2
            p_ttm.setYRange(mn2 - 0.1 * span, mx2 + 0.1 * span, padding=0)
