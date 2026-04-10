"""
finance.utils.chart_styles
===========================
Framework-agnostic indicator style definitions.

Each config dict maps a column name to a style dict with:
  - color: hex color string
  - width: line width (float)
  - dash:  one of "solid", "dash", "dot"

Use the conversion helpers to translate to Qt or matplotlib types.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Dash style helpers
# ---------------------------------------------------------------------------

def to_qt_pen_style(dash: str):
    """Convert a dash string to a Qt PenStyle enum value."""
    from pyqtgraph import QtCore
    return {
        'solid': QtCore.Qt.PenStyle.SolidLine,
        'dash':  QtCore.Qt.PenStyle.DashLine,
        'dot':   QtCore.Qt.PenStyle.DotLine,
    }[dash]


def to_mpl_linestyle(dash: str) -> str:
    """Convert a dash string to a matplotlib linestyle string."""
    return {
        'solid': '-',
        'dash':  '--',
        'dot':   ':',
    }[dash]


def qt_pen_cfg(cfg: dict) -> dict:
    """Return a copy with 'dash' replaced by Qt 'style' for use with pg.mkPen."""
    return {
        'color': cfg['color'],
        'width': cfg['width'],
        'style': to_qt_pen_style(cfg['dash']),
    }


# ---------------------------------------------------------------------------
# Swing Plot configs (PyQtGraph multi-pane chart)
# ---------------------------------------------------------------------------

MA_CONFIGS = {
    'ma5':   {'color': '#89c9f5', 'width': 1.0, 'dash': 'dash'},
    'ma10':  {'color': '#58b0f6', 'width': 1.0, 'dash': 'dash'},
    'ma20':  {'color': '#0181ec', 'width': 1.5, 'dash': 'solid'},
    'ma50':  {'color': '#004b9a', 'width': 1.0, 'dash': 'dash'},
    'ma100': {'color': '#004b9a', 'width': 1.0, 'dash': 'dash'},
    'ma200': {'color': '#00285a', 'width': 1.0, 'dash': 'dash'},
}

ATR_CONFIGS = {
    'atrp9':  {'color': '#f5a1df', 'width': 1.0, 'dash': 'dot'},
    'atrp20': {'color': '#b72494', 'width': 1.5, 'dash': 'solid'},
}

SLOPE_CONFIGS = {
    'ma5_slope':   {'color': '#ffccbc', 'width': 1.0, 'dash': 'solid'},
    'ma10_slope':  {'color': '#ffccbc', 'width': 1.0, 'dash': 'solid'},
    'ma20_slope':  {'color': '#ff8a65', 'width': 1.0, 'dash': 'solid'},
    'ma50_slope':  {'color': '#ff5722', 'width': 1.0, 'dash': 'solid'},
    'ma200_slope': {'color': '#bf360c', 'width': 1.0, 'dash': 'solid'},
}

VOL_CONFIGS = {
    'v':   {'color': '#49bdd9', 'width': 1.0, 'dash': 'dash'},
    'v20': {'color': '#0e64ab', 'width': 1.5, 'dash': 'solid'},
}

# ma100_dist / ma200_dist removed — not decision-relevant for 20-60 day swings
DIST_CONFIGS = {
    'ma10_dist': {'color': '#9b7f6c', 'width': 1.0, 'dash': 'solid'},
    'ma20_dist': {'color': '#b26529', 'width': 1.0, 'dash': 'solid'},
    'ma50_dist': {'color': '#7b4326', 'width': 1.0, 'dash': 'solid'},
}

HV_CONFIGS = {
    'hv9':  {'color': '#b7a3db', 'width': 1.0, 'dash': 'dash'},
    'hv20': {'color': '#673ab7', 'width': 1.5, 'dash': 'solid'},
    'iv':   {'color': '#4ec9e7', 'width': 1.5, 'dash': 'solid'},
}

IVPCT_CONFIGS = {
    'iv_pct': {'color': '#b92c9f', 'width': 1.5, 'dash': 'solid'},
}

BB_CONFIGS = {
    'bb_upper': {'color': '#ffffff', 'width': 1.0, 'dash': 'dot'},
    'bb_lower': {'color': '#ffffff', 'width': 1.0, 'dash': 'dot'},
}

TTM_COLORS = {
    'pos_up':   '#769cc2',
    'pos_down': '#476cc2',
    'neg_down': '#9a0d9e',
    'neg_up':   '#be7ac2',
    'sq_on':    '#ec4533',
    'sq_off':   '#24ad54',
}

# ATR impulse ratio pane reference lines
ATR_RATIO_THRESHOLD = 1.75
ATR_RATIO_COLOR     = '#f5a623'


# ---------------------------------------------------------------------------
# Matplotlib multi-pane chart configs (finance/utils/plots.py)
# ---------------------------------------------------------------------------

MPL_EMA_CONFIGS = {
    'ema5': '#f5deb3', 'ema10': '#f5deb3', 'ema20': '#e2b46d',
    'ema50': '#c68e17', 'vwap3': '#00bfff',
}

MPL_ATR_CONFIGS = {
    'atrp9': '#b0c4de', 'atrp14': '#4682b4', 'atrp20': '#000080',
}

MPL_AC_CONFIGS = {
    'ac100_lag_1': '#e0ffff', 'ac100_lag_5': '#00ced1',
    'ac100_lag_10': '#00bfff', 'ac100_lag_20': '#008080',
}

MPL_AC_REGIME_CONFIGS = {
    'ac_mom': '#e1bee7', 'ac_mr': '#ba68c8', 'ac_comp': '#4a148c',
}

MPL_SLOPE_CONFIGS = {
    'ema10_slope': '#ffccbc', 'ema20_slope': '#ff8a65',
    'ema50_slope': '#ff5722', 'ema100_slope': '#e64a19',
    'ema200_slope': '#bf360c',
}

MPL_HURST_CONFIGS = {
    'hurst50': '#fff59d', 'hurst100': '#fbc02d',
}

MPL_HV_CONFIGS = {
    'hv9': '#a5d6a7', 'hv14': '#66bb6a', 'hv30': '#2e7d32', 'iv': '#ff00ff',
}