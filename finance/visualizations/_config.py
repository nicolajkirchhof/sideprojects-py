"""
finance.visualizations._config
===============================
Plot configuration constants shared across the swing plot modules.
"""
from pyqtgraph import QtCore

MA_CONFIGS = {
    'ma5':   {'color': '#9b7f6c', 'width': 1.0, 'style': QtCore.Qt.PenStyle.DashLine},
    'ma10':  {'color': '#9b7f6c', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},
    'ma20':  {'color': '#b26529', 'width': 1.5, 'style': QtCore.Qt.PenStyle.SolidLine},
    'ma50':  {'color': '#7b4326', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},
    'ma100': {'color': '#703b24', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},
    'ma200': {'color': '#5a2c27', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},
}

ATR_CONFIGS = {
    'atrp1':  {'color': '#f5a1df', 'width': 1.0, 'style': QtCore.Qt.PenStyle.DotLine},
    'atrp20': {'color': '#b72494', 'width': 1.5, 'style': QtCore.Qt.PenStyle.SolidLine},
}

SLOPE_CONFIGS = {
    'ma5_slope':   {'color': '#ffccbc', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},
    'ma10_slope':  {'color': '#ffccbc', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},
    'ma20_slope':  {'color': '#ff8a65', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},
    'ma50_slope':  {'color': '#ff5722', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},
    'ma200_slope': {'color': '#bf360c', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},
}

VOL_CONFIGS = {
    'v':   {'color': '#49bdd9', 'width': 1.0, 'style': QtCore.Qt.PenStyle.DashLine},
    'v20': {'color': '#f3cb21', 'width': 1.5, 'style': QtCore.Qt.PenStyle.SolidLine},
}

# ma100_dist / ma200_dist removed — not decision-relevant for 20-60 day swings
DIST_CONFIGS = {
    'ma10_dist':  {'color': '#9b7f6c', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},
    'ma20_dist':  {'color': '#b26529', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},
    'ma50_dist':  {'color': '#7b4326', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},
}

HV_CONFIGS = {
    'hv9':  {'color': '#b7a3db', 'width': 1.0, 'style': QtCore.Qt.PenStyle.DashLine},
    'hv20': {'color': '#6539b4', 'width': 1.5, 'style': QtCore.Qt.PenStyle.SolidLine},
    'iv':   {'color': '#49bcd8', 'width': 1.5, 'style': QtCore.Qt.PenStyle.SolidLine},
}

IVPCT_CONFIGS = {
    'iv_pct': {'color': '#b72494', 'width': 1.0, 'style': QtCore.Qt.PenStyle.SolidLine},
}

BB_CONFIGS = {
    'bb_upper': {'color': '#9075d6', 'width': 1.0, 'style': QtCore.Qt.PenStyle.DotLine},
    'bb_lower': {'color': '#00aaab', 'width': 1.0, 'style': QtCore.Qt.PenStyle.DotLine},
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
