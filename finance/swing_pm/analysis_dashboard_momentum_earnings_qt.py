from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

# Qt binding: prefer PySide6, fallback to PyQt5
from PySide6 import QtCore, QtGui, QtWidgets

import pyqtgraph as pg
import pyarrow.parquet as pq  # type: ignore

from finance import utils

# %load_ext autoreload
# %autoreload 2


#%%
# Global cache for reusing the window and app across multiple plot calls
_GLOBAL_QT_APP: Optional[QtWidgets.QApplication] = None
_GLOBAL_DASHBOARD_WIN: Optional["DashboardQt"] = None
_GLOBAL_LOADED_DF: Optional[pd.DataFrame] = None

# Run %gui qt only once (re-running it can cause odd behavior / CPU churn in some consoles)
_IPYTHON_GUI_QT_ENABLED = False

# Protect UI from massive scatter clouds
_MAX_SCATTER_POINTS_PER_PERIOD = 1500
_MAX_SCATTER_TOTAL_POINTS = 50000


def _in_ipython() -> bool:
  """Best-effort detection of an IPython environment."""
  try:
    from IPython import get_ipython  # type: ignore
    return get_ipython() is not None
  except Exception:
    return False


def _ensure_ipython_qt_event_loop() -> None:
  """
  In IPython/Jupyter, ensure the Qt event loop is integrated.
  Without this, Qt windows often appear but never repaint (white window).
  """
  global _IPYTHON_GUI_QT_ENABLED

  if _IPYTHON_GUI_QT_ENABLED:
    return
  if not _in_ipython():
    return

  try:
    from IPython import get_ipython  # type: ignore
    ip = get_ipython()
    if ip is None:
      return
    ip.run_line_magic("gui", "qt")  # equivalent to: %gui qt
    _IPYTHON_GUI_QT_ENABLED = True
  except Exception:
    return


@dataclass
class MinMaxSpin:
  label: QtWidgets.QLabel
  min_box: QtWidgets.QDoubleSpinBox
  max_box: QtWidgets.QDoubleSpinBox

  def value(self) -> tuple[float, float]:
    a = float(self.min_box.value())
    b = float(self.max_box.value())
    return (a, b) if a <= b else (b, a)


@dataclass
class MinMaxIntSpin:
  label: QtWidgets.QLabel
  min_box: QtWidgets.QSpinBox
  max_box: QtWidgets.QSpinBox

  def value(self) -> tuple[int, int]:
    a = int(self.min_box.value())
    b = int(self.max_box.value())
    return (a, b) if a <= b else (b, a)


def load_and_prep_data(years: range) -> pd.DataFrame:
  """
  Loads and standardizes the momentum/earnings dataset for the dashboard.

  Important: keep this function aligned with analysis_dashboard_momentum_earnings.py
  so that filters/plots behave the same across both UIs.
  """

  def _required_columns() -> list[str]:
    cols: set[str] = {
      # core
      "date",
      "original_price",
      "c0",
      "cpct0",
      "atrp200",
      "is_earnings",
      "is_etf",
      "spy0",
      "spy5",
      "market_cap_class",
      # event types (new tracking)
      "evt_atrp_breakout",
      "evt_green_line_breakout",
      # filters
      "1M_chg",
      "3M_chg",
      "6M_chg",
      "12M_chg",
      "ma10_dist0",
      "ma20_dist0",
      "ma50_dist0",
      "ma100_dist0",
      "ma200_dist0",
      "spy_ma10_dist0",
      "spy_ma20_dist0",
      "spy_ma50_dist0",
      "spy_ma100_dist0",
      "spy_ma200_dist0",
    }

    # Trajectory / dist / cond filter (daily + weekly)
    for i in range(1, 25):
      cols.add(f"cpct{i}")
    for i in range(1, 9):
      cols.add(f"w_cpct{i}")

    # Probability lines (daily + weekly)
    for i in range(1, 25):
      cols.add(f"ma5_dist{i}")
      cols.add(f"ma10_dist{i}")
      cols.add(f"ma20_dist{i}")
      cols.add(f"ma50_dist{i}")
    for i in range(1, 9):
      cols.add(f"w_ma5_dist{i}")
      cols.add(f"w_ma10_dist{i}")
      cols.add(f"w_ma20_dist{i}")
      cols.add(f"w_ma50_dist{i}")

    # Distribution-over-time options (daily + weekly)
    dist_metrics = [
      "ma5_slope",
      "ma10_slope",
      "ma20_slope",
      "ma50_slope",
      "rvol20",
      "hv20",
      "atrp20",
    ]
    for m in dist_metrics:
      for i in range(1, 25):
        cols.add(f"{m}{i}")
      for i in range(1, 9):
        cols.add(f"w_{m}{i}")

    return sorted(cols)

  required_cols = _required_columns()

  dfs: list[pd.DataFrame] = []
  for year in years:
    parquet_path = f"finance/_data/momentum_earnings/all_{year}.parquet"

    if not os.path.exists(parquet_path):
      print(f"WARNING: {parquet_path} not found. Skipping.")
      continue
    available = set(pq.ParquetFile(parquet_path).schema.names)
    cols_to_read = [c for c in required_cols if c in available]
    dfs.append(pd.read_parquet(parquet_path, columns=cols_to_read))

  if not dfs:
    return pd.DataFrame()

  df = pd.concat(dfs, ignore_index=True)

  # Match matplotlib dashboard: basic cleanup + safety caps
  if "original_price" in df.columns:
    df = df[df["original_price"] < 10e5]

  df = df.replace([np.inf, -np.inf], np.nan).infer_objects()

  if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

  # Match matplotlib dashboard: require c0 if present
  if "c0" in df.columns:
    df = df.dropna(subset=["c0"])

  # Prefer original_price for event_price when present; fallback to c0
  if "original_price" in df.columns and "c0" in df.columns:
    df["event_price"] = df["original_price"].where(df["original_price"].notna(), df["c0"])
  elif "c0" in df.columns:
    df["event_price"] = df["c0"]
  else:
    df["event_price"] = np.nan

  # Event move (signed): match matplotlib dashboard fallback behavior
  if "cpct0" in df.columns and "atrp200" in df.columns:
    df["event_move"] = df["cpct0"] / df["atrp200"]
  else:
    df["event_move"] = 0.0

  # Direction: match matplotlib dashboard
  df["direction"] = np.sign(df["event_move"])
  df["direction"] = df["direction"].replace(0, 1)

  # Basic booleans
  if "is_earnings" in df.columns:
    df["is_earnings"] = df["is_earnings"].fillna(False).astype(bool)
  else:
    df["is_earnings"] = False

  if "is_etf" in df.columns:
    df["is_etf"] = df["is_etf"].fillna(False).astype(bool)
  else:
    df["is_etf"] = False

  # New event flags may be absent in older files; default them to False
  for col in ("evt_atrp_breakout", "evt_green_line_breakout"):
    if col in df.columns:
      df[col] = df[col].fillna(False).astype(bool)
    else:
      df[col] = False

  # SPY Context (Simple alignment check over 5 days): match matplotlib dashboard
  if "spy0" in df.columns and "spy5" in df.columns:
    spy_change = df["spy5"] - df["spy0"]
    aligned_spy = spy_change * df["direction"]

    conditions = [
      aligned_spy > 0.5,  # Supporting
      aligned_spy < -0.5,  # Non-Supporting
    ]
    choices = ["Supporting", "Non-Supporting"]
    df["spy_class"] = np.select(conditions, choices, default="Neutral")
  else:
    df["spy_class"] = "Unknown"

  return df


class DashboardQt(QtWidgets.QMainWindow):
  def __init__(self, df: pd.DataFrame):
    super().__init__()
    self.df = df
    self._has_rendered_once = False
    self._last_sub_df: Optional[pd.DataFrame] = None

    # Persist conditional ranges per (tab, t)
    # key: ("Daily"|"Weekly", int t) -> (min_val, max_val)
    self._cond_saved_ranges: dict[tuple[str, int], tuple[float, float]] = {}
    self._last_cond_key: tuple[str, int] = ("Daily", 0)

    # --- Swing caches for prefilling ---
    self._spy_swing: Optional[object] = None
    self._spy_df_day: Optional[pd.DataFrame] = None
    self._underlying_swing: Optional[object] = None
    self._underlying_df_day: Optional[pd.DataFrame] = None
    self._underlying_symbol: Optional[str] = None
    self._underlying_market_cap: Optional[pd.DataFrame] = None

    self.setWindowTitle("Momentum & Earnings Dashboard (Qt)")
    self.resize(3400, 1900)
    self.setWindowState(self.windowState() | QtCore.Qt.WindowState.WindowMaximized)

    self.view_tab = "Daily"  # keep in sync with matplotlib dashboard behavior

    central = QtWidgets.QWidget()
    self.setCentralWidget(central)

    root = QtWidgets.QHBoxLayout(central)
    root.setContentsMargins(10, 10, 10, 10)
    root.setSpacing(10)

    splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
    root.addWidget(splitter)

    controls = QtWidgets.QWidget()
    splitter.addWidget(controls)
    splitter.setStretchFactor(0, 0)

    plots = QtWidgets.QWidget()
    splitter.addWidget(plots)
    splitter.setStretchFactor(1, 1)

    controls.setMinimumWidth(560)
    controls.setMaximumWidth(820)

    # ---- Controls UI ----
    c_layout = QtWidgets.QVBoxLayout(controls)
    c_layout.setContentsMargins(8, 8, 8, 8)
    c_layout.setSpacing(10)

    # ---- Data loader (Year range -> load files) ----
    load_box = QtWidgets.QGroupBox("Data")
    load_layout = QtWidgets.QGridLayout(load_box)
    load_layout.setHorizontalSpacing(10)
    load_layout.setVerticalSpacing(6)

    self.lbl_load_years = QtWidgets.QLabel("Load Years:")
    self.load_year_min = QtWidgets.QSpinBox()
    self.load_year_max = QtWidgets.QSpinBox()
    for b in (self.load_year_min, self.load_year_max):
      b.setRange(1900, 2100)
      b.setSingleStep(1)
      b.setAccelerated(True)

    this_year = int(pd.Timestamp.now().year)
    self.load_year_min.setValue(2012)
    self.load_year_max.setValue(this_year)

    self.btn_load = QtWidgets.QPushButton("Load")
    self.btn_load.setMinimumHeight(30)
    self.btn_load.clicked.connect(self._load_data_from_ui)

    self.lbl_data_status = QtWidgets.QLabel("")
    self.lbl_data_status.setStyleSheet("color: #AAAAAA;")

    load_layout.addWidget(self.lbl_load_years, 0, 0, 1, 1)
    load_layout.addWidget(self.load_year_min, 0, 1, 1, 1)
    load_layout.addWidget(self.load_year_max, 0, 2, 1, 1)
    load_layout.addWidget(self.btn_load, 1, 0, 1, 3)
    load_layout.addWidget(self.lbl_data_status, 2, 0, 1, 3)

    c_layout.addWidget(load_box)

    # ---- Swing Prefill (Underlying + SPY) ----
    prefill_box = QtWidgets.QGroupBox("Prefill from Swing Data")
    prefill_layout = QtWidgets.QGridLayout(prefill_box)
    prefill_layout.setHorizontalSpacing(10)
    prefill_layout.setVerticalSpacing(6)

    self.txt_underlying = QtWidgets.QLineEdit()
    self.txt_underlying.setPlaceholderText("Ticker (e.g. MSFT)")
    self.txt_underlying.setMaxLength(12)

    self.date_breakout = QtWidgets.QDateEdit()
    self.date_breakout.setCalendarPopup(True)
    self.date_breakout.setDisplayFormat("yyyy-MM-dd")
    self.date_breakout.setDate(QtCore.QDate.currentDate())

    self.btn_prefill = QtWidgets.QPushButton("Load + Prefill")
    self.btn_prefill.setMinimumHeight(30)
    self.btn_prefill.clicked.connect(self._prefill_from_ui)

    self.lbl_prefill_status = QtWidgets.QLabel("")
    self.lbl_prefill_status.setStyleSheet("color: #AAAAAA;")

    prefill_layout.addWidget(QtWidgets.QLabel("Underlying:"), 0, 0, 1, 1)
    prefill_layout.addWidget(self.txt_underlying, 0, 1, 1, 2)
    prefill_layout.addWidget(QtWidgets.QLabel("Breakout Day:"), 1, 0, 1, 1)
    prefill_layout.addWidget(self.date_breakout, 1, 1, 1, 2)
    prefill_layout.addWidget(self.btn_prefill, 2, 0, 1, 3)
    prefill_layout.addWidget(self.lbl_prefill_status, 3, 0, 1, 3)

    c_layout.addWidget(prefill_box)

    title = QtWidgets.QLabel("Filters")
    title.setStyleSheet("font-size: 16px; font-weight: 600;")
    c_layout.addWidget(title)

    # --- Distribution selector (replots middle panel on change) ---
    dist_box = QtWidgets.QGroupBox("Distribution View")
    dist_layout = QtWidgets.QHBoxLayout(dist_box)

    self.cmb_dist = QtWidgets.QComboBox()
    self.cmb_dist.addItem("Price Change (cpct1..N)", userData="cpct")
    self.cmb_dist.addItem("MA5 slope (1..N)", userData="ma5_slope")
    self.cmb_dist.addItem("MA10 slope (1..N)", userData="ma10_slope")
    self.cmb_dist.addItem("MA20 slope (1..N)", userData="ma20_slope")
    self.cmb_dist.addItem("MA50 slope (1..N)", userData="ma50_slope")
    self.cmb_dist.addItem("RVOL20 (1..N)", userData="rvol20")
    self.cmb_dist.addItem("HV20 (1..N)", userData="hv20")
    self.cmb_dist.addItem("ATRP20 (1..N)", userData="atrp20")
    self.cmb_dist.setCurrentIndex(0)

    def _on_dist_change(_idx: int) -> None:
      # Recalculate distribution plot immediately using last filtered subset (no need to hit Update)
      if (not self._has_rendered_once) or (self._last_sub_df is None):
        return
      self._plot(self._last_sub_df)

    self.cmb_dist.currentIndexChanged.connect(_on_dist_change)

    dist_layout.addWidget(QtWidgets.QLabel("Show:"))
    dist_layout.addWidget(self.cmb_dist, 1)
    c_layout.addWidget(dist_box)

    # --- View Tab (Daily / Weekly) ---
    tab_row = QtWidgets.QHBoxLayout()
    c_layout.addLayout(tab_row)
    self.btn_tab_d = QtWidgets.QPushButton("Daily")
    self.btn_tab_w = QtWidgets.QPushButton("Weekly")
    self.btn_tab_d.setCheckable(True)
    self.btn_tab_w.setCheckable(True)
    self.btn_tab_d.setChecked(True)

    def _set_tab(name: str) -> None:
      # Save current conditional edits under the old key before switching tab
      self._save_cond_range_for_key(self._last_cond_key)

      self.view_tab = name
      self.btn_tab_d.setChecked(name == "Daily")
      self.btn_tab_w.setChecked(name == "Weekly")

      # New active key after tab switch
      self._last_cond_key = (self.view_tab, int(self.slider_cond_t.value()))

      self._update_cond_label_and_bounds()
      # IMPORTANT: do NOT auto-render on tab change.
      # User wants the dashboard to stay empty until Update is clicked.

    self.btn_tab_d.clicked.connect(lambda _=False: _set_tab("Daily"))
    self.btn_tab_w.clicked.connect(lambda _=False: _set_tab("Weekly"))
    tab_row.addWidget(self.btn_tab_d)
    tab_row.addWidget(self.btn_tab_w)

    # Grid with a single "min/max" header row (requested)
    grid = QtWidgets.QGridLayout()
    grid.setHorizontalSpacing(10)
    grid.setVerticalSpacing(8)
    c_layout.addLayout(grid)

    grid.addWidget(QtWidgets.QLabel(""), 0, 0)
    hdr_min = QtWidgets.QLabel("min")
    hdr_max = QtWidgets.QLabel("max")
    hdr_min.setStyleSheet("color: #AAAAAA;")
    hdr_max.setStyleSheet("color: #AAAAAA;")
    grid.addWidget(hdr_min, 0, 1)
    grid.addWidget(hdr_max, 0, 2)

    self._row = 1

    # Event Move (signed)
    self.event_move = self._add_range(grid, "Event Move (Signed)", df, col="event_move", decimals=3, step=0.25)

    # Breakout Price
    self.event_price = self._add_range(grid, "Breakout Price", df, col="event_price", decimals=2, step=0.5)

    # Momentum Filters (added 12M_chg)
    self.mom_1m = self._add_range(grid, "1M_chg", df, col="1M_chg", decimals=1, step=1.0)
    self.mom_3m = self._add_range(grid, "3M_chg", df, col="3M_chg", decimals=1, step=1.0)
    self.mom_6m = self._add_range(grid, "6M_chg", df, col="6M_chg", decimals=1, step=1.0)
    self.mom_12m = self._add_range(grid, "12M_chg", df, col="12M_chg", decimals=1, step=1.0)

    # 9-13. Underlying ma dist0 filters
    self.ma_filters: dict[str, MinMaxSpin] = {}
    for ma_name in ["ma10", "ma20", "ma50", "ma100", "ma200"]:
      col = f"{ma_name}_dist0"
      self.ma_filters[col] = self._add_range(grid, f"{ma_name} Dist", df, col=col, decimals=2, step=0.25)

    # 14-18. SPY ma dist0 filters
    self.spy_ma_filters: dict[str, MinMaxSpin] = {}
    for ma_name in ["ma10", "ma20", "ma50", "ma100", "ma200"]:
      col = f"spy_{ma_name}_dist0"
      self.spy_ma_filters[col] = self._add_range(grid, f"SPY {ma_name} Dist", df, col=col, decimals=2, step=0.25)

    # Instrument Type (NEW): ETFs vs Stocks
    inst_box = QtWidgets.QGroupBox("Instrument")
    inst_layout = QtWidgets.QHBoxLayout(inst_box)
    self.inst_all = QtWidgets.QRadioButton("Both")
    self.inst_stocks = QtWidgets.QRadioButton("Stocks")
    self.inst_etfs = QtWidgets.QRadioButton("ETFs")
    self.inst_all.setChecked(True)
    for w in (self.inst_all, self.inst_stocks, self.inst_etfs):
      inst_layout.addWidget(w)
    c_layout.addWidget(inst_box)

    # Event Types (NEW): individual toggles
    evt_box = QtWidgets.QGroupBox("Events")
    evt_layout = QtWidgets.QHBoxLayout(evt_box)
    self.evt_earnings = QtWidgets.QCheckBox("Earnings")
    self.evt_atrp = QtWidgets.QCheckBox("ATRP")
    self.evt_green = QtWidgets.QCheckBox("GreenLine")

    # Default: ATRP checked (requested)
    self.evt_atrp.setChecked(True)

    # Default: none checked = no event-type filtering (show all)
    for w in (self.evt_earnings, self.evt_atrp, self.evt_green):
      evt_layout.addWidget(w)
    evt_layout.addStretch(1)
    c_layout.addWidget(evt_box)


    # SPY Context
    spy_box = QtWidgets.QGroupBox("SPY Context")
    spy_layout = QtWidgets.QHBoxLayout(spy_box)
    self.spy_all = QtWidgets.QRadioButton("All")
    self.spy_support = QtWidgets.QRadioButton("Supporting")
    self.spy_neutral = QtWidgets.QRadioButton("Neutral")
    self.spy_nonsupport = QtWidgets.QRadioButton("Non-Supporting")
    self.spy_all.setChecked(True)
    for w in (self.spy_all, self.spy_support, self.spy_neutral, self.spy_nonsupport):
      spy_layout.addWidget(w)
    c_layout.addWidget(spy_box)

    # Market Cap
    mcap_box = QtWidgets.QGroupBox("Market Cap")
    mcap_layout = QtWidgets.QHBoxLayout(mcap_box)
    self.mcap_group = QtWidgets.QButtonGroup(self)
    self.mcap_buttons: dict[str, QtWidgets.QRadioButton] = {}
    for name in ["All", "Large", "Mid", "Small", "Micro"]:
      rb = QtWidgets.QRadioButton(name)
      self.mcap_group.addButton(rb)
      self.mcap_buttons[name] = rb
      mcap_layout.addWidget(rb)
    self.mcap_buttons["All"].setChecked(True)
    c_layout.addWidget(mcap_box)

    # Conditional Survival
    cond_box = QtWidgets.QGroupBox("Conditional Survival")
    cond_layout = QtWidgets.QGridLayout(cond_box)
    cond_layout.setHorizontalSpacing(10)
    cond_layout.setVerticalSpacing(6)

    self.lbl_cond_t = QtWidgets.QLabel("Cond. Day/Wk:")
    self.slider_cond_t = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
    self.slider_cond_t.setRange(0, 24)
    self.slider_cond_t.setSingleStep(1)
    self.slider_cond_t.setValue(0)

    # Show the current slider value next to the slider
    self.lbl_cond_val = QtWidgets.QLabel("0")
    self.lbl_cond_val.setStyleSheet("color: #AAAAAA; font-size: 11px;")
    self.lbl_cond_val.setFixedWidth(34)
    self.lbl_cond_val.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)

    self.btn_cond_reset = QtWidgets.QPushButton("Reset")
    self.btn_cond_reset.setToolTip("Reset conditional range for the selected period to the full data range")
    self.btn_cond_reset.setMinimumHeight(22)
    self.btn_cond_reset.setFixedWidth(70)

    self.cond_range = self._add_range(
      grid=None, title="Cond. Range", df=df, col="__cond_placeholder__", decimals=1, step=0.5,
      add_to_grid=False
    )

    cond_layout.addWidget(self.lbl_cond_t, 0, 0, 1, 1)
    cond_layout.addWidget(self.slider_cond_t, 0, 1, 1, 2)
    cond_layout.addWidget(self.lbl_cond_val, 0, 3, 1, 1)
    cond_layout.addWidget(self.btn_cond_reset, 0, 4, 1, 1)

    cond_layout.addWidget(self.cond_range.label, 1, 0, 1, 1)
    cond_layout.addWidget(self.cond_range.min_box, 1, 1, 1, 1)
    cond_layout.addWidget(self.cond_range.max_box, 1, 2, 1, 1)

    # Keep saved conditional values in sync when user edits min/max
    self.cond_range.min_box.valueChanged.connect(lambda _v: self._save_current_cond_range())
    self.cond_range.max_box.valueChanged.connect(lambda _v: self._save_current_cond_range())

    # Slider handler that saves the PREVIOUS key correctly, then updates bounds/label
    self._last_cond_key = (self.view_tab, int(self.slider_cond_t.value()))
    self.slider_cond_t.valueChanged.connect(self._on_cond_t_changed)

    self.btn_cond_reset.clicked.connect(self._reset_current_cond_range_to_full)

    c_layout.addWidget(cond_box)

    # ---- Breakout Day Snapshot (Underlying) ----
    snap_box = QtWidgets.QGroupBox("Breakout Day Snapshot")
    snap_layout = QtWidgets.QGridLayout(snap_box)
    snap_layout.setHorizontalSpacing(10)
    snap_layout.setVerticalSpacing(4)

    self.lbl_snap_symbol = QtWidgets.QLabel("Symbol: —")
    self.lbl_snap_date = QtWidgets.QLabel("Date: —")
    self.lbl_snap_close = QtWidgets.QLabel("Close: —")
    self.lbl_snap_orig = QtWidgets.QLabel("Original Price: —")
    self.lbl_snap_mcap = QtWidgets.QLabel("Market Cap Class: —")
    for w in (self.lbl_snap_symbol, self.lbl_snap_date, self.lbl_snap_close, self.lbl_snap_orig, self.lbl_snap_mcap):
      w.setStyleSheet("color: #AAAAAA;")

    snap_layout.addWidget(self.lbl_snap_symbol, 0, 0, 1, 3)
    snap_layout.addWidget(self.lbl_snap_date, 1, 0, 1, 3)
    snap_layout.addWidget(self.lbl_snap_close, 2, 0, 1, 3)
    snap_layout.addWidget(self.lbl_snap_orig, 3, 0, 1, 3)
    snap_layout.addWidget(self.lbl_snap_mcap, 4, 0, 1, 3)

    self.lbl_snap_mas: dict[str, QtWidgets.QLabel] = {}
    r = 5
    for ma_name in ["ma10", "ma20", "ma50", "ma100", "ma200"]:
      lbl = QtWidgets.QLabel(f"{ma_name}_dist: —")
      lbl.setStyleSheet("color: #AAAAAA;")
      self.lbl_snap_mas[ma_name] = lbl
      snap_layout.addWidget(lbl, r, 0, 1, 3)
      r += 1

    c_layout.addWidget(snap_box)

    # Update button (filtering decoupled)
    btn_row = QtWidgets.QHBoxLayout()
    c_layout.addLayout(btn_row)

    self.btn_apply = QtWidgets.QPushButton("Update")
    self.btn_apply.setMinimumHeight(34)
    self.btn_apply.clicked.connect(self.apply_filters)
    btn_row.addWidget(self.btn_apply)

    self.lbl_status = QtWidgets.QLabel("")
    self.lbl_status.setStyleSheet("color: #AAAAAA;")
    btn_row.addWidget(self.lbl_status, 1)

    c_layout.addStretch(1)

    # ---- Plots (pyqtgraph) ----
    p_layout = QtWidgets.QVBoxLayout(plots)
    p_layout.setContentsMargins(0, 0, 0, 0)
    p_layout.setSpacing(8)

    pg.setConfigOptions(antialias=True)

    self.plot_path = pg.PlotWidget()
    self.plot_dist = pg.PlotWidget()
    self.plot_probs = pg.PlotWidget()

    self.plot_path.setMouseEnabled(x=False, y=True)
    self.plot_dist.setMouseEnabled(x=False, y=True)
    self.plot_probs.setMouseEnabled(x=False, y=True)
    self.plot_dist.setXLink(self.plot_path)
    self.plot_dist.setYLink(self.plot_path)

    for pw in (self.plot_path, self.plot_dist, self.plot_probs):
      pw.setBackground("k")
      pw.showGrid(x=True, y=True, alpha=0.2)
      pw.getPlotItem().getAxis("left").setPen(pg.mkPen("#AAAAAA"))
      pw.getPlotItem().getAxis("bottom").setPen(pg.mkPen("#AAAAAA"))
      pw.getPlotItem().getAxis("left").setTextPen(pg.mkPen("#AAAAAA"))
      pw.getPlotItem().getAxis("bottom").setTextPen(pg.mkPen("#AAAAAA"))

    self.plot_path.setTitle("Trajectory (5th/50th/95th Quantiles)", color="#DDDDDD", size="12pt")
    self.plot_dist.setTitle("Distribution of Changes", color="#DDDDDD", size="12pt")
    self.plot_probs.setTitle("Probability of Holding Levels", color="#DDDDDD", size="12pt")

    p_layout.addWidget(self.plot_path, 1)
    p_layout.addWidget(self.plot_dist, 1)
    p_layout.addWidget(self.plot_probs, 1)

    # Load SPY swing data upfront so prefills are instant.
    self._load_spy_swing_initial()

    # Initialize UI bounds, but DO NOT render any data yet.
    self._update_cond_label_and_bounds()
    self._show_empty_state()
    self._update_data_status()

  def _load_spy_swing_initial(self) -> None:
    try:
      spy = utils.swing_trading_data.SwingTradingData("SPY", offline=True, metainfo=False)
    except Exception as e:
      self._spy_swing = None
      self._spy_df_day = None
      self.lbl_prefill_status.setText(f"SPY swing load failed: {e}")
      return

    if getattr(spy, "empty", True) or (getattr(spy, "df_day", None) is None) or spy.df_day.empty:
      self._spy_swing = None
      self._spy_df_day = None
      self.lbl_prefill_status.setText("SPY swing data is empty.")
      return

    self._spy_swing = spy
    self._spy_df_day = spy.df_day

  def _qdate_to_ts(self, qd: QtCore.QDate) -> pd.Timestamp:
    # Use midnight, naive timestamp; must match df_day index exactly per requirement.
    return pd.Timestamp(year=qd.year(), month=qd.month(), day=qd.day()).normalize()

  def _set_spin_band_pm25(self, w: MinMaxSpin, center: float, *, decimals: Optional[int] = None) -> None:
    if not np.isfinite(center):
      return
    lo = float(center) * 0.75
    hi = float(center) * 1.25
    if lo > hi:
      lo, hi = hi, lo

    # Expand allowed range if needed; otherwise Qt clamps silently.
    cur_lo = float(w.min_box.minimum())
    cur_hi = float(w.max_box.maximum())
    new_lo = min(cur_lo, lo)
    new_hi = max(cur_hi, hi)

    for b in (w.min_box, w.max_box):
      b.blockSignals(True)
      b.setRange(new_lo, new_hi)
      if decimals is not None:
        b.setDecimals(int(decimals))
      b.blockSignals(False)

    w.min_box.blockSignals(True)
    w.max_box.blockSignals(True)
    w.min_box.setValue(lo)
    w.max_box.setValue(hi)
    w.min_box.blockSignals(False)
    w.max_box.blockSignals(False)

  def _set_market_cap_radio(self, mcap_class: str) -> None:
    val = str(mcap_class or "").strip()
    if val not in self.mcap_buttons:
      val = "All"
    self.mcap_buttons[val].setChecked(True)

  def _prefill_from_ui(self) -> None:
    sym = (self.txt_underlying.text() or "").strip().upper()
    if not sym:
      QtWidgets.QMessageBox.critical(self, "Prefill Error", "Please enter an underlying ticker.")
      return

    if self._spy_df_day is None or self._spy_df_day.empty:
      QtWidgets.QMessageBox.critical(self, "Prefill Error", "SPY swing data is not loaded.")
      return

    breakout_ts = self._qdate_to_ts(self.date_breakout.date())

    # Load underlying swing with metainfo=True so market cap class is available
    self.lbl_prefill_status.setText("Loading swing data…")
    QtWidgets.QApplication.processEvents()

    try:
      und = utils.swing_trading_data.SwingTradingData(sym, offline=True)
    except Exception as e:
      QtWidgets.QMessageBox.critical(self, "Prefill Error", f"Failed to load swing data for {sym}: {e}")
      self.lbl_prefill_status.setText("")
      return

    if getattr(und, "empty", True) or (getattr(und, "df_day", None) is None) or und.df_day.empty:
      QtWidgets.QMessageBox.critical(self, "Prefill Error", f"Swing data is empty for {sym}.")
      self.lbl_prefill_status.setText("")
      return

    df_u = und.df_day
    df_spy = self._spy_df_day

    # Requirement: error if the trading day does not exist (no nearest/ffill)
    if breakout_ts not in df_u.index:
      QtWidgets.QMessageBox.critical(
        self,
        "Prefill Error",
        f"{sym} has no trading day at {breakout_ts.date()}. Pick an actual trading day.",
      )
      self.lbl_prefill_status.setText("")
      return
    if breakout_ts not in df_spy.index:
      QtWidgets.QMessageBox.critical(
        self,
        "Prefill Error",
        f"SPY has no trading day at {breakout_ts.date()}. Pick an actual trading day.",
      )
      self.lbl_prefill_status.setText("")
      return

    self._underlying_swing = und
    self._underlying_df_day = df_u
    self._underlying_symbol = sym
    self._underlying_market_cap = getattr(und, "market_cap", None)

    row_u = df_u.loc[breakout_ts]
    row_spy = df_spy.loc[breakout_ts]

    # --- Prefill breakout price (band around original_price if available else close) ---
    px_col = "original_price" if ("original_price" in df_u.columns and np.isfinite(float(row_u.get("original_price", np.nan)))) else "c"
    px = float(row_u.get(px_col, np.nan))
    self._set_spin_band_pm25(self.event_price, px, decimals=2)

    # --- Prefill event move (chg / ATR$20) ---
    # ATR$20 = (atrp20% / 100) * close. Then xATR = chg / ATR$20
    chg = float(row_u.get("chg", np.nan))
    atrp20 = float(row_u.get("atrp20", np.nan))
    c = float(row_u.get("c", np.nan))
    atr20_dollars = (atrp20 / 100.0) * c if (np.isfinite(atrp20) and np.isfinite(c)) else np.nan
    ev_move = (chg / atr20_dollars) if (np.isfinite(chg) and np.isfinite(atr20_dollars) and atr20_dollars != 0.0) else np.nan
    self._set_spin_band_pm25(self.event_move, ev_move, decimals=3)

    # --- Prefill momentum filters from df_day (1M_chg / 3M_chg / 6M_chg / 12M_chg) ---
    for w, col in (
      (self.mom_1m, "1M_chg"),
      (self.mom_3m, "3M_chg"),
      (self.mom_6m, "6M_chg"),
      (self.mom_12m, "12M_chg"),
    ):
      v = float(row_u.get(col, np.nan))
      self._set_spin_band_pm25(w, v, decimals=1)

    # --- Prefill MA dist0 filters from swing ma*_dist at breakout day ---
    for ma_name in ["ma10", "ma20", "ma50", "ma100", "ma200"]:
      swing_col = f"{ma_name}_dist"  # e.g. ma10_dist
      ui_col_u = f"{ma_name}_dist0"
      ui_col_spy = f"spy_{ma_name}_dist0"

      if (ui_col_u in self.ma_filters) and (swing_col in df_u.columns):
        v = float(row_u.get(swing_col, np.nan))
        self._set_spin_band_pm25(self.ma_filters[ui_col_u], v, decimals=2)

      if (ui_col_spy in self.spy_ma_filters) and (swing_col in df_spy.columns):
        v_spy = float(row_spy.get(swing_col, np.nan))
        self._set_spin_band_pm25(self.spy_ma_filters[ui_col_spy], v_spy, decimals=2)

    # --- Prefill market cap class from underlying market_cap (nearest available fundamental record) ---
    mcap_class = "All"
    mc = self._underlying_market_cap
    if isinstance(mc, pd.DataFrame) and (not mc.empty) and ("market_cap_class" in mc.columns):
      try:
        idx = mc.index.get_indexer([breakout_ts], method="nearest")
        if len(idx) and idx[0] >= 0:
          mcap_class = str(mc.iloc[int(idx[0])].get("market_cap_class", "All") or "All")
      except Exception:
        mcap_class = "All"
    self._set_market_cap_radio(mcap_class)

    # --- Update bottom snapshot ---
    self._update_breakout_snapshot(sym, breakout_ts, row_u, mcap_class)

    self.lbl_prefill_status.setText(f"Prefilled from {sym} @ {breakout_ts.date()}")

  def _update_breakout_snapshot(self, sym: str, ts: pd.Timestamp, row_u: pd.Series, mcap_class: str) -> None:
    self.lbl_snap_symbol.setText(f"Symbol: {sym}")
    self.lbl_snap_date.setText(f"Date: {ts.date()}")

    c = row_u.get("c", np.nan)
    self.lbl_snap_close.setText(f"Close: {float(c):.2f}" if np.isfinite(float(c)) else "Close: —")

    op = row_u.get("original_price", np.nan)
    self.lbl_snap_orig.setText(f"Original Price: {float(op):.2f}" if np.isfinite(float(op)) else "Original Price: —")

    self.lbl_snap_mcap.setText(f"Market Cap Class: {mcap_class if mcap_class else '—'}")

    for ma_name, lbl in self.lbl_snap_mas.items():
      col = f"{ma_name}_dist"
      v = row_u.get(col, np.nan)
      lbl.setText(f"{col}: {float(v):.2f}" if np.isfinite(float(v)) else f"{col}: —")

  def _update_data_status(self) -> None:
    n = 0 if self.df is None else int(len(self.df))
    years_str = "—"
    if self.df is not None and (not self.df.empty) and ("date" in self.df.columns):
      yy = self.df["date"].dt.year.dropna()
      if not yy.empty:
        years_str = f"{int(yy.min())}–{int(yy.max())}"
    self.lbl_data_status.setText(f"Loaded: {years_str}   N={n:,}")

  def _refresh_filter_bounds_from_df(self) -> None:
    """
    After loading new data, refresh the spinbox ranges/values so filters match the new DF.
    Keeps it simple: reset each filter to the full available data range.
    """
    df = self.df
    if df is None or df.empty:
      return

    # Clear conditional cache because ranges are tied to the loaded df
    self._cond_saved_ranges.clear()

    def _reset_from_series(w: MinMaxSpin, s: pd.Series) -> None:
      s = s.dropna()
      if s.empty:
        return
      a = float(np.nanmin(s.to_numpy(dtype=float)))
      b = float(np.nanmax(s.to_numpy(dtype=float)))
      if not np.isfinite(a) or not np.isfinite(b):
        return
      if a > b:
        a, b = b, a
      for box, val in ((w.min_box, a), (w.max_box, b)):
        box.blockSignals(True)
        box.setRange(a, b)
        box.setValue(val)
        box.blockSignals(False)

    # Simple numeric filters
    if "event_move" in df.columns:
      _reset_from_series(self.event_move, df["event_move"])
    if "event_price" in df.columns:
      _reset_from_series(self.event_price, df["event_price"])

    for w, col in (
        (self.mom_1m, "1M_chg"),
        (self.mom_3m, "3M_chg"),
        (self.mom_6m, "6M_chg"),
        (self.mom_12m, "12M_chg"),
    ):
      if col in df.columns:
        _reset_from_series(w, df[col])

    for col, w in self.ma_filters.items():
      if col in df.columns:
        _reset_from_series(w, df[col])

    for col, w in self.spy_ma_filters.items():
      if col in df.columns:
        _reset_from_series(w, df[col])

    self._update_cond_label_and_bounds()

  def _load_data_from_ui(self) -> None:
    global _GLOBAL_LOADED_DF
    y0 = int(self.load_year_min.value())
    y1 = int(self.load_year_max.value())
    if y0 > y1:
      y0, y1 = y1, y0

    self.lbl_data_status.setText("Loading…")
    QtWidgets.QApplication.processEvents()

    df_new = load_and_prep_data(range(y0, y1 + 1))
    if df_new is None or df_new.empty:
      self.df = pd.DataFrame()
      _GLOBAL_LOADED_DF = self.df
      self._show_empty_state()
      self._update_data_status()
      self.lbl_status.setText("No data loaded for that range.")
      return

    self.df = df_new
    _GLOBAL_LOADED_DF = df_new

    self._refresh_filter_bounds_from_df()
    self._show_empty_state()
    self._update_data_status()
    self.lbl_status.setText("Loaded. Click Update to render.")

  def _show_empty_state(self) -> None:
    self.plot_path.clear()
    self.plot_dist.clear()
    self.plot_probs.clear()

    msg = pg.TextItem("Click Update to render", color="#DDDDDD", anchor=(0.5, 0.5))
    self.plot_path.addItem(msg)

    msg2 = pg.TextItem("No plot yet", color="#DDDDDD", anchor=(0.5, 0.5))
    self.plot_dist.addItem(msg2)

    msg3 = pg.TextItem("No plot yet", color="#DDDDDD", anchor=(0.5, 0.5))
    self.plot_probs.addItem(msg3)

    self.lbl_status.setText("Idle (no render yet).")

  def closeEvent(self, event: QtGui.QCloseEvent) -> None:
    global _GLOBAL_DASHBOARD_WIN
    try:
      _GLOBAL_DASHBOARD_WIN = None
    finally:
      super().closeEvent(event)

  def _cond_col_name_for(self, tab: str, t: int) -> str:
    return (f"cpct{t}" if tab == "Daily" else f"w_cpct{t}")

  def _cond_col_name(self, t: int) -> str:
    return self._cond_col_name_for(self.view_tab, t)

  def _save_cond_range_for_key(self, key: tuple[str, int]) -> None:
    tab, t = key
    if t <= 0:
      return
    a, b = self.cond_range.value()
    self._cond_saved_ranges[(tab, t)] = (float(a), float(b))

  def _save_current_cond_range(self) -> None:
    self._save_cond_range_for_key((self.view_tab, int(self.slider_cond_t.value())))

  def _on_cond_t_changed(self, new_t: int) -> None:
    # Save edits for the previous (tab, t) before switching to new_t
    self._save_cond_range_for_key(self._last_cond_key)

    self._last_cond_key = (self.view_tab, int(new_t))
    self._update_cond_label_and_bounds()

  def _reset_current_cond_range_to_full(self) -> None:
    t = int(self.slider_cond_t.value())
    if t <= 0:
      return
    col = self._cond_col_name_for(self.view_tab, t)
    if col not in self.df.columns:
      return

    vals = self.df[col].dropna()
    if vals.empty:
      return

    vmin = float(vals.min())
    vmax = float(vals.max())
    if vmin >= vmax:
      vmax = vmin + 1.0

    for b in (self.cond_range.min_box, self.cond_range.max_box):
      b.blockSignals(True)
      b.setRange(vmin, vmax)
      b.setSingleStep(0.5)

    self.cond_range.min_box.setValue(vmin)
    self.cond_range.max_box.setValue(vmax)

    for b in (self.cond_range.min_box, self.cond_range.max_box):
      b.blockSignals(False)

    self._cond_saved_ranges[(self.view_tab, t)] = (vmin, vmax)
    self._update_cond_label_and_bounds()

  def _update_cond_label_and_bounds(self) -> None:
    # Label and maximum depend on tab (Daily: 24, Weekly: 8)
    max_t = 24 if self.view_tab == "Daily" else 8
    self.slider_cond_t.setMaximum(max_t)

    t = int(self.slider_cond_t.value())

    # Show current slider value in UI
    self.lbl_cond_val.setText(str(t))
    self.lbl_cond_t.setText(("Cond. Day:" if self.view_tab == "Daily" else "Cond. Wk:") + f"  (0–{max_t})")

    if t <= 0:
      # Keep label informative even when disabled
      self.cond_range.label.setText("Cond. Range (disabled):")
      return

    col = self._cond_col_name_for(self.view_tab, t)
    if col in self.df.columns:
      vals = self.df[col].dropna()
      if not vals.empty:
        vmin = float(vals.min())
        vmax = float(vals.max())
      else:
        vmin, vmax = -20.0, 50.0
    else:
      vmin, vmax = -20.0, 50.0

    if vmin >= vmax:
      vmax = vmin + 1.0

    # Update the Cond. Range label on EVERY slider change
    self.cond_range.label.setText(f"Cond. Range  [{vmin:.1f} … {vmax:.1f}]:")

    saved = self._cond_saved_ranges.get((self.view_tab, t), None)
    if saved is not None:
      cur_min, cur_max = saved
      cur_min = float(np.clip(cur_min, vmin, vmax))
      cur_max = float(np.clip(cur_max, vmin, vmax))
      if cur_min > cur_max:
        cur_min, cur_max = cur_max, cur_min
    else:
      cur_min, cur_max = vmin, vmax
      self._cond_saved_ranges[(self.view_tab, t)] = (cur_min, cur_max)

    # Update allowed bounds always, but restore saved/current values
    for b in (self.cond_range.min_box, self.cond_range.max_box):
      b.blockSignals(True)
      b.setRange(vmin, vmax)
      b.setSingleStep(0.5)

    self.cond_range.min_box.setValue(cur_min)
    self.cond_range.max_box.setValue(cur_max)

    for b in (self.cond_range.min_box, self.cond_range.max_box):
      b.blockSignals(False)

  # ---------- Controls builders ----------
  def _add_year_range(self, grid: QtWidgets.QGridLayout, title: str, df: pd.DataFrame) -> MinMaxSpin:
    years = df["date"].dt.year.dropna() if ("date" in df.columns) else pd.Series([], dtype=float)
    y0 = int(years.min()) if not years.empty else 2010
    y1 = int(years.max()) if not years.empty else pd.Timestamp.now().year

    # Ensure a sensible default starting point but respect data
    y_start = max(y0, 2010)

    label = QtWidgets.QLabel(f"{title}:")
    min_box = QtWidgets.QDoubleSpinBox()
    max_box = QtWidgets.QDoubleSpinBox()

    for b in (min_box, max_box):
      b.setDecimals(0)
      b.setRange(1900, 2100)
      b.setSingleStep(1)
      b.setKeyboardTracking(False)
      b.setAccelerated(True)

    # Match matplotlib defaults: start at 2010 (or data min), end at max year
    min_box.setValue(y_start)
    max_box.setValue(y1)

    grid.addWidget(label, self._row, 0)
    grid.addWidget(min_box, self._row, 1)
    grid.addWidget(max_box, self._row, 2)
    self._row += 1

    return MinMaxSpin(label, min_box, max_box)

  def _add_range(
      self,
      grid: Optional[QtWidgets.QGridLayout],
      title: str,
      df: pd.DataFrame,
      col: str,
      decimals: int,
      step: float,
      *,
      add_to_grid: bool = True,
  ) -> MinMaxSpin:
    # Data-derived bounds (available range)
    series = df[col].dropna() if (col in df.columns) else pd.Series([], dtype=float)
    if not series.empty:
      hard_min = float(series.min())
      hard_max = float(series.max())
    else:
      hard_min, hard_max = -1e9, 1e9

    # Show static data bounds behind the label
    if np.isfinite(hard_min) and np.isfinite(hard_max):
      label_txt = f"{title}  [{hard_min:.{decimals}f} … {hard_max:.{decimals}f}]:"
    else:
      label_txt = f"{title}:"

    label = QtWidgets.QLabel(label_txt)
    label.setStyleSheet("color: #DDDDDD; font-size: 11px;")

    min_box = QtWidgets.QDoubleSpinBox()
    max_box = QtWidgets.QDoubleSpinBox()

    # Smaller inputs
    for b in (min_box, max_box):
      b.setDecimals(decimals)
      b.setRange(hard_min, hard_max)
      b.setSingleStep(step)
      b.setKeyboardTracking(False)
      b.setAccelerated(True)
      b.setFixedWidth(140)
      b.setMinimumHeight(22)
      b.setStyleSheet("font-size: 11px; padding: 1px 4px;")

    # Set initial values to the calculated range
    min_box.setValue(hard_min)
    max_box.setValue(hard_max)

    if add_to_grid:
      if grid is None:
        raise ValueError("grid must be provided when add_to_grid=True")
      grid.addWidget(label, self._row, 0)
      grid.addWidget(min_box, self._row, 1)
      grid.addWidget(max_box, self._row, 2)
      self._row += 1

    return MinMaxSpin(label, min_box, max_box)

  # ---------- Filtering + plotting ----------
  def apply_filters(self) -> None:
    t0 = time.perf_counter()

    df = self.df
    if df.empty:
      self.lbl_status.setText("No data loaded.")
      self._show_empty_state()
      return

    mask = pd.Series(True, index=df.index)

    # Event move + price
    ev_min, ev_max = self.event_move.value()
    if "event_move" in df.columns:
      mask &= (df["event_move"] >= ev_min) & (df["event_move"] <= ev_max)

    p_min, p_max = self.event_price.value()
    if "event_price" in df.columns:
      mask &= (df["event_price"] >= p_min) & (df["event_price"] <= p_max)

    # Momentum (include 12M)
    for w, col in (
        (self.mom_1m, "1M_chg"),
        (self.mom_3m, "3M_chg"),
        (self.mom_6m, "6M_chg"),
        (self.mom_12m, "12M_chg"),
    ):
      a, b = w.value()
      if col in df.columns:
        mask &= (df[col] >= a) & (df[col] <= b)

    # Underlying ma dist0
    for col, w in self.ma_filters.items():
      a, b = w.value()
      if col in df.columns:
        mask &= (df[col] >= a) & (df[col] <= b)

    # SPY ma dist0
    for col, w in self.spy_ma_filters.items():
      a, b = w.value()
      if col in df.columns:
        mask &= (df[col] >= a) & (df[col] <= b)

    # Instrument type (ETFs vs Stocks)
    if "is_etf" in df.columns:
      if self.inst_stocks.isChecked():
        mask &= (df["is_etf"] == False)
      elif self.inst_etfs.isChecked():
        mask &= (df["is_etf"] == True)

    # Event types: if none checked => no filtering; else OR together selected event types
    selected_cols: list[str] = []
    if self.evt_earnings.isChecked():
      selected_cols.append("is_earnings")
    if self.evt_atrp.isChecked():
      selected_cols.append("evt_atrp_breakout")
    if self.evt_green.isChecked():
      selected_cols.append("evt_green_line_breakout")

    if selected_cols:
      evt_mask = pd.Series(False, index=df.index)
      for c in selected_cols:
        if c in df.columns:
          evt_mask |= df[c].fillna(False).astype(bool)
      mask &= evt_mask

    # SPY context
    if "spy_class" in df.columns:
      if self.spy_support.isChecked():
        mask &= (df["spy_class"] == "Supporting")
      elif self.spy_neutral.isChecked():
        mask &= (df["spy_class"] == "Neutral")
      elif self.spy_nonsupport.isChecked():
        mask &= (df["spy_class"] == "Non-Supporting")

    # Market cap
    mcap_selected = next((k for k, rb in self.mcap_buttons.items() if rb.isChecked()), "All")
    if mcap_selected != "All" and "market_cap_class" in df.columns:
      mask &= (df["market_cap_class"] == mcap_selected)

    # Conditional survival
    cond_t = int(self.slider_cond_t.value())
    if cond_t > 0:
      col = f"cpct{cond_t}" if self.view_tab == "Daily" else f"w_cpct{cond_t}"
      cmin, cmax = self.cond_range.value()
      if col in df.columns:
        mask &= (df[col] >= cmin) & (df[col] <= cmax)

    sub = df.loc[mask]
    self._last_sub_df = sub

    self._plot(sub)
    self._has_rendered_once = True

    dt = (time.perf_counter() - t0) * 1000.0
    self.lbl_status.setText(f"N={len(sub):,}   render={dt:.0f}ms")

  def _plot(self, sub_df: pd.DataFrame) -> None:
    # ... keep your existing _plot implementation, including the scatter downsampling caps ...
    self.plot_path.clear()
    self.plot_dist.clear()
    self.plot_probs.clear()

    if sub_df.empty:
      for pw in (self.plot_path, self.plot_dist, self.plot_probs):
        pw.addItem(pg.TextItem("No Data", color="#DDDDDD", anchor=(0.5, 0.5)))
      return

    # Ensure legend exists BEFORE plotting (pyqtgraph only auto-registers items created after the legend)
    probs_pi = self.plot_probs.getPlotItem()
    probs_pi.addLegend(
      offset=(10, 10),
      labelTextColor="#DDDDDD",
      brush=pg.mkBrush(0, 0, 0, 120),
      pen=pg.mkPen("#666666"),
    )

    # ---- Selected metric for BOTH top + middle ----
    metric = str(self.cmb_dist.currentData() or "cpct")
    metric_prefix = "" if self.view_tab == "Daily" else "w_"
    max_n = 24 if self.view_tab == "Daily" else 8

    periods: list[int] = []
    metric_cols: list[str] = []
    for i in range(1, max_n + 1):
      c = f"{metric_prefix}{metric}{i}"
      if c in sub_df.columns:
        periods.append(i)
        metric_cols.append(c)

    # ---- Trajectory (top plot) now reflects the selected metric ----
    self.plot_path.setTitle(f"Trajectory ({metric_prefix}{metric} quantiles)", color="#DDDDDD", size="12pt")

    if metric_cols:
      data = sub_df[metric_cols].to_numpy(dtype=float)
      q05 = np.nanquantile(data, 0.05, axis=0)
      q25 = np.nanquantile(data, 0.25, axis=0)
      q50 = np.nanquantile(data, 0.50, axis=0)
      q75 = np.nanquantile(data, 0.75, axis=0)
      q95 = np.nanquantile(data, 0.95, axis=0)

      x = np.array(periods, dtype=float)

      self.plot_path.plot(x, q50, pen=pg.mkPen("c", width=3), name="50th (Median)")

      dash_pen = pg.mkPen((0, 255, 255, 140), width=2, style=QtCore.Qt.PenStyle.DashLine)
      self.plot_path.addItem(pg.PlotDataItem(x, q25, pen=dash_pen))
      self.plot_path.addItem(pg.PlotDataItem(x, q75, pen=dash_pen))

      upper_item = pg.PlotDataItem(x, q95, pen=pg.mkPen((0, 255, 255, 80)))
      lower_item = pg.PlotDataItem(x, q05, pen=pg.mkPen((0, 255, 255, 80)))
      self.plot_path.addItem(upper_item)
      self.plot_path.addItem(lower_item)
      self.plot_path.addItem(pg.FillBetweenItem(upper_item, lower_item, brush=pg.mkBrush(0, 255, 255, 35)))

    self.plot_path.addLine(y=0, pen=pg.mkPen("#888888", style=QtCore.Qt.PenStyle.DashLine))

    # ---- Distribution plot (middle) uses the SAME selected metric ----
    dist_periods = periods
    dist_cols = metric_cols

    def _add_violin_fast(
        x0: float,
        vals: np.ndarray,
        *,
        width: float = 0.38,
        max_points: int = 2500,
        bins: int = 28,
    ) -> None:
      vals = vals[np.isfinite(vals)]
      if vals.size < 10:
        return

      if vals.size > max_points:
        rr = np.random.default_rng(0)
        vals = rr.choice(vals, size=max_points, replace=False)

      y_lo = float(np.nanquantile(vals, 0.01))
      y_hi = float(np.nanquantile(vals, 0.99))
      if not np.isfinite(y_lo) or not np.isfinite(y_hi) or y_hi <= y_lo:
        y_lo = float(np.nanmin(vals))
        y_hi = float(np.nanmax(vals))
      if not np.isfinite(y_lo) or not np.isfinite(y_hi) or y_hi <= y_lo:
        return

      hist, edges = np.histogram(vals, bins=bins, range=(y_lo, y_hi), density=True)
      centers = 0.5 * (edges[:-1] + edges[1:])

      dens = np.asarray(hist, dtype=float)
      dens[~np.isfinite(dens)] = 0.0
      if dens.max() <= 0:
        return

      half_w = (dens / dens.max()) * width
      x_left = x0 - half_w
      x_right = x0 + half_w

      poly_x = np.concatenate([x_left, x_right[::-1]])
      poly_y = np.concatenate([centers, centers[::-1]])

      item = pg.PlotDataItem(
        poly_x,
        poly_y,
        pen=pg.mkPen((72, 219, 251, 90), width=1),
        brush=pg.mkBrush(72, 219, 251, 60),
      )
      self.plot_dist.addItem(item)

      med = float(np.nanmedian(vals))
      q1 = float(np.nanquantile(vals, 0.25))
      q3 = float(np.nanquantile(vals, 0.75))

      self.plot_dist.addItem(
        pg.PlotDataItem([x0 - width * 0.55, x0 + width * 0.55], [med, med], pen=pg.mkPen("w", width=2))
      )
      self.plot_dist.addItem(
        pg.PlotDataItem([x0, x0], [q1, q3], pen=pg.mkPen("w", width=2))
      )

    self.plot_dist.setTitle(f"Distribution ({metric_prefix}{metric}1..N)", color="#DDDDDD", size="12pt")
    self.plot_dist.addLine(y=0, pen=pg.mkPen("#888888", style=QtCore.Qt.PenStyle.DashLine))

    if dist_cols:
      for i, col in zip(dist_periods, dist_cols):
        vals = sub_df[col].dropna().to_numpy(dtype=float)
        _add_violin_fast(float(i), vals)
    else:
      self.plot_dist.addItem(
        pg.TextItem(f"No columns for: {metric_prefix}{metric}1..{max_n}", color="#DDDDDD", anchor=(0.5, 0.5))
      )

    # ---- Probability plot (bottom) ----
    def prob_curve(col_gen: Callable[[int], str], thresh: float = 0.0) -> np.ndarray:
      ys = []
      for i in range(1, max_n + 1):
        col = col_gen(i)
        if col not in sub_df.columns:
          ys.append(np.nan)
          continue
        v = sub_df[col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size:
          ys.append(float(np.mean(v >= thresh) * 100.0))
        else:
          ys.append(np.nan)
      return np.array(ys, dtype=float)

    x = np.arange(1, max_n + 1, dtype=float)

    entry_prefix = "cpct" if self.view_tab == "Daily" else "w_cpct"
    y_entry = prob_curve(lambda i: f"{entry_prefix}{i}", 0.0)
    self.plot_probs.plot(
      x, y_entry,
      pen=pg.mkPen("#ff6b6b", width=2),
      symbol="o", symbolSize=5,
      name=f"> Entry",
    )

    if self.view_tab == "Daily":
      ma5_gen = lambda i: f"ma5_dist{i}"
      ma10_gen = lambda i: f"ma10_dist{i}"
      ma20_gen = lambda i: f"ma20_dist{i}"
      ma50_gen = lambda i: f"ma50_dist{i}"
    else:
      ma5_gen = lambda i: f"w_ma5_dist{i}"
      ma10_gen = lambda i: f"w_ma10_dist{i}"
      ma20_gen = lambda i: f"w_ma20_dist{i}"
      ma50_gen = lambda i: f"w_ma50_dist{i}"

    y_ma5 = prob_curve(ma5_gen, 0.0)
    y_ma10 = prob_curve(ma10_gen, 0.0)
    y_ma20 = prob_curve(ma20_gen, 0.0)
    y_ma50 = prob_curve(ma50_gen, 0.0)

    self.plot_probs.plot(x, y_ma5, pen=pg.mkPen("#c8d6e5", width=2), symbol="o", symbolSize=5, name=f"> ma5")
    self.plot_probs.plot(x, y_ma10, pen=pg.mkPen("#feca57", width=2), symbol="o", symbolSize=5, name=f"> ma10")
    self.plot_probs.plot(x, y_ma20, pen=pg.mkPen("#48dbfb", width=2), symbol="o", symbolSize=5, name=f"> ma20")
    self.plot_probs.plot(x, y_ma50, pen=pg.mkPen("#1dd1a1", width=2), symbol="o", symbolSize=5, name=f"> ma50")

    self.plot_probs.setYRange(0, 105)

def _apply_dark_palette(app: QtWidgets.QApplication) -> None:
  """Safe to call repeatedly; keeps styling consistent."""
  app.setStyle("Fusion")
  palette = QtGui.QPalette()
  palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("#111111"))
  palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("#000000"))
  palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor("#DDDDDD"))
  palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor("#222222"))
  palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor("#DDDDDD"))
  app.setPalette(palette)


def main(start_year, *, exec_: Optional[bool] = None) -> DashboardQt:
  global _GLOBAL_QT_APP, _GLOBAL_DASHBOARD_WIN, _GLOBAL_LOADED_DF

  _ensure_ipython_qt_event_loop()

  if _GLOBAL_QT_APP is None:
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    _GLOBAL_QT_APP = pg.mkQApp()

  _apply_dark_palette(_GLOBAL_QT_APP)

  if _GLOBAL_DASHBOARD_WIN is not None:
    try:
      if not _GLOBAL_DASHBOARD_WIN.isVisible():
        _GLOBAL_DASHBOARD_WIN = None
    except Exception:
      _GLOBAL_DASHBOARD_WIN = None

  # If caller passed no usable df, reuse global loaded data; if none, load defaults (2012..this year)
  if _GLOBAL_LOADED_DF is not None and (not _GLOBAL_LOADED_DF.empty):
    df = _GLOBAL_LOADED_DF
  else:
    this_year = int(pd.Timestamp.now().year)
    df = load_and_prep_data(range(start_year, this_year + 1))
    _GLOBAL_LOADED_DF = df

  if _GLOBAL_DASHBOARD_WIN is None:
    _GLOBAL_DASHBOARD_WIN = DashboardQt(df)
  else:
    _GLOBAL_DASHBOARD_WIN.df = df
    _GLOBAL_LOADED_DF = df
    # IMPORTANT: do NOT auto-render on reopen; keep it empty until user clicks Update.
    _GLOBAL_DASHBOARD_WIN._update_cond_label_and_bounds()
    _GLOBAL_DASHBOARD_WIN._refresh_filter_bounds_from_df()
    _GLOBAL_DASHBOARD_WIN._show_empty_state()
    _GLOBAL_DASHBOARD_WIN._update_data_status()

  _GLOBAL_DASHBOARD_WIN.show()
  _GLOBAL_DASHBOARD_WIN.showMaximized()
  _GLOBAL_DASHBOARD_WIN.raise_()
  _GLOBAL_DASHBOARD_WIN.activateWindow()

  try:
    _GLOBAL_QT_APP.processEvents()
    _GLOBAL_QT_APP.processEvents()
  except Exception:
    pass

  if exec_ is None:
    exec_ = not _in_ipython()

  if exec_:
    _GLOBAL_QT_APP.exec()

  return _GLOBAL_DASHBOARD_WIN


# %%
# Only run automatically when executed as a script, not when imported.
if __name__ == "__main__":
  #%%
  main(start_year=2022, exec_=True)
