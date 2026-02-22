from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable, Optional
from scipy.stats import gaussian_kde  # type: ignore

import numpy as np
import pandas as pd

# Qt binding: prefer PySide6, fallback to PyQt5
from PySide6 import QtCore, QtGui, QtWidgets

import pyqtgraph as pg
use_violin = True

#%%
# Global cache for reusing the window and app across multiple plot calls
_GLOBAL_QT_APP: Optional[QtWidgets.QApplication] = None
_GLOBAL_DASHBOARD_WIN: Optional["DashboardQt"] = None

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
  dfs: list[pd.DataFrame] = []
  for year in years:
    filename = f"finance/_data/momentum_earnings/all_{year}.pkl"
    try:
      dfs.append(pd.read_pickle(filename))
    except FileNotFoundError:
      continue

  if not dfs:
    return pd.DataFrame()

  df = pd.concat(dfs, ignore_index=True)

  # Match matplotlib dashboard: basic cleanup + safety caps
  if "original_price" in df.columns:
    df = df[df["original_price"] < 10e5]

  df = df.replace([np.inf, -np.inf], np.nan).infer_objects(copy=False)

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

    self.setWindowTitle("Momentum & Earnings Dashboard (Qt)")
    self.resize(3400, 1900)

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

    title = QtWidgets.QLabel("Filters")
    title.setStyleSheet("font-size: 16px; font-weight: 600;")
    c_layout.addWidget(title)

    # --- View Tab (Daily / Weekly) ---
    tab_row = QtWidgets.QHBoxLayout()
    c_layout.addLayout(tab_row)
    self.btn_tab_d = QtWidgets.QPushButton("Daily")
    self.btn_tab_w = QtWidgets.QPushButton("Weekly")
    self.btn_tab_d.setCheckable(True)
    self.btn_tab_w.setCheckable(True)
    self.btn_tab_d.setChecked(True)

    def _set_tab(name: str) -> None:
      self.view_tab = name
      self.btn_tab_d.setChecked(name == "Daily")
      self.btn_tab_w.setChecked(name == "Weekly")
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

    # 2. Year Range
    self.year = self._add_year_range(grid, "Year Range", df)

    # 3. Event Move (signed)
    self.event_move = self._add_range(grid, "Event Move (Signed)", df, col="event_move", decimals=3, step=0.25)

    # 4. Breakout Price
    self.event_price = self._add_range(grid, "Breakout Price", df, col="event_price", decimals=2, step=0.5)

    # 6-8. Momentum Filters
    self.mom_1m = self._add_range(grid, "1M_chg", df, col="1M_chg", decimals=1, step=1.0)
    self.mom_3m = self._add_range(grid, "3M_chg", df, col="3M_chg", decimals=1, step=1.0)
    self.mom_6m = self._add_range(grid, "6M_chg", df, col="6M_chg", decimals=1, step=1.0)

    # 9-13. Underlying EMA dist0 filters
    self.ema_filters: dict[str, MinMaxSpin] = {}
    for ema_name in ["ema10", "ema20", "ema50", "ema100", "ema200"]:
      col = f"{ema_name}_dist0"
      self.ema_filters[col] = self._add_range(grid, f"{ema_name} Dist", df, col=col, decimals=2, step=0.25)

    # 14-18. SPY EMA dist0 filters
    self.spy_ema_filters: dict[str, MinMaxSpin] = {}
    for ema_name in ["ema10", "ema20", "ema50", "ema100", "ema200"]:
      col = f"spy_{ema_name}_dist0"
      self.spy_ema_filters[col] = self._add_range(grid, f"SPY {ema_name} Dist", df, col=col, decimals=2, step=0.25)

    # Direction
    dir_box = QtWidgets.QGroupBox("Direction")
    dir_layout = QtWidgets.QHBoxLayout(dir_box)
    self.dir_pos = QtWidgets.QRadioButton("Positive")
    self.dir_neg = QtWidgets.QRadioButton("Negative")
    self.dir_pos.setChecked(True)
    for w in (self.dir_pos, self.dir_neg):
      dir_layout.addWidget(w)
    c_layout.addWidget(dir_box)

    # Earnings
    earn_box = QtWidgets.QGroupBox("Event Type")
    earn_layout = QtWidgets.QHBoxLayout(earn_box)
    self.earn_all = QtWidgets.QRadioButton("All")
    self.earn_e = QtWidgets.QRadioButton("Earnings Only")
    self.earn_ne = QtWidgets.QRadioButton("Non-Earnings")
    self.earn_all.setChecked(True)
    for w in (self.earn_all, self.earn_e, self.earn_ne):
      earn_layout.addWidget(w)
    c_layout.addWidget(earn_box)

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

    self.cond_range = self._add_range(
      grid=None, title="Cond. Range", df=df, col="__cond_placeholder__", decimals=1, step=0.5,
      add_to_grid=False
    )

    cond_layout.addWidget(self.lbl_cond_t, 0, 0, 1, 1)
    cond_layout.addWidget(self.slider_cond_t, 0, 1, 1, 2)
    cond_layout.addWidget(self.cond_range.label, 1, 0, 1, 1)
    cond_layout.addWidget(self.cond_range.min_box, 1, 1, 1, 1)
    cond_layout.addWidget(self.cond_range.max_box, 1, 2, 1, 1)

    self.slider_cond_t.valueChanged.connect(lambda _v: self._update_cond_label_and_bounds())
    c_layout.addWidget(cond_box)

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

    p_layout.addWidget(self.plot_path, 2)
    p_layout.addWidget(self.plot_dist, 2)
    p_layout.addWidget(self.plot_probs, 1)

    # Initialize UI bounds, but DO NOT render any data yet.
    self._update_cond_label_and_bounds()
    self._show_empty_state()

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

  def _update_cond_label_and_bounds(self) -> None:
    # Label and maximum depend on tab (Daily: 24, Weekly: 8)
    max_t = 24 if self.view_tab == "Daily" else 8
    self.slider_cond_t.setMaximum(max_t)

    t = int(self.slider_cond_t.value())
    self.lbl_cond_t.setText("Cond. Day:" if self.view_tab == "Daily" else "Cond. Wk:")

    # Update Cond. Range allowed bounds based on data distribution (like matplotlib)
    if t <= 0:
      vmin, vmax = -20.0, 50.0
    else:
      col = f"cpct{t}" if self.view_tab == "Daily" else f"w_cpct{t}"
      if col in self.df.columns:
        vals = self.df[col].dropna()
        if not vals.empty:
          vmin = float(vals.quantile(0.01))
          vmax = float(vals.quantile(0.99))
        else:
          vmin, vmax = -20.0, 50.0
      else:
        vmin, vmax = -20.0, 50.0

    if vmin >= vmax:
      vmax = vmin + 1.0

    # Update spinbox ranges and values
    for b in (self.cond_range.min_box, self.cond_range.max_box):
      b.blockSignals(True)
      b.setRange(vmin, vmax)
      b.setSingleStep(0.5)
    self.cond_range.min_box.setValue(vmin)
    self.cond_range.max_box.setValue(vmax)
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
    label = QtWidgets.QLabel(f"{title}:")
    min_box = QtWidgets.QDoubleSpinBox()
    max_box = QtWidgets.QDoubleSpinBox()

    # Data-derived bounds (available range)
    series = df[col].dropna() if (col in df.columns) else pd.Series([], dtype=float)
    if not series.empty:
      hard_min = float(series.min())
      hard_max = float(series.max())
    else:
      hard_min, hard_max = -1e9, 1e9

    for b in (min_box, max_box):
      b.setDecimals(decimals)
      # Set the range of the spinbox to the actual data boundaries
      b.setRange(hard_min, hard_max)
      b.setSingleStep(step)
      b.setKeyboardTracking(False)
      b.setAccelerated(True)

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
    import time

    t0 = time.perf_counter()

    df = self.df
    if df.empty:
      self.lbl_status.setText("No data loaded.")
      self._show_empty_state()
      return

    mask = pd.Series(True, index=df.index)

    # Year
    y_min, y_max = self.year.value()
    if "date" in df.columns:
      yy = df["date"].dt.year
      mask &= (yy >= int(y_min)) & (yy <= int(y_max))

    # Event move + price
    ev_min, ev_max = self.event_move.value()
    if "event_move" in df.columns:
      mask &= (df["event_move"] >= ev_min) & (df["event_move"] <= ev_max)

    p_min, p_max = self.event_price.value()
    if "event_price" in df.columns:
      mask &= (df["event_price"] >= p_min) & (df["event_price"] <= p_max)

    # Momentum
    for w, col in [(self.mom_1m, "1M_chg"), (self.mom_3m, "3M_chg"), (self.mom_6m, "6M_chg")]:
      a, b = w.value()
      if col in df.columns:
        mask &= (df[col] >= a) & (df[col] <= b)

    # Underlying EMA dist0
    for col, w in self.ema_filters.items():
      a, b = w.value()
      if col in df.columns:
        mask &= (df[col] >= a) & (df[col] <= b)

    # SPY EMA dist0
    for col, w in self.spy_ema_filters.items():
      a, b = w.value()
      if col in df.columns:
        mask &= (df[col] >= a) & (df[col] <= b)

    # Direction
    if self.dir_pos.isChecked():
      mask &= (df["event_move"] > 0)
    elif self.dir_neg.isChecked():
      mask &= (df["event_move"] < 0)

    # Earnings
    if self.earn_e.isChecked() and "is_earnings" in df.columns:
      mask &= (df["is_earnings"] == True)
    elif self.earn_ne.isChecked() and "is_earnings" in df.columns:
      mask &= (df["is_earnings"] == False)

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

    self._plot(sub)
    self._has_rendered_once = True

    dt = (time.perf_counter() - t0) * 1000.0
    self.lbl_status.setText(f"N={len(sub):,}   render={dt:.0f}ms")

  def _plot(self, sub_df: pd.DataFrame) -> None:
    # ... keep your existing _plot implementation, including the scatter downsampling caps ...
    # (no change required here beyond what you already have)
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

    prefix = "cpct" if self.view_tab == "Daily" else "w_cpct"
    max_n = 24 if self.view_tab == "Daily" else 8

    periods: list[int] = []
    cols: list[str] = []
    for i in range(1, max_n + 1):
      c = f"{prefix}{i}"
      if c in sub_df.columns:
        periods.append(i)
        cols.append(c)

    if cols:
      data = sub_df[cols].to_numpy(dtype=float)
      q05 = np.nanquantile(data, 0.05, axis=0)
      q50 = np.nanquantile(data, 0.50, axis=0)
      q95 = np.nanquantile(data, 0.95, axis=0)

      x = np.array(periods, dtype=float)
      self.plot_path.plot(x, q50, pen=pg.mkPen("c", width=2), name="50th (Median)")

      # 5th / 95th quantiles + filled band
      upper_item = pg.PlotDataItem(x, q95, pen=pg.mkPen((0, 255, 255, 80)))
      lower_item = pg.PlotDataItem(x, q05, pen=pg.mkPen((0, 255, 255, 80)))
      self.plot_path.addItem(upper_item)
      self.plot_path.addItem(lower_item)
      self.plot_path.addItem(pg.FillBetweenItem(upper_item, lower_item, brush=pg.mkBrush(0, 255, 255, 35)))

      self.plot_path.addLine(y=0, pen=pg.mkPen("#888888", style=QtCore.Qt.PenStyle.DashLine))
      self.plot_dist.addLine(y=0, pen=pg.mkPen("#888888", style=QtCore.Qt.PenStyle.DashLine))

      # --- Violin-style distribution in the middle panel (replaces scatter cloud) ---
      self.plot_dist.setTitle("Distribution (Violin)", color="#DDDDDD", size="12pt")

      def _add_violin_fast(
          x0: float,
          vals: np.ndarray,
          *,
          width: float = 0.38,
          max_points: int = 2500,   # cap per period to keep it snappy
          bins: int = 28,           # fewer bins = faster
      ) -> None:
        vals = vals[np.isfinite(vals)]
        if vals.size < 10:
          return

        # Downsample aggressively (histogram shape is stable with a couple thousand points)
        if vals.size > max_points:
          rr = np.random.default_rng(0)
          vals = rr.choice(vals, size=max_points, replace=False)

        # Robust bounds so outliers don't crush the violin
        y_lo = float(np.nanquantile(vals, 0.01))
        y_hi = float(np.nanquantile(vals, 0.99))
        if not np.isfinite(y_lo) or not np.isfinite(y_hi) or y_hi <= y_lo:
          y_lo = float(np.nanmin(vals))
          y_hi = float(np.nanmax(vals))
        if not np.isfinite(y_lo) or not np.isfinite(y_hi) or y_hi <= y_lo:
          return

        # Histogram-density "violin" (fast approximation)
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

        # Median + IQR markers
        med = float(np.nanmedian(vals))
        q1 = float(np.nanquantile(vals, 0.25))
        q3 = float(np.nanquantile(vals, 0.75))

        self.plot_dist.addItem(
          pg.PlotDataItem([x0 - width * 0.55, x0 + width * 0.55], [med, med], pen=pg.mkPen("w", width=2))
        )
        self.plot_dist.addItem(
          pg.PlotDataItem([x0, x0], [q1, q3], pen=pg.mkPen("w", width=2))
        )

      for i, col in zip(periods, cols):
        vals = sub_df[col].dropna().to_numpy(dtype=float)
        _add_violin_fast(float(i), vals)

        # Direction-aware survival:
        # - Positive breakouts: want values > 0  ("> Entry", "> EMAx")
        # - Negative breakouts: want values < 0  ("< Entry", "< EMAx")
        dir_sign = 1.0 if self.dir_pos.isChecked() else -1.0
        dir_prefix = ">" if dir_sign > 0 else "<"

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
            # sign-flip makes the condition always "continuation is positive"
            ys.append(float(np.mean((dir_sign * v) > thresh) * 100.0))
          else:
            ys.append(np.nan)
        return np.array(ys, dtype=float)

      x = np.arange(1, max_n + 1, dtype=float)

      # Entry (breakout) survival
      y_entry = prob_curve(lambda i: f"{prefix}{i}", 0.0)
      self.plot_probs.plot(
        x, y_entry,
        pen=pg.mkPen("#ff6b6b", width=2),
        symbol="o", symbolSize=5,
        name=f"{dir_prefix} Entry",
      )

      # EMA survival lines (direction-aware labels + sign)
      if self.view_tab == "Daily":
        ema5_gen = lambda i: f"ema5_dist{i}"
        ema10_gen = lambda i: f"ema10_dist{i}"
        ema20_gen = lambda i: f"ema20_dist{i}"
        ema50_gen = lambda i: f"ema50_dist{i}"
      else:
        ema5_gen = lambda i: f"w_ema5_dist{i}"
        ema10_gen = lambda i: f"w_ema10_dist{i}"
        ema20_gen = lambda i: f"w_ema20_dist{i}"
        ema50_gen = lambda i: f"w_ema50_dist{i}"

      y_ema5 = prob_curve(ema5_gen, 0.0)
      y_ema10 = prob_curve(ema10_gen, 0.0)
      y_ema20 = prob_curve(ema20_gen, 0.0)
      y_ema50 = prob_curve(ema50_gen, 0.0)

      self.plot_probs.plot(x, y_ema5, pen=pg.mkPen("#c8d6e5", width=2), symbol="o", symbolSize=5, name=f"{dir_prefix} EMA5")
      self.plot_probs.plot(x, y_ema10, pen=pg.mkPen("#feca57", width=2), symbol="o", symbolSize=5, name=f"{dir_prefix} EMA10")
      self.plot_probs.plot(x, y_ema20, pen=pg.mkPen("#48dbfb", width=2), symbol="o", symbolSize=5, name=f"{dir_prefix} EMA20")
      self.plot_probs.plot(x, y_ema50, pen=pg.mkPen("#1dd1a1", width=2), symbol="o", symbolSize=5, name=f"{dir_prefix} EMA50")

    self.plot_probs.setYRange(0, 105)
    self.plot_probs.getPlotItem().addLegend(
      offset=(10, 10),
      labelTextColor="#DDDDDD",
      brush=pg.mkBrush(0, 0, 0, 120),
      pen=pg.mkPen("#666666"),
    )


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


def main(df: pd.DataFrame, *, exec_: Optional[bool] = None) -> DashboardQt:
  global _GLOBAL_QT_APP, _GLOBAL_DASHBOARD_WIN

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

  if _GLOBAL_DASHBOARD_WIN is None:
    _GLOBAL_DASHBOARD_WIN = DashboardQt(df)
  else:
    _GLOBAL_DASHBOARD_WIN.df = df
    # IMPORTANT: do NOT auto-render on reopen; keep it empty until user clicks Update.
    _GLOBAL_DASHBOARD_WIN._update_cond_label_and_bounds()
    _GLOBAL_DASHBOARD_WIN._show_empty_state()

  _GLOBAL_DASHBOARD_WIN.show()
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
# Only run automatically when executed as a script, not when imported into IPython.
if __name__ == "__main__":
  # df = load_and_prep_data(range(2012, 2026))
  df = load_and_prep_data(range(2022, 2026))
  main(df, exec_=True)

# %%
df = load_and_prep_data(range(2022, 2026))
# df = load_and_prep_data(range(2012, 2026))
# %%
main(df)   # returns immediately in IPython; window stays open
