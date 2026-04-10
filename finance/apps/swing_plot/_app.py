"""
finance.apps.swing_plot._app
===============================
SwingPlotWindow — the main QMainWindow subclass.
All UI state is stored as instance variables; no module-level globals.
Signal wiring happens once in __init__; load_symbol() updates data and
calls update_all() without rebuilding the window.
"""
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph import QtCore
from pyqtgraph.Qt import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ._chart import (
    _setup_plot_panes, _add_plot_content, _auto_scale_panes,
    _force_layout_and_scene_sync, _sync_twin_viewboxes,
)
from ._tabs import (
    render_stats_violins,
    render_volatility_analysis,
    render_drawdown_analysis,
    render_trend_regime,
    render_pullback_vcp,
    render_move_otm,
    render_pead,
)


class SwingPlotWindow(QtWidgets.QMainWindow):
    """
    Persistent main window for the swing plot dashboard.
    Instantiated once; call load_symbol() to switch instruments.
    """

    DISPLAY_RANGE = 250

    def __init__(self):
        super().__init__()

        # ---- data state ------------------------------------------------
        self._full_df:  pd.DataFrame = pd.DataFrame()
        self._spy_df:   pd.DataFrame = pd.DataFrame()
        self._title:    str = ''
        self._symbol:   str = ''
        self._active_plots: list = []

        # ---- IBKR connection state ------------------------------------
        self._ib_con = None

        # ---- window skeleton -------------------------------------------
        self.setMinimumSize(1200, 700)
        self.resize(1600, 900)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root_layout = QtWidgets.QVBoxLayout(central)

        # Toolbar
        toolbar = self._build_toolbar()
        root_layout.addLayout(toolbar)

        # Tabs
        self._tabs = QtWidgets.QTabWidget()
        root_layout.addWidget(self._tabs)

        # Tab 1: Price Chart (PyQtGraph)
        chart_tab = QtWidgets.QWidget()
        chart_layout = QtWidgets.QVBoxLayout(chart_tab)
        self._glw = pg.GraphicsLayoutWidget()
        chart_layout.addWidget(self._glw)
        self._tabs.addTab(chart_tab, 'Price Chart')

        # Tab 2: Probability Trees
        self._canvas_trees, trees_tab = self._make_scroll_tab(
            fig_size=(10, 8), tab_label='Probability Trees')

        # Tab 3: Seasonality
        self._canvas_stats, stats_tab = self._make_scroll_tab(
            fig_size=(12, 18), tab_label='Seasonality')

        # Tab 4: Volatility
        self._canvas_vol, vol_tab = self._make_scroll_tab(
            fig_size=(12, 22), tab_label='Volatility')

        # Tab 5: Drawdown
        self._canvas_dd, dd_tab = self._make_scroll_tab(
            fig_size=(12, 18), tab_label='Drawdown')

        # Tab 6: Trend Regime (trader)
        self._canvas_tr, tr_tab = self._make_scroll_tab(
            fig_size=(24, 35), tab_label='Trend Regime')

        # Tab 7: Pullback and VCP (trader)
        self._canvas_pv, pv_tab = self._make_scroll_tab(
            fig_size=(24, 35), tab_label='Pullback and VCP')

        # Tab 8: Move and Options (trader)
        self._canvas_mo, mo_tab = self._make_scroll_tab(
            fig_size=(24, 60), tab_label='Move and Options')

        # Tab 9: Earnings Drift (trader — PM-02 / PM-03)
        self._canvas_pead, pead_tab = self._make_scroll_tab(
            fig_size=(24, 40), tab_label='Earnings Drift')

        # ---- per-chart state (crosshair, hover) -------------------------
        self._chart_state = {
            'proxy':        None,
            'df':           None,
            'x_dates':      None,
            'hover_labels': None,
            'v_lines':      None,
            'h_lines':      None,
        }

        # ---- wire signals -----------------------------------------------
        self._load_btn.clicked.connect(self._on_load_clicked)
        self._ticker_input.returnPressed.connect(self._on_load_clicked)
        self._tree_tf_combo.currentIndexChanged.connect(
            lambda: self._update_trees())
        self._start_year_cb.currentIndexChanged.connect(
            lambda: self.update_all())
        self._end_year_cb.currentIndexChanged.connect(
            lambda: self.update_all())

        # ---- IBKR status polling ---------------------------------------
        self._status_timer = QtCore.QTimer(self)
        self._status_timer.setInterval(5000)
        self._status_timer.timeout.connect(self._update_ib_status)
        self._status_timer.start()
        self._update_ib_status()

    # ------------------------------------------------------------------
    # IBKR helpers
    # ------------------------------------------------------------------

    def _is_ib_connected(self) -> bool:
        con = self._ib_con
        try:
            return bool(con is not None and con.isConnected())
        except Exception:
            return False

    def _ensure_ib_connection(self, instance: str):
        """Connect to IBKR using the given instance name; returns the connection or None."""
        if instance == 'offline':
            return None
        if self._is_ib_connected():
            return self._ib_con
        from finance.utils import ibkr as ib_utils
        try:
            self._ib_con = ib_utils.connect(instance, 17, 1)
            self._update_ib_status()
            return self._ib_con
        except Exception as e:
            print(f'IBKR connect failed ({instance}): {e}')
            self._ib_con = None
            self._update_ib_status()
            return None

    def _on_ibkr_instance_changed(self, instance: str):
        """Disconnect the old connection and attempt connection to the new instance."""
        # Disconnect existing
        if self._ib_con is not None:
            try:
                self._ib_con.disconnect()
            except Exception:
                pass
            self._ib_con = None
        # Attempt new connection immediately (unless offline)
        if instance != 'offline':
            self._ensure_ib_connection(instance)
        self._update_ib_status()

    def _update_ib_status(self):
        label = getattr(self, '_status_label', None)
        if label is None:
            return
        instance = self._ibkr_combo.currentText() if hasattr(self, '_ibkr_combo') else 'offline'
        if instance == 'offline':
            label.setText('● IBKR (offline)')
            label.setStyleSheet('color: #888888; font-weight: 600;')
            return
        if self._is_ib_connected():
            label.setText(f'● IBKR ({instance})')
            label.setStyleSheet('color: #24ad54; font-weight: 600;')
        else:
            label.setText(f'● IBKR ({instance})')
            label.setStyleSheet('color: #ec4533; font-weight: 600;')

    def _refresh_from_ibkr(self, symbol: str, instance: str, show_errors: bool = True) -> bool:
        """Fetch fresh bars for `symbol` from IBKR. Returns True on success."""
        from finance.utils import ibkr as ib_utils
        con = self._ensure_ib_connection(instance)
        if con is None:
            if show_errors:
                QtWidgets.QMessageBox.warning(
                    self, 'IBKR',
                    f'Could not connect to IBKR ({instance}). Using cached data if available.')
            return False
        try:
            df = ib_utils.daily_w_volatility(symbol, api=instance, offline=False,
                                              ib_con=con, refresh_offset_days=0)
            self._update_ib_status()
            return df is not None and not df.empty
        except Exception as e:
            self._update_ib_status()
            if show_errors:
                QtWidgets.QMessageBox.warning(
                    self, 'IBKR',
                    f'Failed to refresh {symbol} from IBKR: {e}')
            return False

    def _prepare_symbol_cache(self, symbol: str, instance: str) -> bool:
        """Ensure a cache exists for `symbol` before loading. Refresh is manual
        via the Refresh button — we only prompt when no cache exists at all.
        """
        from finance.utils import ibkr as ib_utils

        if ib_utils.has_cache(symbol):
            return True

        if instance == 'offline':
            QtWidgets.QMessageBox.warning(
                self, 'No Data',
                f'No cached data for {symbol}. Select an IBKR instance to acquire it.')
            return False

        reply = QtWidgets.QMessageBox.question(
            self, 'Acquire Data',
            f'No cached data for {symbol}.\n\nAcquire from IBKR ({instance}) now?',
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.Yes,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return False
        ok = self._refresh_from_ibkr(symbol, instance, show_errors=True)
        return ok and ib_utils.has_cache(symbol)

    def _update_freshness_label(self):
        """Update the right-side freshness badge for the currently loaded symbol."""
        label = getattr(self, '_freshness_label', None)
        if label is None:
            return
        symbol = self._symbol
        if not symbol:
            label.setText('')
            return
        from finance.utils import ibkr as ib_utils
        last = ib_utils.get_cached_last_bar_date(symbol)
        if last is None:
            label.setText(f'{symbol}: no data')
            label.setStyleSheet('color: #ec4533; font-weight: 600;')
            return
        today = datetime.now().date()
        days = (today - last).days
        if days <= 1:
            label.setText(f'{symbol}: fresh ({last.isoformat()})')
            label.setStyleSheet('color: #24ad54; font-weight: 600;')
        else:
            label.setText(f'{symbol}: STALE ({days}d, {last.isoformat()})')
            label.setStyleSheet('color: #f5a623; font-weight: 600;')

    def _on_refresh_clicked(self):
        """Manual refresh of the currently loaded symbol (and SPY overlay)."""
        symbol = self._symbol
        if not symbol:
            return
        instance = self._ibkr_combo.currentText()
        if instance == 'offline':
            QtWidgets.QMessageBox.information(
                self, 'IBKR',
                'Cannot refresh while IBKR mode is set to offline.')
            return
        self._refresh_btn.setEnabled(False)
        try:
            ok = self._refresh_from_ibkr(symbol, instance, show_errors=True)
            # Also refresh SPY overlay on demand
            if symbol != 'SPY':
                self._refresh_from_ibkr('SPY', instance, show_errors=False)
            if ok:
                # Reload the symbol so all tabs pick up the new data
                self.load_symbol(symbol)
            self._update_freshness_label()
        finally:
            self._refresh_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_symbol(self, symbol: str, datasource: str = 'offline'):
        """Load a new symbol (and SPY overlay) then refresh all tabs.

        Flow:
          1. If IBKR combo is 'offline' or cache is fresh (<= 1 day): load offline.
          2. If no cache at all: prompt before acquiring.
          3. Otherwise: refresh from IBKR, then load offline.
        """
        from finance.utils.swing_trading_data import SwingTradingData
        from finance.utils import ibkr as ib_utils

        symbol = symbol.upper()
        instance = self._ibkr_combo.currentText()

        if not self._prepare_symbol_cache(symbol, instance):
            return  # user declined or fetch failed without any cache

        try:
            data = SwingTradingData(symbol, datasource='offline')
            if data.df_day is None or data.df_day.empty:
                QtWidgets.QMessageBox.warning(
                    self, 'Load Error', f'No data found for {symbol}')
                return
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to load {symbol}: {e}')
            return

        self._full_df = data.df_day
        self._symbol  = symbol
        self._title   = symbol
        self.setWindowTitle(f'Swing Trading Analysis — {symbol}')

        # Load SPY for overlay (skip if symbol IS SPY). No auto-refresh — user
        # can click Refresh on the toolbar to sync everything manually.
        if symbol != 'SPY' and ib_utils.has_cache('SPY'):
            try:
                spy_data     = SwingTradingData('SPY', datasource='offline')
                self._spy_df = spy_data.df_day if spy_data.df_day is not None else pd.DataFrame()
            except Exception:
                self._spy_df = pd.DataFrame()
        else:
            self._spy_df = pd.DataFrame()

        self._update_freshness_label()

        # Reset year filters
        all_years = sorted(self._full_df.index.year.unique())
        self._start_year_cb.blockSignals(True)
        self._end_year_cb.blockSignals(True)
        self._start_year_cb.clear()
        self._end_year_cb.clear()
        self._start_year_cb.addItems([str(y) for y in all_years])
        self._end_year_cb.addItems([str(y) for y in all_years])
        default_start = max(2006, all_years[0])
        if default_start > all_years[-1]:
            default_start = all_years[0]
        self._start_year_cb.setCurrentText(str(default_start))
        self._end_year_cb.setCurrentText(str(all_years[-1]))
        self._start_year_cb.blockSignals(False)
        self._end_year_cb.blockSignals(False)

        self.update_all()

    def update_all(self):
        """Refresh all tabs with the current data and year filters."""
        self._update_chart()
        self._update_trees()
        self._update_stats()
        self._update_vol()
        self._update_drawdown()
        self._update_trend_regime()
        self._update_pullback_vcp()
        self._update_move_otm()
        self._update_pead()

    # ------------------------------------------------------------------
    # Tab updaters
    # ------------------------------------------------------------------

    def _filtered_df(self) -> pd.DataFrame:
        if self._full_df.empty:
            return pd.DataFrame()
        try:
            sy = int(self._start_year_cb.currentText())
            ey = int(self._end_year_cb.currentText())
            return self._full_df[
                (self._full_df.index.year >= sy) & (self._full_df.index.year <= ey)
            ]
        except Exception:
            return pd.DataFrame()

    def _update_chart(self):
        # Disconnect old signals
        if self._active_plots:
            try:
                self._active_plots[0].sigXRangeChanged.disconnect()
            except Exception:
                pass
        if self._chart_state['proxy']:
            try:
                self._chart_state['proxy'].disconnect()
            except Exception:
                pass
        # Remove twin viewboxes from the scene before clearing the layout —
        # otherwise the orphaned right-axis curves stay drawn over the new chart.
        for p in self._active_plots:
            vb = getattr(p, '_right_vb', None)
            if vb is not None:
                try:
                    self._glw.scene().removeItem(vb)
                except Exception:
                    pass
                p._right_vb = None
        for p in self._active_plots:
            p.deleteLater()
        self._glw.clear()
        self._active_plots = []

        df = self._filtered_df()
        if df.empty or not isinstance(df.index, pd.DatetimeIndex):
            self._glw.addItem(
                pg.LabelItem('No Data Available', justify='center', size='12pt'), row=0, col=0)
            return

        self._chart_state['df']      = df
        self._chart_state['x_dates'] = df.index

        plots = _setup_plot_panes(self._glw, df.index, row_offset=0)
        self._active_plots = plots

        spy_df = self._spy_df if not self._spy_df.empty else None
        _add_plot_content(plots, df, vlines=None, spy_df=spy_df)

        self._wire_crosshair(plots, df)

        plots[0].sigXRangeChanged.connect(self._update_y_views)
        plots[0].setXRange(max(0, len(df) - self.DISPLAY_RANGE), len(df))
        self._update_y_views()
        # Force twin viewboxes (COTR / IV-HV) to match their parent geometry now
        # AND once more after the event loop settles, to prevent the right-axis
        # series from briefly drawing inside the wrong pane.
        _sync_twin_viewboxes(plots)
        QtCore.QTimer.singleShot(0, lambda: _sync_twin_viewboxes(plots))

    def _update_trees(self):
        from finance.utils.plots import plot_probability_tree
        df = self._filtered_df()
        if df.empty:
            return

        canv = self._canvas_trees
        canv.figure.clear()
        canv.figure.patch.set_facecolor('#111111')
        canv.figure.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.02)
        ax = canv.figure.add_subplot(111)
        ax.set_facecolor('#111111')

        tf = self._tree_tf_combo.currentText()
        if tf == 'Daily':
            plot_probability_tree(df['pct'], depth=6, title='Daily Moves', ax=ax)
        elif tf == 'Weekly':
            df_w = df['c'].resample('1W').last().pct_change() * 100
            plot_probability_tree(df_w.dropna(), depth=6, title='Weekly Moves', ax=ax)
        elif tf == 'Monthly':
            df_m = df['c'].resample('1ME').last().pct_change() * 100
            plot_probability_tree(df_m.dropna(), depth=6, title='Monthly Moves', ax=ax)
        canv.draw()

    def _update_stats(self):
        df = self._filtered_df()
        sy, ey = self._year_range()
        if df.empty or sy is None:
            return
        render_stats_violins(self._canvas_stats.figure, df, sy, ey)
        self._canvas_stats.draw()

    def _update_vol(self):
        df = self._filtered_df()
        sy, ey = self._year_range()
        if df.empty or sy is None:
            return
        spy_df = self._spy_df if not self._spy_df.empty else None
        render_volatility_analysis(self._canvas_vol.figure, df, sy, ey, spy_df=spy_df)
        self._canvas_vol.draw()

    def _update_drawdown(self):
        df = self._filtered_df()
        sy, ey = self._year_range()
        if df.empty or sy is None:
            return
        render_drawdown_analysis(self._canvas_dd.figure, df, sy, ey)
        self._canvas_dd.draw()

    def _update_trend_regime(self):
        df = self._full_df
        if df.empty:
            return
        render_trend_regime(self._canvas_tr.figure, df, symbol=self._symbol)
        self._canvas_tr.draw()

    def _update_pullback_vcp(self):
        df = self._full_df
        if df.empty:
            return
        spy_df = self._spy_df if not self._spy_df.empty else None
        render_pullback_vcp(self._canvas_pv.figure, df, symbol=self._symbol, spy_df=spy_df)
        self._canvas_pv.draw()

    def _update_move_otm(self):
        df = self._full_df
        if df.empty:
            return
        render_move_otm(self._canvas_mo.figure, df, symbol=self._symbol)
        self._canvas_mo.draw()

    def _update_pead(self):
        if not self._symbol:
            return
        render_pead(self._canvas_pead.figure, symbol=self._symbol)
        self._canvas_pead.draw()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _year_range(self):
        try:
            return int(self._start_year_cb.currentText()), int(self._end_year_cb.currentText())
        except Exception:
            return None, None

    def _update_y_views(self):
        df = self._chart_state['df']
        if df is None or not self._active_plots:
            return
        vr = self._active_plots[0].viewRange()[0]
        _auto_scale_panes(self._active_plots, df, vr[0], vr[1])

    def _wire_crosshair(self, plots, df):
        """Set up crosshair lines, hover labels, and mouse proxy for the 6-pane chart."""
        from finance.utils.chart_styles import (
            MA_CONFIGS, BB_CONFIGS, VOL_CONFIGS,
            ATR_CONFIGS, HV_CONFIGS, IVPCT_CONFIGS,
        )

        crosshair_pen = pg.mkPen(color=(128, 128, 128, 100), width=1,
                                  style=QtCore.Qt.PenStyle.DashLine)
        v_lines, h_lines = [], []
        for p in plots:
            v = pg.InfiniteLine(angle=90,  movable=False, pen=crosshair_pen)
            h = pg.InfiniteLine(angle=0,   movable=False, pen=crosshair_pen)
            p.addItem(v, ignoreBounds=True)
            p.addItem(h, ignoreBounds=True)
            v_lines.append(v)
            h_lines.append(h)
            h.hide()

        hover_labels = []
        for p in plots:
            lbl = pg.TextItem(anchor=(0, 0), color='#ccc', fill='#000e')
            p.addItem(lbl, ignoreBounds=True)
            hover_labels.append(lbl)

        self._chart_state['v_lines']      = v_lines
        self._chart_state['h_lines']      = h_lines
        self._chart_state['hover_labels'] = hover_labels

        x_dates = self._chart_state['x_dates']
        symbol  = self._symbol

        def _legend_html(row):
            """Build a single-line legend overlay shown above the OHLC chart."""
            date_str = ''
            if hasattr(row, 'name') and isinstance(row.name, pd.Timestamp):
                date_str = row.name.strftime('%a %Y-%m-%d')
            price = float(row.get('c', np.nan))
            parts = [
                f"<b>{symbol}</b>",
                f"<span style='color:#888;'>{date_str}</span>",
                f"O:<b>{row.o:.2f}</b> H:<b>{row.h:.2f}</b> L:<b>{row.l:.2f}</b> C:<b>{row.c:.2f}</b>",
            ]
            # MA entries with distance
            for col, cfg in MA_CONFIGS.items():
                if col not in df.columns:
                    continue
                val = row.get(col, np.nan)
                if not np.isfinite(val):
                    continue
                if np.isfinite(price) and val != 0:
                    dist_pct = (price - val) / val * 100
                    dist_str = f"({dist_pct:+.1f}%)"
                else:
                    dist_str = ''
                parts.append(
                    f"<span style='color:{cfg['color']};'>{col.upper()}</span>:<b>{val:.2f}</b>"
                    f"<span style='color:#888;'>{dist_str}</span>"
                )
            # BB entries
            for col, cfg in BB_CONFIGS.items():
                if col not in df.columns:
                    continue
                val = row.get(col, np.nan)
                if not np.isfinite(val):
                    continue
                label = 'BBU' if 'upper' in col else 'BBL'
                parts.append(
                    f"<span style='color:{cfg['color']};'>{label}</span>:<b>{val:.2f}</b>"
                )
            return (
                "<span style='font-size:10.5pt; color:#ddd;'>"
                + " &nbsp; ".join(parts) +
                "</span>"
            )

        def _fmt(cfg_dict, fmt_fn, label_fn=None, tx_fn=None, row=None):
            parts = []
            for col, cfg in cfg_dict.items():
                if col not in df.columns:
                    continue
                val = row.get(col, np.nan)
                if not np.isfinite(val):
                    continue
                if tx_fn:
                    val = tx_fn(val)
                color = cfg['color'] if isinstance(cfg, dict) and 'color' in cfg else str(cfg)
                label = label_fn(col) if label_fn else col
                parts.append(f"<span style='color:{color};'>{label}&nbsp;:&nbsp;<b>{fmt_fn(val)}</b></span>")
            return "<span style='font-size:10.5pt;'>" + '&nbsp;&nbsp;|&nbsp;&nbsp;'.join(parts) + "</span>"

        def update_hover(evt):
            pos = evt[0]
            mouse_pt = plots[0].vb.mapSceneToView(pos)
            idx = int(mouse_pt.x() + 0.5)
            if not (0 <= idx < len(df)):
                return
            row = df.iloc[idx]

            for v in v_lines:
                v.setPos(idx)
            for i, p in enumerate(plots):
                if p.sceneBoundingRect().contains(pos):
                    mp = p.vb.mapSceneToView(pos)
                    h_lines[i].setPos(mp.y())
                    h_lines[i].show()
                else:
                    h_lines[i].hide()

            # Pane 0: big legend-style overlay
            hover_labels[0].setHtml(_legend_html(row))

            # Pane 1: Volume
            vol_txt = ''
            if 'v' in df.columns and np.isfinite(row.get('v', np.nan)):
                vol_txt += f"<span style='color:#dddddd;'>VOL</span>&nbsp;:&nbsp;<b>{row.v / 1000:,.0f}k</b>"
            if 'v20' in df.columns and np.isfinite(row.get('v20', np.nan)):
                cfg = VOL_CONFIGS.get('v20', {})
                vol_txt += (f"&nbsp;&nbsp;|&nbsp;&nbsp;<span style='color:{cfg.get('color', '#ccc')};'>V20</span>"
                            f"&nbsp;:&nbsp;<b>{row.v20 / 1000:,.0f}k</b>")
            hover_labels[1].setHtml(f"<span style='font-size:10.5pt;'>{vol_txt}</span>")

            # Pane 2: ATR% + COTR
            atr_text = _fmt(ATR_CONFIGS, lambda v: f'{v:.2f}%', lambda c: c.upper(), row=row)
            if 'pct' in df.columns and 'atrp20' in df.columns:
                pct_val   = row.get('pct', np.nan)
                atr20_val = row.get('atrp20', np.nan)
                if np.isfinite(pct_val) and np.isfinite(atr20_val) and atr20_val != 0:
                    ratio = pct_val / atr20_val
                    cotr_html = f"<span style='color:#ffaf1c;'>COTR&nbsp;:&nbsp;<b>{ratio:+.2f}</b></span>"
                    atr_text = (atr_text + '&nbsp;&nbsp;||&nbsp;&nbsp;' + cotr_html) if atr_text else cotr_html
            hover_labels[2].setHtml(atr_text)

            # Pane 3: HV/IV + IVP combined
            hv_text = _fmt(HV_CONFIGS, lambda v: f'{v:.2f}', lambda c: c.upper(), row=row)
            ivp_text = _fmt(IVPCT_CONFIGS, lambda v: f'{v:.1f}', lambda c: c.upper(), row=row)
            combined = hv_text
            if ivp_text:
                combined = (combined + '&nbsp;&nbsp;||&nbsp;&nbsp;' + ivp_text) if combined else ivp_text
            hover_labels[3].setHtml(combined)

            # Pane 4: RS vs SPY
            rs_text = ''
            rs_series = getattr(plots[4], '_rs_series', None)
            rsma_series = getattr(plots[4], '_rsma_series', None)
            if rs_series is not None and 0 <= idx < len(rs_series):
                rs_val = float(rs_series[idx])
                if np.isfinite(rs_val):
                    rs_text = f"<span style='color:#dc9b56;'>RS</span>&nbsp;:&nbsp;<b>{rs_val:.4f}</b>"
            if rsma_series is not None and 0 <= idx < len(rsma_series):
                rsma_val = float(rsma_series[idx])
                if np.isfinite(rsma_val):
                    sep = '&nbsp;&nbsp;|&nbsp;&nbsp;' if rs_text else ''
                    rs_text += f"{sep}<span style='color:#865371;'>RSMA20</span>&nbsp;:&nbsp;<b>{rsma_val:.4f}</b>"
            hover_labels[4].setHtml(f"<span style='font-size:10.5pt;'>{rs_text}</span>")

            # Pane 5: TTM
            ttm_txt = ''
            if 'ttm_mom' in df.columns and np.isfinite(row.get('ttm_mom', np.nan)):
                ttm_txt += f"<span style='color:#ddd;'>TTM_MOM</span>&nbsp;:&nbsp;<b>{row.ttm_mom:.2f}</b>"
            if 'squeeze_on' in df.columns and pd.notna(row.get('squeeze_on', np.nan)):
                sq = bool(row.squeeze_on)
                ttm_txt += ((' &nbsp;&nbsp;|&nbsp;&nbsp; ' if ttm_txt else '') +
                            f"<span style='color:#ddd;'>SQUEEZE</span>&nbsp;:&nbsp;<b>{'ON' if sq else 'OFF'}</b>")
            hover_labels[5].setHtml(f"<span style='font-size:10.5pt;'>{ttm_txt}</span>")

            # Reposition labels — pane 0 at top-left, others top-left
            for i, p in enumerate(plots):
                vb_range = p.vb.viewRange()
                x_pos = vb_range[0][0] + (vb_range[0][1] - vb_range[0][0]) * 0.01
                y_pos = vb_range[1][1] - (vb_range[1][1] - vb_range[1][0]) * 0.01
                hover_labels[i].setPos(x_pos, y_pos)

        # Seed the legend with the latest bar so it is visible before any hover
        if len(df) > 0:
            hover_labels[0].setHtml(_legend_html(df.iloc[-1]))
            # Position once using current view
            for i, p in enumerate(plots):
                vb_range = p.vb.viewRange()
                x_pos = vb_range[0][0] + (vb_range[0][1] - vb_range[0][0]) * 0.01
                y_pos = vb_range[1][1] - (vb_range[1][1] - vb_range[1][0]) * 0.01
                hover_labels[i].setPos(x_pos, y_pos)

        proxy = pg.SignalProxy(plots[0].scene().sigMouseMoved, rateLimit=60, slot=update_hover)
        self._chart_state['proxy'] = proxy

    # ------------------------------------------------------------------
    # UI builders
    # ------------------------------------------------------------------

    def _build_toolbar(self) -> QtWidgets.QHBoxLayout:
        toolbar = QtWidgets.QHBoxLayout()

        ticker_label = QtWidgets.QLabel('Ticker:')
        self._ticker_input = QtWidgets.QLineEdit('SPY')
        self._ticker_input.setFixedWidth(60)
        toolbar.addWidget(ticker_label)
        toolbar.addWidget(self._ticker_input)

        ibkr_label = QtWidgets.QLabel('IBKR:')
        self._ibkr_combo = QtWidgets.QComboBox()
        # 'offline' = never refresh; others match instance names in finance.utils.ibkr.connect
        self._ibkr_combo.addItems(['offline', 'api_paper', 'paper', 'api', 'real'])
        self._ibkr_combo.setCurrentText('api_paper')
        self._ibkr_combo.currentTextChanged.connect(self._on_ibkr_instance_changed)
        toolbar.addWidget(ibkr_label)
        toolbar.addWidget(self._ibkr_combo)

        self._load_btn = QtWidgets.QPushButton('Load')
        toolbar.addWidget(self._load_btn)

        self._refresh_btn = QtWidgets.QPushButton('Refresh')
        self._refresh_btn.setToolTip('Sync current symbol from IBKR up to now')
        self._refresh_btn.clicked.connect(self._on_refresh_clicked)
        toolbar.addWidget(self._refresh_btn)
        toolbar.addSpacing(20)

        toolbar.addWidget(QtWidgets.QLabel('Start Year:'))
        self._start_year_cb = QtWidgets.QComboBox()
        toolbar.addWidget(self._start_year_cb)

        toolbar.addWidget(QtWidgets.QLabel('End Year:'))
        self._end_year_cb = QtWidgets.QComboBox()
        toolbar.addWidget(self._end_year_cb)
        toolbar.addSpacing(20)

        toolbar.addWidget(QtWidgets.QLabel('Timeframe:'))
        self._tree_tf_combo = QtWidgets.QComboBox()
        self._tree_tf_combo.addItems(['Daily', 'Weekly', 'Monthly'])
        toolbar.addWidget(self._tree_tf_combo)
        toolbar.addStretch()

        # Right side — data freshness for currently loaded symbol
        self._freshness_label = QtWidgets.QLabel('')
        self._freshness_label.setStyleSheet('color: #888888; font-weight: 600;')
        self._freshness_label.setToolTip('Cache freshness of currently loaded symbol')
        toolbar.addWidget(self._freshness_label)
        toolbar.addSpacing(10)

        # Right side — IBKR connection status indicator
        self._status_label = QtWidgets.QLabel('● IBKR')
        self._status_label.setStyleSheet('color: #ec4533; font-weight: 600;')
        self._status_label.setToolTip('IBKR connection status')
        toolbar.addWidget(self._status_label)

        return toolbar

    def _make_scroll_tab(self, fig_size: tuple, tab_label: str):
        """Create a scrollable matplotlib canvas tab and return (canvas, tab_widget)."""
        tab_widget = QtWidgets.QWidget()
        tab_widget.setStyleSheet('background-color: #111111;')
        tab_layout = QtWidgets.QVBoxLayout(tab_widget)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet('background-color: #111111; border: none;')
        content = QtWidgets.QWidget()
        content.setStyleSheet('background-color: #111111;')
        content_layout = QtWidgets.QVBoxLayout(content)

        fig    = Figure(figsize=fig_size, facecolor='#111111')
        fig.patch.set_facecolor('#111111')
        canvas = FigureCanvas(fig)
        canvas.setStyleSheet('background-color: #111111;')
        content_layout.addWidget(canvas)

        scroll.setWidget(content)
        tab_layout.addWidget(scroll)
        self._tabs.addTab(tab_widget, tab_label)

        return canvas, tab_widget

    # ------------------------------------------------------------------
    # Slot handlers
    # ------------------------------------------------------------------

    def _on_load_clicked(self):
        symbol = self._ticker_input.text().strip()
        if symbol:
            self.load_symbol(symbol)

    # ------------------------------------------------------------------
    # Post-show layout fix
    # ------------------------------------------------------------------

    def show(self):
        super().show()
        QtCore.QTimer.singleShot(0, self._after_show)

    def _after_show(self):
        self.resize(1600, 900)
        _force_layout_and_scene_sync(self._glw)
