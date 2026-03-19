"""
finance.visualizations._app
=============================
SwingPlotWindow — the main QMainWindow subclass.
All UI state is stored as instance variables; no module-level globals.
Signal wiring happens once in __init__; load_symbol() updates data and
calls update_all() without rebuilding the window.
"""
import sys

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph import QtCore
from pyqtgraph.Qt import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ._chart import (
    _setup_plot_panes, _add_plot_content, _auto_scale_panes,
    _force_layout_and_scene_sync,
)
from ._tabs import (
    render_stats_violins,
    render_volatility_analysis,
    render_drawdown_analysis,
    render_move_character,
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

        # Tab 3: Daily / Monthly Stats
        self._canvas_stats, stats_tab = self._make_scroll_tab(
            fig_size=(12, 18), tab_label='Daily/Monthly Stats')

        # Tab 4: Volatility Analysis
        self._canvas_vol, vol_tab = self._make_scroll_tab(
            fig_size=(12, 22), tab_label='Volatility Analysis')

        # Tab 5: Drawdown Analysis
        self._canvas_dd, dd_tab = self._make_scroll_tab(
            fig_size=(12, 18), tab_label='Drawdown Analysis')

        # Tab 6: Move Character
        self._canvas_mc, mc_tab = self._make_scroll_tab(
            fig_size=(24, 90), tab_label='Move Character')

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_symbol(self, symbol: str, datasource: str = 'offline'):
        """Load a new symbol (and SPY overlay) then refresh all tabs."""
        from finance.utils.swing_trading_data import SwingTradingData

        symbol = symbol.upper()
        try:
            data = SwingTradingData(symbol, datasource=datasource)
            if data.df_day is None or data.df_day.empty:
                QtWidgets.QMessageBox.warning(
                    self, 'Load Error', f'No data found for {symbol} ({datasource})')
                return
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to load {symbol}: {e}')
            return

        self._full_df = data.df_day
        self._symbol  = symbol
        self._title   = symbol
        self.setWindowTitle(f'Swing Trading Analysis — {symbol}')

        # Load SPY for overlay (skip if symbol IS SPY)
        if symbol != 'SPY':
            try:
                spy_data     = SwingTradingData('SPY', datasource=datasource)
                self._spy_df = spy_data.df_day if spy_data.df_day is not None else pd.DataFrame()
            except Exception:
                self._spy_df = pd.DataFrame()
        else:
            self._spy_df = pd.DataFrame()

        # Reset year filters
        all_years = sorted(self._full_df.index.year.unique())
        self._start_year_cb.blockSignals(True)
        self._end_year_cb.blockSignals(True)
        self._start_year_cb.clear()
        self._end_year_cb.clear()
        self._start_year_cb.addItems([str(y) for y in all_years])
        self._end_year_cb.addItems([str(y) for y in all_years])
        self._start_year_cb.setCurrentText(str(all_years[0]))
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
        self._update_move_character()

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
        for p in self._active_plots:
            p.deleteLater()
        self._glw.clear()
        self._active_plots = []

        title_item = pg.LabelItem(justify='center', size='14pt')
        title_item.setText(self._title)
        self._glw.addItem(title_item, row=0, col=0)
        self._glw.ci.layout.setRowStretchFactor(0, 3)

        df = self._filtered_df()
        if df.empty or not isinstance(df.index, pd.DatetimeIndex):
            self._glw.addItem(
                pg.LabelItem('No Data Available', justify='center', size='12pt'), row=1, col=0)
            return

        self._chart_state['df']      = df
        self._chart_state['x_dates'] = df.index

        plots = _setup_plot_panes(self._glw, df.index, row_offset=1)
        self._active_plots = plots

        spy_df = self._spy_df if not self._spy_df.empty else None
        _add_plot_content(plots, df, vlines=None, spy_df=spy_df)

        self._wire_crosshair(plots, df)

        plots[0].sigXRangeChanged.connect(self._update_y_views)
        plots[0].setXRange(max(0, len(df) - self.DISPLAY_RANGE), len(df))
        self._update_y_views()

    def _update_trees(self):
        from finance.utils.plots import plot_probability_tree
        df = self._filtered_df()
        if df.empty:
            return

        canv = self._canvas_trees
        canv.figure.clear()
        canv.figure.patch.set_facecolor('#111111')
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
        render_volatility_analysis(self._canvas_vol.figure, df, sy, ey)
        self._canvas_vol.draw()

    def _update_drawdown(self):
        df = self._filtered_df()
        sy, ey = self._year_range()
        if df.empty or sy is None:
            return
        render_drawdown_analysis(self._canvas_dd.figure, df, sy, ey)
        self._canvas_dd.draw()

    def _update_move_character(self):
        df = self._full_df   # use full history for character analysis (not year-filtered)
        if df.empty:
            return
        render_move_character(self._canvas_mc.figure, df, symbol=self._symbol)
        self._canvas_mc.draw()

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
        """Set up crosshair lines, hover labels, and mouse proxy for a new chart."""
        from ._config import (
            MA_CONFIGS, BB_CONFIGS, VOL_CONFIGS, DIST_CONFIGS,
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

        def _fmt_group(config_dict, value_fmt, col_label_fn=None, transform_fn=None):
            parts = []
            for col, cfg in config_dict.items():
                if col not in df.columns:
                    continue
                val = df.iloc[0].get(col, np.nan)  # placeholder; real value set in update_hover
                color = cfg['color'] if isinstance(cfg, dict) and 'color' in cfg else str(cfg)
                label = col_label_fn(col) if col_label_fn else col
                parts.append((col, color, label))
            return parts

        x_dates = self._chart_state['x_dates']

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

            date_str = x_dates[idx].strftime('%a %Y-%m-%d')

            def _fmt(cfg_dict, fmt_fn, label_fn=None, tx_fn=None):
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
                    parts.append(f"<span style='color:{color};'>{label}:{fmt_fn(val)}</span>")
                return ' | '.join(parts)

            # Pane 0: OHLC
            txt0 = (f"<span style='font-size:11pt; color:white; font-weight:bold;'>{date_str}</span> "
                    f"O:{row.o:.2f} H:{row.h:.2f} L:{row.l:.2f} C:{row.c:.2f}")
            if 'v' in df.columns and pd.notna(row.get('v', np.nan)):
                txt0 += f" V:{row.v / 1000:,.0f}k"
            emas = _fmt(MA_CONFIGS, lambda v: f'{v:.2f}', lambda c: c.upper())
            bbs  = _fmt(BB_CONFIGS, lambda v: f'{v:.2f}', lambda c: c.upper())
            if emas: txt0 += ' | ' + emas
            if bbs:  txt0 += ' | ' + bbs
            hover_labels[0].setHtml(txt0)

            # Pane 1: Volume
            hover_labels[1].setHtml(_fmt(VOL_CONFIGS, lambda v: f'{v:.2f}k',
                                          lambda c: c.upper(), lambda v: v / 1000.0))
            # Pane 2: EMA Dist
            hover_labels[2].setHtml(_fmt(DIST_CONFIGS, lambda v: f'{v:.2f}',
                                          lambda c: c.replace('_dist', '').upper()))
            # Pane 3: ATR
            hover_labels[3].setHtml(_fmt(ATR_CONFIGS, lambda v: f'{v:.2f}%', lambda c: c.upper()))

            # Pane 4: ATR Ratio
            if 'pct' in df.columns and 'atrp20' in df.columns:
                pct_val   = row.get('pct', np.nan)
                atr20_val = row.get('atrp20', np.nan)
                if np.isfinite(pct_val) and np.isfinite(atr20_val) and atr20_val != 0:
                    ratio = pct_val / atr20_val
                    color = '#f5a623' if abs(ratio) >= 1.75 else '#e0e0e0'
                    hover_labels[4].setHtml(
                        f"<span style='color:{color};'>ATR_Ratio:{ratio:.2f}</span>")
                else:
                    hover_labels[4].setHtml('')
            else:
                hover_labels[4].setHtml('')

            # Pane 5: HV/IV
            hover_labels[5].setHtml(_fmt(HV_CONFIGS, lambda v: f'{v:.2f}', lambda c: c.upper()))
            # Pane 6: IV Pct
            hover_labels[6].setHtml(_fmt(IVPCT_CONFIGS, lambda v: f'{v:.2f}', lambda c: c))

            # Pane 7: TTM
            ttm_txt = ''
            if 'ttm_mom' in df.columns and np.isfinite(row.get('ttm_mom', np.nan)):
                ttm_txt += f"TTM_MOM:{row.ttm_mom:.2f}"
            if 'squeeze_on' in df.columns and pd.notna(row.get('squeeze_on', np.nan)):
                sq = bool(row.squeeze_on)
                ttm_txt += (' | ' if ttm_txt else '') + f"SQUEEZE:{'ON' if sq else 'OFF'}"
            hover_labels[7].setHtml(ttm_txt)

            # Reposition labels
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

        from finance.utils.swing_trading_data import DATASOURCES
        ds_label      = QtWidgets.QLabel('Source:')
        self._ds_combo = QtWidgets.QComboBox()
        self._ds_combo.addItems(DATASOURCES)
        self._ds_combo.setCurrentText('offline')
        toolbar.addWidget(ds_label)
        toolbar.addWidget(self._ds_combo)

        self._load_btn = QtWidgets.QPushButton('Load')
        toolbar.addWidget(self._load_btn)
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

        return toolbar

    def _make_scroll_tab(self, fig_size: tuple, tab_label: str):
        """Create a scrollable matplotlib canvas tab and return (canvas, tab_widget)."""
        tab_widget = QtWidgets.QWidget()
        tab_layout = QtWidgets.QVBoxLayout(tab_widget)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)

        fig    = Figure(figsize=fig_size)
        fig.patch.set_facecolor('#111111')
        canvas = FigureCanvas(fig)
        content_layout.addWidget(canvas)

        scroll.setWidget(content)
        tab_layout.addWidget(scroll)
        self._tabs.addTab(tab_widget, tab_label)

        return canvas, tab_widget

    # ------------------------------------------------------------------
    # Slot handlers
    # ------------------------------------------------------------------

    def _on_load_clicked(self):
        symbol     = self._ticker_input.text().strip()
        datasource = self._ds_combo.currentText()
        if symbol:
            self.load_symbol(symbol, datasource)

    # ------------------------------------------------------------------
    # Post-show layout fix
    # ------------------------------------------------------------------

    def show(self):
        super().show()
        QtCore.QTimer.singleShot(0, self._after_show)

    def _after_show(self):
        self.resize(1600, 900)
        _force_layout_and_scene_sync(self._glw)
