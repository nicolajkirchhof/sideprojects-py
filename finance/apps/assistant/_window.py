"""
finance.apps.assistant._window
================================
Main window for the Trading Assistant.

Three-panel layout (QSplitter):
  Left  (300px)  — Market Context panel
  Centre (flex)  — Watchlist Table panel
  Right (350px)  — Detail Panel

Window geometry and splitter positions are persisted across restarts
via QSettings (key prefix: TradingAssistant/).
"""
from __future__ import annotations

import logging
import subprocess
from datetime import date, datetime
from pathlib import Path

import yaml

from pyqtgraph.Qt import QtCore, QtWidgets

from finance.apps.assistant._data import (
    _DRIFT_REGISTRY,
    compute_drift_tier,
    compute_drift_underlying,
    compute_go_nogo,
    compute_trend_status,
    compute_vix_status,
    load_daily,
)
from finance.apps.assistant._detail_panel import CandidateDetailPanel
from finance.apps.assistant._export import export_barchart, export_tws
from finance.apps.assistant._filter_bar import FilterBar
from finance.apps.assistant._filter_proxy import FilterState, WatchlistFilterProxy
from finance.apps.assistant._header_checkbox import CheckableHeader
from finance.apps.assistant._swing_panel import SwingRegimePanel
from finance.apps.assistant._watchlist_model import Col, WatchlistModel

log = logging.getLogger(__name__)

_SETTINGS_ORG = "sideprojects-py"
_SETTINGS_APP = "TradingAssistant"
_KEY_GEOMETRY = "geometry"
_KEY_SPLITTER = "splitter"

_DEFAULT_WIDTH = 1400
_DEFAULT_HEIGHT = 900
_LEFT_WIDTH = 300
_RIGHT_WIDTH = 350

# Explicit pixel widths for each column; SECTOR is omitted because it uses
# horizontalHeader().setStretchLastSection(True) to fill remaining space.
_COLUMN_WIDTHS: dict[Col, int] = {
    Col.CHECK:     28,
    Col.SYMBOL:    70,
    Col.DIRECTION: 30,
    Col.SCORE:     50,
    Col.D1:        40,
    Col.D2:        40,
    Col.D3:        40,
    Col.D4:        40,
    Col.D5:        40,
    Col.TAGS:     160,
    Col.PRICE:     65,
    Col.CHANGE_5D: 55,
    Col.RVOL:      50,
}

_SELECT_TOP_N: int = 20  # default for "Select Top N" toolbar button


class AssistantWindow(QtWidgets.QMainWindow):
    """
    Trading Assistant main window.

    Hosts three resizable panels and a toolbar. Panel content is populated
    by subsequent stories; this class provides the shell, layout, and
    pipeline orchestration.

    Attributes
    ----------
    _results:
        In-memory list of result rows (dicts) from the most recent
        pipeline run or cache load. Consumed by the watchlist table (TA-E4).
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Trading Assistant")
        self.resize(_DEFAULT_WIDTH, _DEFAULT_HEIGHT)

        self._settings = QtCore.QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        self._results: list[dict] = []
        self._pipeline_thread: QtCore.QThread | None = None
        self._claude_thread: QtCore.QThread | None = None
        self._analysis_thread: QtCore.QThread | None = None
        self._top_n_thread: QtCore.QThread | None = None
        self._positions_window = None
        self._watchlist_model = WatchlistModel(self)

        self._build_toolbar()
        self._build_panels()
        self._build_status_bar()
        self._restore_geometry()
        self._load_today_cache()
        self._load_regime_data()

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------

    def _build_toolbar(self) -> None:
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        tb.setIconSize(QtCore.QSize(20, 20))

        self._btn_run = QtWidgets.QToolButton()
        self._btn_run.setText("▶  Run Pipeline")
        self._btn_run.setToolTip("Enrich, score, and cache today's scanner candidates")
        self._btn_run.setMinimumWidth(130)
        self._btn_run.clicked.connect(lambda: self._on_run_pipeline())
        tb.addWidget(self._btn_run)

        tb.addSeparator()

        self._btn_load_csv = QtWidgets.QToolButton()
        self._btn_load_csv.setText("Load CSV…")
        self._btn_load_csv.setToolTip("Manually import a Barchart screener CSV")
        self._btn_load_csv.clicked.connect(self._on_load_csv)
        tb.addWidget(self._btn_load_csv)

        tb.addSeparator()

        self._btn_select_top_n = QtWidgets.QToolButton()
        self._btn_select_top_n.setText(f"▣  Top {_SELECT_TOP_N}")
        self._btn_select_top_n.setToolTip(f"Check the top {_SELECT_TOP_N} candidates by score")
        self._btn_select_top_n.clicked.connect(lambda: self._on_select_top_n())
        tb.addWidget(self._btn_select_top_n)

        tb.addSeparator()

        self._btn_export_bc = QtWidgets.QToolButton()
        self._btn_export_bc.setText("Export → Barchart")
        self._btn_export_bc.setToolTip("Copy selected tickers to clipboard (comma-separated)")
        self._btn_export_bc.setEnabled(False)
        self._btn_export_bc.clicked.connect(lambda: self._on_export_barchart())
        tb.addWidget(self._btn_export_bc)

        self._btn_export_tws = QtWidgets.QToolButton()
        self._btn_export_tws.setText("Export → TWS")
        self._btn_export_tws.setToolTip("Save selected tickers as a TWS-importable CSV")
        self._btn_export_tws.setEnabled(False)
        self._btn_export_tws.clicked.connect(lambda: self._on_export_tws())
        tb.addWidget(self._btn_export_tws)

        tb.addSeparator()

        self._btn_positions = QtWidgets.QToolButton()
        self._btn_positions.setText("Positions ↗")
        self._btn_positions.setToolTip("Open Position Management window")
        self._btn_positions.clicked.connect(self._on_open_positions)
        tb.addWidget(self._btn_positions)

        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        tb.addWidget(spacer)

        self._lbl_last_run = QtWidgets.QLabel("")
        self._lbl_last_run.setStyleSheet("color: #666; font-size: 11px; padding-right: 8px;")
        tb.addWidget(self._lbl_last_run)

    def _build_panels(self) -> None:
        self._splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        self._left_panel = SwingRegimePanel(self)
        self._centre_panel = self._build_watchlist_panel()
        self._right_panel = CandidateDetailPanel(self)

        self._splitter.addWidget(self._left_panel)
        self._splitter.addWidget(self._centre_panel)
        self._splitter.addWidget(self._right_panel)

        self._right_panel.analyze_requested.connect(self._start_candidate_analysis)

        total = _DEFAULT_WIDTH
        centre = total - _LEFT_WIDTH - _RIGHT_WIDTH
        self._splitter.setSizes([_LEFT_WIDTH, centre, _RIGHT_WIDTH])
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setStretchFactor(2, 0)

        self.setCentralWidget(self._splitter)

    def _build_watchlist_panel(self) -> QtWidgets.QWidget:
        """
        Build the centre panel containing the filter bar and scored-candidate table.

        Layout (top → bottom):
          FilterBar  — score, direction, tag, sector, text search controls
          QTableView — WatchlistModel behind WatchlistFilterProxy for sort+filter
        """
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._filter_bar = FilterBar(panel)
        layout.addWidget(self._filter_bar)

        proxy = WatchlistFilterProxy(self)
        proxy.setSourceModel(self._watchlist_model)

        header = CheckableHeader(self)
        header.toggle_all.connect(self._on_toggle_all)

        table = QtWidgets.QTableView()
        table.setHorizontalHeader(header)
        table.setModel(proxy)
        table.setSortingEnabled(True)
        table.sortByColumn(3, QtCore.Qt.SortOrder.DescendingOrder)  # Score desc
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        header.setStretchLastSection(True)
        table.setShowGrid(False)

        # Column widths driven by _COLUMN_WIDTHS constant; SECTOR stretches.
        for col, width in _COLUMN_WIDTHS.items():
            header.resizeSection(col, width)

        # Update export buttons + header checkbox whenever check state changes.
        self._watchlist_model.dataChanged.connect(self._on_model_data_changed)

        # Drive the detail panel from row selection.
        table.selectionModel().selectionChanged.connect(self._on_selection_changed)

        # Push filter state changes from the bar into the proxy.
        self._filter_bar.filters_changed.connect(proxy.update_filter)

        layout.addWidget(table)
        self._watchlist_table = table
        self._watchlist_proxy = proxy
        self._watchlist_header = header
        return panel

    def _build_status_bar(self) -> None:
        self._status_bar = self.statusBar()
        self._lbl_status = QtWidgets.QLabel("Ready")
        self._lbl_candidate_count = QtWidgets.QLabel("")
        self._status_bar.addWidget(self._lbl_status, 1)
        self._status_bar.addPermanentWidget(self._lbl_candidate_count)

    @staticmethod
    def _make_placeholder_panel(name: str) -> QtWidgets.QWidget:
        """Return a labelled placeholder widget for an unimplemented panel."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)

        lbl = QtWidgets.QLabel(name)
        lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("color: #555; font-size: 16px;")
        layout.addWidget(lbl)

        return widget

    # ------------------------------------------------------------------
    # QSettings — geometry persistence
    # ------------------------------------------------------------------

    def _restore_geometry(self) -> None:
        geom = self._settings.value(_KEY_GEOMETRY)
        if geom is not None:
            self.restoreGeometry(geom)

        splitter_state = self._settings.value(_KEY_SPLITTER)
        if splitter_state is not None:
            self._splitter.restoreState(splitter_state)

    def closeEvent(self, event: QtCore.QEvent) -> None:  # type: ignore[override]
        self._settings.setValue(_KEY_GEOMETRY, self.saveGeometry())
        self._settings.setValue(_KEY_SPLITTER, self._splitter.saveState())
        if self._pipeline_thread is not None and self._pipeline_thread.isRunning():
            self._pipeline_thread.quit()
            self._pipeline_thread.wait(3000)
        if self._claude_thread is not None and self._claude_thread.isRunning():
            self._claude_thread.quit()
            self._claude_thread.wait(3000)
        if self._analysis_thread is not None and self._analysis_thread.isRunning():
            self._analysis_thread.quit()
            self._analysis_thread.wait(3000)
        if self._top_n_thread is not None and self._top_n_thread.isRunning():
            self._top_n_thread.quit()
            self._top_n_thread.wait(3000)
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Filter bar helpers
    # ------------------------------------------------------------------

    def _update_filter_options(self, rows: list[dict]) -> None:
        """
        Rebuild the filter bar's tag and sector option menus from *rows*.

        Counts are computed from the full unfiltered result set so the user
        sees the universe size, not just what's currently visible.
        """
        tag_counts: dict[str, int] = {}
        sector_counts: dict[str, int] = {}

        for row in rows:
            for tag in row.get("tags") or []:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            sector = row.get("sector") or ""
            if sector:
                sector_counts[sector] = sector_counts.get(sector, 0) + 1

        self._filter_bar.update_options(tag_counts, sector_counts)

    # ------------------------------------------------------------------
    # Status bar helpers
    # ------------------------------------------------------------------

    def set_status(self, message: str) -> None:
        """Update the left status bar label."""
        self._lbl_status.setText(message)

    def set_candidate_count(self, count: int) -> None:
        """Update the permanent candidate-count label."""
        self._lbl_candidate_count.setText(f"{count} candidates" if count is not None else "")

    # ------------------------------------------------------------------
    # Cache loading on launch
    # ------------------------------------------------------------------

    def _load_today_cache(self) -> None:
        """Load today's JSON cache if it exists, skipping the pipeline."""
        from finance.apps.assistant._pipeline import (
            read_cache,
            read_events_from_cache,
            read_market_summary_from_cache,
        )

        rows = read_cache(date.today())
        if rows is not None:
            self._results = rows
            self._watchlist_model.load_rows(rows)
            self._update_filter_options(rows)
            ts = datetime.now().strftime("%H:%M")
            self.set_status(f"Loaded from cache ({ts})")
            self.set_candidate_count(len(rows))
            self._lbl_last_run.setText(f"Cache  {ts}")
            log.info("Loaded %d rows from today's cache", len(rows))

        events = read_events_from_cache(date.today())
        if events:
            self._left_panel.update_events(events)
            log.info("Loaded %d calendar events from cache", len(events))

        summary = read_market_summary_from_cache(date.today())
        if summary:
            self._left_panel.update_market_summary(summary)
            log.info("Loaded market summary from cache (regime=%s)", summary.get("regime", "?"))

    def _load_regime_data(self) -> None:
        """Load daily data for SPY, QQQ, and VIX and update the regime panel.

        Reads from the local IBKR parquet cache (offline=True).  Missing
        data yields None statuses which the panel renders as dashes.
        """
        try:
            spy_df = load_daily("SPY")
            qqq_df = load_daily("QQQ")
            vix_df = load_daily("$VIX")

            spy = compute_trend_status("SPY", spy_df) if spy_df is not None else None
            qqq = compute_trend_status("QQQ", qqq_df) if qqq_df is not None else None
            vix = compute_vix_status(vix_df) if vix_df is not None else None
            status = compute_go_nogo(spy, vix)

            self._spy_status = spy
            self._qqq_status = qqq
            self._vix_status = vix
            self._regime_status = status

            self._left_panel.update_indicators(spy, qqq, vix, status)
            log.info("Regime panel updated: %s", status)

            # DRIFT regime
            drift_tier = compute_drift_tier(spy_df, vix)
            drift_underlyings = [
                compute_drift_underlying(sym, load_daily(sym), block, reg_tier)
                for sym, block, reg_tier in _DRIFT_REGISTRY
            ]
            self._left_panel.update_drift(drift_tier, drift_underlyings)
            log.info("DRIFT tier: %s (dd=%.1f%%)", drift_tier.name, drift_tier.drawdown_pct)
        except Exception:
            log.exception("Failed to load regime data — panel shows dashes")

    # ------------------------------------------------------------------
    # Pipeline orchestration
    # ------------------------------------------------------------------

    def _on_run_pipeline(self, *, csv_paths: list[Path] | None = None) -> None:
        """Start the background pipeline thread."""
        from finance.apps.assistant._pipeline import PipelineThread

        if self._pipeline_thread is not None and self._pipeline_thread.isRunning():
            return  # already running — ignore

        thread = PipelineThread(csv_paths=csv_paths)
        thread.stage_changed.connect(self._on_pipeline_stage)
        thread.candidate_count_changed.connect(self.set_candidate_count)
        thread.finished_ok.connect(self._on_pipeline_finished)
        thread.calendar_updated.connect(self._on_calendar_updated)
        thread.error.connect(self._on_pipeline_error)

        self._pipeline_thread = thread
        self._set_pipeline_running(True)
        thread.start()

    def _on_load_csv(self) -> None:
        """Open a file dialog and run the pipeline with the selected CSV(s)."""
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select Barchart Screener CSV(s)",
            str(Path("finance/_data/barchart/screener")),
            "CSV files (*.csv)",
        )
        if not paths:
            return
        self._on_run_pipeline(csv_paths=[Path(p) for p in paths])

    def _on_pipeline_stage(self, message: str) -> None:
        self.set_status(message)

    def _on_calendar_updated(self, events: list) -> None:
        """Update the left panel with freshly fetched calendar events."""
        self._left_panel.update_events(events)

    def _on_pipeline_finished(self, rows: object) -> None:
        """Handle successful pipeline completion."""
        result_rows: list[dict] = rows  # type: ignore[assignment]
        self._results = result_rows
        self._watchlist_model.load_rows(result_rows)
        self._update_filter_options(result_rows)
        ts = datetime.now().strftime("%H:%M:%S")
        self.set_status(f"Done — {len(result_rows)} candidates  ({ts})")
        self._lbl_last_run.setText(f"Run  {ts}")
        self._set_pipeline_running(False)
        self._pipeline_thread = None
        log.info("Pipeline finished: %d candidates", len(result_rows))
        self._start_claude_summary()
        self._start_top_n_analysis(result_rows)

    def _on_pipeline_error(self, message: str) -> None:
        """Handle pipeline failure — show error dialog and re-enable UI."""
        from finance.apps.assistant._error_dialog import show_pipeline_error

        self.set_status("Pipeline failed — see error dialog")
        self._set_pipeline_running(False)
        self._pipeline_thread = None
        show_pipeline_error(self, message)

    def _load_claude_config(self) -> dict:
        """Read and return the ``claude:`` sub-dict from config.yaml."""
        cfg_path = Path(__file__).parent / "config.yaml"
        try:
            raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except Exception:
            raw = {}
        return raw.get("claude", {})

    def _start_claude_summary(self) -> None:
        """Fetch today's market emails and start the Claude summary thread."""
        from finance.apps.analyst._config import load_config
        from finance.apps.assistant._gmail import fetch_email_bodies
        from finance.apps.assistant._pipeline import ClaudeSummaryThread

        if self._claude_thread is not None and self._claude_thread.isRunning():
            return

        claude_cfg = self._load_claude_config()
        model = claude_cfg.get("model_summary", "claude-sonnet-4-6")
        max_emails = int(claude_cfg.get("max_emails", 10))

        try:
            config = load_config()
            emails = fetch_email_bodies(config.gmail, date.today(), max_emails=max_emails)
        except Exception:
            log.warning("Failed to fetch email bodies for Claude summary", exc_info=True)
            self._left_panel.summary_widget.set_error("Email fetch failed — check Gmail credentials")
            return

        if not emails:
            log.info("No market emails found — skipping Claude summary")
            self._left_panel.summary_widget.set_error("No TradeAnalyst emails found for today")
            return

        self._left_panel.summary_widget.set_generating()

        thread = ClaudeSummaryThread(
            emails=emails,
            model=model,
            trade_date=date.today(),
            spy_status=getattr(self, "_spy_status", None),
            qqq_status=getattr(self, "_qqq_status", None),
            vix_status=getattr(self, "_vix_status", None),
            regime_status=getattr(self, "_regime_status", ""),
        )
        thread.summary_ready.connect(self._on_summary_ready)
        thread.error.connect(self._on_summary_error)
        self._claude_thread = thread
        thread.start()

    def _on_summary_ready(self, summary_dict: dict) -> None:
        """Display the Claude summary and cache it for today."""
        from finance.apps.assistant._pipeline import update_cache_market_summary

        self._left_panel.update_market_summary(summary_dict)
        update_cache_market_summary(date.today(), summary_dict)
        self._claude_thread = None
        log.info("Market summary ready: regime=%s", summary_dict.get("regime", "?"))

    def _on_summary_error(self, message: str) -> None:
        """Log Claude summary failure and show error state in the summary widget."""
        self._left_panel.summary_widget.set_error(f"Summary error: {message}")
        self._claude_thread = None
        log.warning("Claude summary failed: %s", message)

    def _start_candidate_analysis(self, row: dict) -> None:
        """Start a background Claude analysis for *row* (triggered by Analyze button)."""
        from finance.apps.assistant._pipeline import CandidateAnalysisThread

        if self._analysis_thread is not None and self._analysis_thread.isRunning():
            return  # analysis already in progress — ignore

        claude_cfg = self._load_claude_config()
        model = claude_cfg.get("model_analysis", "claude-haiku-4-5-20251001")

        self._right_panel.set_analyzing(True)
        thread = CandidateAnalysisThread(row=row, model=model)
        thread.analysis_ready.connect(self._on_analysis_ready)
        thread.error.connect(self._on_analysis_error)
        self._analysis_thread = thread
        thread.start()

    def _on_analysis_ready(self, analysis_dict: dict) -> None:
        """Display Claude analysis in the detail panel and cache it."""
        from finance.apps.assistant._pipeline import update_cache_candidate_analysis

        self._right_panel.set_analyzing(False)
        self._right_panel.load_reasoning(analysis_dict)
        self._analysis_thread = None

        current_row = self._right_panel._current_row
        symbol = current_row.get("symbol", "") if current_row else ""
        if symbol:
            update_cache_candidate_analysis(date.today(), symbol, analysis_dict)
        log.info("Candidate analysis ready for %s", symbol)

    def _on_analysis_error(self, message: str) -> None:
        """Re-enable the Analyze button and surface the error to the user."""
        from finance.apps.assistant._error_dialog import show_pipeline_error
        self._right_panel.set_analyzing(False)
        self._analysis_thread = None
        log.warning("Candidate analysis failed: %s", message)
        show_pipeline_error(self, message)

    def _start_top_n_analysis(self, rows: list[dict]) -> None:
        """Start auto-analysis for the top-N candidates after pipeline completes."""
        from finance.apps.assistant._pipeline import TopNAnalysisThread

        claude_cfg = self._load_claude_config()
        model = claude_cfg.get("model_analysis", "claude-haiku-4-5-20251001")
        top_n = int(claude_cfg.get("top_n_auto_analyze", 0))

        if top_n <= 0 or not rows:
            return

        thread = TopNAnalysisThread(rows=rows, model=model, top_n=top_n)
        thread.row_analysis_ready.connect(self._on_top_n_row_ready)
        thread.error.connect(self._on_top_n_analysis_error)
        self._top_n_thread = thread
        thread.start()
        log.info("Auto-analysis started for top %d candidates", top_n)

    def _on_top_n_row_ready(self, symbol: str, analysis_dict: object) -> None:
        """Cache each auto-analysed row as it arrives."""
        from finance.apps.assistant._pipeline import update_cache_candidate_analysis

        update_cache_candidate_analysis(date.today(), symbol, analysis_dict)  # type: ignore[arg-type]
        log.info("Auto-analysis cached for %s", symbol)

    def _on_top_n_analysis_error(self, message: str) -> None:
        """Log a structural failure of the top-N analysis thread."""
        self._top_n_thread = None
        log.warning("Top-N analysis thread failed: %s", message)

    def _set_pipeline_running(self, running: bool) -> None:
        """Toggle UI controls while the pipeline is active."""
        self._btn_run.setEnabled(not running)
        self._btn_load_csv.setEnabled(not running)
        if running:
            self._btn_run.setText("⏳  Running…")
        else:
            self._btn_run.setText("▶  Run Pipeline")

    # ------------------------------------------------------------------
    # Checkbox batch selection
    # ------------------------------------------------------------------

    def _on_select_top_n(self) -> None:
        """Check the top N candidates by score and update UI."""
        self._watchlist_model.check_top_n(_SELECT_TOP_N)
        self._update_checked_ui()

    def _on_toggle_all(self, check: bool) -> None:
        """Check or uncheck all currently visible (proxy) rows."""
        if not check:
            self._watchlist_model.uncheck_all()
        else:
            proxy = self._watchlist_proxy
            source_indices = [
                proxy.mapToSource(proxy.index(i, 0)).row()
                for i in range(proxy.rowCount())
            ]
            self._watchlist_model.check_rows(source_indices)
        self._update_checked_ui()

    def _on_model_data_changed(
        self,
        top_left: QtCore.QModelIndex,
        bottom_right: QtCore.QModelIndex,
        roles: list[int],
    ) -> None:
        """React to model data changes — only care about CheckStateRole."""
        _CheckStateRole = QtCore.Qt.ItemDataRole.CheckStateRole
        if not roles or _CheckStateRole in roles:
            self._update_checked_ui()

    def _update_checked_ui(self) -> None:
        """Sync export buttons and header checkbox state with current check counts."""
        checked = self._watchlist_model.checked_count()
        visible = self._watchlist_proxy.rowCount()
        total = self._watchlist_model.rowCount()

        # Export buttons — enabled whenever at least one row is checked
        has_checked = checked > 0
        self._btn_export_bc.setEnabled(has_checked)
        self._btn_export_tws.setEnabled(has_checked)

        # Header checkbox tristate
        if checked == 0:
            state = QtCore.Qt.CheckState.Unchecked
        elif checked >= visible and visible > 0:
            state = QtCore.Qt.CheckState.Checked
        else:
            state = QtCore.Qt.CheckState.PartiallyChecked
        self._watchlist_header.set_check_state(state)

        # Status bar candidate count — show selection when non-zero
        if checked > 0:
            self._lbl_candidate_count.setText(f"{total} candidates | {checked} selected")
        else:
            self._lbl_candidate_count.setText(f"{total} candidates" if total else "")

    def _on_selection_changed(self) -> None:
        """Load the selected row's data into the detail panel, plus cached analysis."""
        from finance.apps.assistant._pipeline import read_candidate_analysis_from_cache

        indexes = self._watchlist_table.selectionModel().selectedRows()
        if not indexes:
            self._right_panel.load_row(None)
            return
        proxy_index = indexes[0]
        source_index = self._watchlist_proxy.mapToSource(proxy_index)
        row = self._watchlist_model._rows[source_index.row()]
        self._right_panel.load_row(row)

        symbol = row.get("symbol", "")
        cached = read_candidate_analysis_from_cache(date.today(), symbol)
        if cached:
            self._right_panel.load_reasoning(cached)

    # ------------------------------------------------------------------
    # Export handlers — TA-E6-S1 / TA-E6-S2
    # ------------------------------------------------------------------

    def _on_export_barchart(self) -> None:
        """Copy checked tickers to clipboard and save to txt file."""
        symbols = self._watchlist_model.checked_symbols()
        if not symbols:
            return
        export_barchart(symbols, date.today())
        QtWidgets.QApplication.clipboard().setText(",".join(symbols))
        self._show_toast(f"{len(symbols)} tickers copied to clipboard")
        log.info("Barchart export: %d symbols", len(symbols))

    def _on_export_tws(self) -> None:
        """Write TWS-importable CSV and open containing folder in Explorer."""
        symbols = self._watchlist_model.checked_symbols()
        if not symbols:
            return
        path = export_tws(symbols, date.today())
        try:
            subprocess.Popen(["explorer", f"/select,{path}"])
        except Exception:
            log.warning("Failed to open Explorer for %s", path, exc_info=True)
        self._show_toast(f"TWS file saved: {path.name}")
        log.info("TWS export: %d symbols → %s", len(symbols), path)

    # ------------------------------------------------------------------
    # Position Management window — TA-E7-S5
    # ------------------------------------------------------------------

    def _on_open_positions(self) -> None:
        """Open (or raise) the Position Management window."""
        from finance.apps.assistant._positions_window import PositionManagementWindow

        if self._positions_window is None or not self._positions_window.isVisible():
            self._positions_window = PositionManagementWindow(
                spy_status=getattr(self, "_spy_status", None),
                qqq_status=getattr(self, "_qqq_status", None),
                vix_status=getattr(self, "_vix_status", None),
                regime_status=getattr(self, "_regime_status", ""),
                parent=None,  # free-floating window
            )
            self._positions_window.show()
        else:
            self._positions_window.raise_()
            self._positions_window.activateWindow()
            # Push fresh regime state each time the window is raised
            self._positions_window.update_regime(
                spy_status=getattr(self, "_spy_status", None),
                qqq_status=getattr(self, "_qqq_status", None),
                vix_status=getattr(self, "_vix_status", None),
                regime_status=getattr(self, "_regime_status", ""),
            )

    def _show_toast(self, message: str, duration_ms: int = 3000) -> None:
        """Display a timed status bar message that resets to 'Ready' after duration."""
        self.set_status(message)
        QtCore.QTimer.singleShot(duration_ms, lambda: self.set_status("Ready"))
