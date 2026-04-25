"""
finance.apps.assistant._positions_window
==========================================
Position Management Window (TA-E7-S5).

Two-tab QMainWindow:
  Tab 1: Live Positions — open IBKR positions with R-multiples + rule alerts
  Tab 2: Trade Review   — closed Tradelog trades with rule flags + Claude review

Regime context (spy_status, qqq_status, vix_status, regime_status) is
passed in at construction and forwarded to the panels for rule evaluation.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pyqtgraph.Qt import QtCore, QtWidgets

if TYPE_CHECKING:
    from finance.apps.assistant._data import TrendStatus, VixStatus

_DEFAULT_WIDTH  = 1000
_DEFAULT_HEIGHT = 600

_SETTINGS_ORG = "sideprojects-py"
_SETTINGS_APP = "PositionManagement"
_KEY_GEOMETRY  = "geometry"


def _load_config() -> dict:
    cfg_path = Path(__file__).parent / "config.yaml"
    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        raw = {}
    return raw


class PositionManagementWindow(QtWidgets.QMainWindow):
    """
    Standalone window for trade management.

    Parameters
    ----------
    spy_status, qqq_status, vix_status:
        Optional regime data objects forwarded to option rules.
    regime_status:
        Current GO/NO-GO string.
    """

    def __init__(
        self,
        *,
        spy_status=None,
        qqq_status=None,
        vix_status=None,
        regime_status: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Position Management")
        self.resize(_DEFAULT_WIDTH, _DEFAULT_HEIGHT)

        self._spy_status = spy_status
        self._qqq_status = qqq_status
        self._vix_status = vix_status
        self._regime_status = regime_status

        self._settings = QtCore.QSettings(_SETTINGS_ORG, _SETTINGS_APP)

        cfg = _load_config()
        tradelog_cfg = cfg.get("tradelog", {})
        self._tradelog_url = tradelog_cfg.get("base_url", "http://localhost:5186")

        claude_cfg = cfg.get("claude", {})
        self._claude_model = claude_cfg.get("model_analysis", "claude-haiku-4-5-20251001")

        ibkr_cfg = cfg.get("ibkr", {})
        self._ibkr_host = ibkr_cfg.get("host", "127.0.0.1")
        self._ibkr_port = int(ibkr_cfg.get("port", 7497))

        self._build_ui()
        self._restore_geometry()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        from finance.apps.assistant._positions_panel import LivePositionsPanel
        from finance.apps.assistant._trade_review_panel import TradeReviewPanel

        tabs = QtWidgets.QTabWidget(self)

        self._live_panel = LivePositionsPanel(
            ibkr_host=self._ibkr_host,
            ibkr_port=self._ibkr_port,
            regime_status=self._regime_status,
            parent=tabs,
        )
        tabs.addTab(self._live_panel, "Live Positions")

        self._review_panel = TradeReviewPanel(
            tradelog_base_url=self._tradelog_url,
            claude_model=self._claude_model,
            regime_status=self._regime_status,
            parent=tabs,
        )
        tabs.addTab(self._review_panel, "Trade Review")

        self.setCentralWidget(tabs)

    # ------------------------------------------------------------------
    # Geometry persistence
    # ------------------------------------------------------------------

    def _restore_geometry(self) -> None:
        geom = self._settings.value(_KEY_GEOMETRY)
        if geom is not None:
            self.restoreGeometry(geom)

    def closeEvent(self, event: QtCore.QEvent) -> None:  # type: ignore[override]
        self._settings.setValue(_KEY_GEOMETRY, self.saveGeometry())
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Public API — allow main window to push regime updates
    # ------------------------------------------------------------------

    def update_regime(
        self,
        *,
        spy_status=None,
        qqq_status=None,
        vix_status=None,
        regime_status: str = "",
    ) -> None:
        """Update regime context and propagate to child panels."""
        self._spy_status = spy_status
        self._qqq_status = qqq_status
        self._vix_status = vix_status
        self._regime_status = regime_status
        self._live_panel.set_regime_status(regime_status)
        self._review_panel.set_regime_status(regime_status)
