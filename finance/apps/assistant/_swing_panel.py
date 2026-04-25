"""
finance.apps.assistant._swing_panel
======================================
Left panel — Swing Trading Regime indicators.

Shows: SPY/QQQ trend rows, VIX row, composite GO/NO-GO banner,
macro event proximity warning, and economic events calendar.

Migrated from finance.apps.conditions._swing_panel; extended for
TA-E3-S3 (events calendar).
"""
from __future__ import annotations

from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

from finance.apps.assistant._data import (
    DriftTier,
    DriftUnderlyingStatus,
    TrendStatus,
    VixStatus,
)

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

GREEN = "#4CAF50"
AMBER = "#FFA726"
RED = "#F44336"
DIM = "#666"

_ZONE_COLOURS = {"low": GREEN, "elevated": AMBER, "high": RED}
_SLOPE_ARROWS = {"rising": "\u2197", "flat": "\u2192", "falling": "\u2198"}  # ↗ → ↘
_DIRECTION_ARROWS = {"falling": "\u2193", "rising": "\u2191", "spiking": "\u26a0"}  # ↓ ↑ ⚠

_GONOGO_STYLES = {
    "GO": (GREEN, "#111"),
    "CAUTION": (AMBER, "#111"),
    "NO-GO": (RED, "#fff"),
}

MONO_FONT = "Roboto Mono"


# ---------------------------------------------------------------------------
# Reusable helpers
# ---------------------------------------------------------------------------

def _dot(colour: str) -> str:
    return f'<span style="color:{colour}; font-size:18px;">\u25cf</span>'


def _label(text: str = "", mono: bool = False, size: int = 13) -> QtWidgets.QLabel:
    lbl = QtWidgets.QLabel(text)
    font_family = MONO_FONT if mono else "Segoe UI"
    lbl.setFont(QtGui.QFont(font_family, size))
    lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)
    return lbl


def _hsep() -> QtWidgets.QFrame:
    line = QtWidgets.QFrame()
    line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
    line.setStyleSheet("color: #333;")
    return line


# ---------------------------------------------------------------------------
# TrendRow — one row per symbol (SPY / QQQ)
# ---------------------------------------------------------------------------

class TrendRow(QtWidgets.QWidget):
    """Displays: SYMBOL  price  50d SMA ●↗  200d SMA ●↗"""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        h = QtWidgets.QHBoxLayout(self)
        h.setContentsMargins(0, 2, 0, 2)
        h.setSpacing(12)

        self._sym = _label(mono=False, size=13)
        self._sym.setFixedWidth(48)
        self._sym.setStyleSheet("font-weight: bold;")
        h.addWidget(self._sym)

        self._price = _label(mono=True, size=13)
        self._price.setFixedWidth(80)
        h.addWidget(self._price)

        self._sma50 = _label(mono=True, size=12)
        self._sma50.setFixedWidth(170)
        h.addWidget(self._sma50)

        self._sma200 = _label(mono=True, size=12)
        self._sma200.setFixedWidth(170)
        h.addWidget(self._sma200)

        h.addStretch()

    def update_data(self, ts: TrendStatus) -> None:
        self._sym.setText(ts.symbol)

        self._price.setText(f"{ts.last_price:,.2f}")

        # 50d SMA
        col_50 = GREEN if ts.price_above_50 else RED
        arrow_50 = _SLOPE_ARROWS.get(ts.sma_50_slope, "")
        self._sma50.setText(
            f"50d {ts.sma_50:,.1f} {_dot(col_50)} "
            f'<span style="color:{col_50};">{arrow_50}</span>'
        )

        # 200d SMA
        col_200 = GREEN if ts.price_above_200 else RED
        arrow_200 = _SLOPE_ARROWS.get(ts.sma_200_slope, "")
        self._sma200.setText(
            f"200d {ts.sma_200:,.1f} {_dot(col_200)} "
            f'<span style="color:{col_200};">{arrow_200}</span>'
        )

    def clear_data(self) -> None:
        for lbl in (self._price, self._sma50, self._sma200):
            lbl.setText('<span style="color:#555;">—</span>')


# ---------------------------------------------------------------------------
# VixRow
# ---------------------------------------------------------------------------

class VixRow(QtWidgets.QWidget):
    """Displays: VIX  level  zone-badge  direction-arrow"""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        h = QtWidgets.QHBoxLayout(self)
        h.setContentsMargins(0, 2, 0, 2)
        h.setSpacing(12)

        sym = _label("VIX", mono=False, size=13)
        sym.setFixedWidth(48)
        sym.setStyleSheet("font-weight: bold;")
        h.addWidget(sym)

        self._level = _label(mono=True, size=13)
        self._level.setFixedWidth(80)
        h.addWidget(self._level)

        self._badge = _label(size=12)
        self._badge.setFixedWidth(100)
        self._badge.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        h.addWidget(self._badge)

        self._direction = _label(size=14)
        self._direction.setFixedWidth(40)
        h.addWidget(self._direction)

        h.addStretch()

    def update_data(self, vs: VixStatus) -> None:
        colour = _ZONE_COLOURS.get(vs.zone, DIM)

        self._level.setText(f"{vs.level:.2f}")
        self._level.setStyleSheet(f"color: {colour};")

        zone_text = vs.zone.upper()
        self._badge.setText(zone_text)
        self._badge.setStyleSheet(
            f"background: {colour}; color: #111; padding: 2px 8px; "
            f"border-radius: 3px; font-weight: bold;"
        )

        arrow = _DIRECTION_ARROWS.get(vs.direction, "")
        self._direction.setText(arrow)
        self._direction.setStyleSheet(f"color: {colour}; font-size: 16px;")

    def clear_data(self) -> None:
        self._level.setText('<span style="color:#555;">—</span>')
        self._badge.setText("")
        self._badge.setStyleSheet("")
        self._direction.setText("")


# ---------------------------------------------------------------------------
# GoNoGoBar
# ---------------------------------------------------------------------------

class GoNoGoBar(QtWidgets.QLabel):
    """Large coloured badge showing GO / CAUTION / NO-GO."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setFixedHeight(56)
        self.setFont(QtGui.QFont("Segoe UI", 20, QtGui.QFont.Weight.Bold))
        self.set_status("NO-GO")

    def set_status(self, status: str) -> None:
        bg, fg = _GONOGO_STYLES.get(status, (DIM, "#fff"))
        self.setText(status)
        self.setStyleSheet(
            f"background: {bg}; color: {fg}; border-radius: 6px; "
            f"padding: 4px 16px;"
        )


# ---------------------------------------------------------------------------
# MacroEventLabel — inline risk text above GO/NO-GO bar
# ---------------------------------------------------------------------------

class MacroEventLabel(QtWidgets.QLabel):
    """Single-line label showing the nearest high-impact macro event risk."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWordWrap(True)
        self.setFont(QtGui.QFont("Segoe UI", 11))
        self.setStyleSheet("color: #aaa;")
        self.hide()

    def set_events(self, event_dicts: list[dict]) -> None:
        """Update from a list of serialised event dicts (from cache or pipeline)."""
        from finance.apps.assistant._calendar import check_macro_risk_from_dicts
        has_risk, risks = check_macro_risk_from_dicts(event_dicts)
        if has_risk and risks:
            # Show only the most urgent risk
            text = risks[0]
            if text.startswith("NO-GO"):
                self.setStyleSheet(f"color: {RED};")
            elif text.startswith("CAUTION"):
                self.setStyleSheet(f"color: {AMBER};")
            else:
                self.setStyleSheet("color: #aaa;")
            self.setText(text)
            self.show()
        else:
            self.hide()


# ---------------------------------------------------------------------------
# EventsCalendarWidget — compact events table
# ---------------------------------------------------------------------------

_MAX_EVENTS_SHOWN: int = 10


class EventsCalendarWidget(QtWidgets.QWidget):
    """Compact scrollable list of upcoming high-impact economic events.

    Each entry shows: [impact colour] title / date-time.
    Events within 48 hours are highlighted.
    NO-GO keywords get a red warning colour.
    Max _MAX_EVENTS_SHOWN rows.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        header = _label("ECONOMIC EVENTS (5d, High+Med)", size=11)
        header.setStyleSheet("color: #888;")
        layout.addWidget(header)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedHeight(160)

        self._container = QtWidgets.QWidget()
        self._container_layout = QtWidgets.QVBoxLayout(self._container)
        self._container_layout.setContentsMargins(0, 0, 0, 0)
        self._container_layout.setSpacing(3)
        self._container_layout.addStretch()

        scroll.setWidget(self._container)
        layout.addWidget(scroll)

        self._placeholder = _label("No events loaded", size=11)
        self._placeholder.setStyleSheet("color: #555;")
        self._container_layout.insertWidget(0, self._placeholder)

    def set_events(self, event_dicts: list[dict]) -> None:
        """Rebuild the event list from serialised event dicts."""
        from datetime import datetime, timezone

        # Clear existing rows (keep stretch at end)
        while self._container_layout.count() > 1:
            item = self._container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        now = datetime.now(timezone.utc)
        shown = 0

        for d in event_dicts[:_MAX_EVENTS_SHOWN]:
            try:
                dt = datetime.fromisoformat(d["date"])
            except (ValueError, KeyError):
                continue

            hours_away = (dt - now).total_seconds() / 3600
            if hours_away < 0:
                continue  # already passed
            title = d.get("title", "")
            country = d.get("country", "")

            from finance.apps.assistant._calendar import NO_GO_KEYWORDS
            is_nogo = any(kw.lower() in title.lower() for kw in NO_GO_KEYWORDS)
            is_imminent = hours_away <= 48

            if is_nogo and is_imminent:
                colour = RED
            elif is_imminent:
                colour = AMBER
            else:
                colour = "#888"

            date_str = dt.strftime("%a %b %d %H:%M")
            row_lbl = _label(f"{title}\n{country}  {date_str}", size=11)
            row_lbl.setWordWrap(True)
            row_lbl.setStyleSheet(f"color: {colour};")
            self._container_layout.insertWidget(shown, row_lbl)
            shown += 1

        if shown == 0:
            placeholder = _label("No upcoming high-impact events", size=11)
            placeholder.setStyleSheet("color: #555;")
            self._container_layout.insertWidget(0, placeholder)


# ---------------------------------------------------------------------------
# DriftSection — collapsible DRIFT regime + underlying eligibility block
# ---------------------------------------------------------------------------

_DRIFT_TIER_COLOURS = {
    "Normal":          GREEN,
    "Elevated":        AMBER,
    "Correction":      "#FF7043",
    "Deep Correction": RED,
    "Bear":            RED,
}

_BP_WARNING_THRESHOLD: float = 50.0


class _UnderlyingRow(QtWidgets.QWidget):
    """One row: symbol  IVP%  ●eligible  structure-text"""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        h = QtWidgets.QHBoxLayout(self)
        h.setContentsMargins(0, 1, 0, 1)
        h.setSpacing(8)

        self._sym = _label(mono=False, size=12)
        self._sym.setFixedWidth(36)
        self._sym.setStyleSheet("font-weight: bold;")
        h.addWidget(self._sym)

        self._ivp = _label(mono=True, size=11)
        self._ivp.setFixedWidth(60)
        h.addWidget(self._ivp)

        self._dot_lbl = _label(size=14)
        self._dot_lbl.setFixedWidth(20)
        h.addWidget(self._dot_lbl)

        self._structure = _label(size=11)
        h.addWidget(self._structure)
        h.addStretch()

    def update_data(self, status: DriftUnderlyingStatus) -> None:
        self._sym.setText(status.symbol)

        if status.ivp is not None:
            self._ivp.setText(f"IVP {status.ivp:.0f}")
            ivp_colour = GREEN if status.ivp >= 50 else AMBER
            self._ivp.setStyleSheet(f"color: {ivp_colour};")
        else:
            self._ivp.setText("IVP —")
            self._ivp.setStyleSheet(f"color: {DIM};")

        eligible_colour = GREEN if status.eligible else DIM
        self._dot_lbl.setText(_dot(eligible_colour))

        self._structure.setText(status.structure)
        self._structure.setStyleSheet(f"color: {'#ccc' if status.eligible else DIM};")


class DriftSection(QtWidgets.QWidget):
    """Collapsible DRIFT regime block.

    Header row: ▶/▼ toggle + «DRIFT» label + tier badge.
    Collapsed by default; click toggle to expand.

    Expanded content:
      - Tier info row: BP% + structure + DTE range
      - One _UnderlyingRow per underlying
      - BP guardrail: labelled spinbox + warning when > 50%
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ---- header row ----
        header_row = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(header_row)
        h.setContentsMargins(0, 4, 0, 4)
        h.setSpacing(6)

        self._toggle_btn = QtWidgets.QPushButton("\u25b6 DRIFT")  # ▶
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setFlat(True)
        self._toggle_btn.setFont(QtGui.QFont("Segoe UI", 11, QtGui.QFont.Weight.Bold))
        self._toggle_btn.setStyleSheet("color: #888; text-align: left; padding: 0;")
        self._toggle_btn.toggled.connect(self._on_toggle)
        h.addWidget(self._toggle_btn)

        self._tier_label = _label("", size=11)
        self._tier_label.setStyleSheet(
            f"color: #111; background: {DIM}; border-radius: 3px; padding: 1px 6px; font-weight: bold;"
        )
        h.addWidget(self._tier_label)
        h.addStretch()
        outer.addWidget(header_row)

        # ---- collapsible content ----
        self._content = QtWidgets.QWidget()
        self._content.hide()
        inner = QtWidgets.QVBoxLayout(self._content)
        inner.setContentsMargins(8, 0, 0, 4)
        inner.setSpacing(3)

        # Tier info row
        self._tier_info = _label("", size=11)
        self._tier_info.setWordWrap(True)
        self._tier_info.setStyleSheet("color: #aaa;")
        inner.addWidget(self._tier_info)

        inner.addWidget(_hsep())

        # Underlying rows container
        self._underlying_container = QtWidgets.QWidget()
        self._underlying_layout = QtWidgets.QVBoxLayout(self._underlying_container)
        self._underlying_layout.setContentsMargins(0, 0, 0, 0)
        self._underlying_layout.setSpacing(1)
        inner.addWidget(self._underlying_container)

        inner.addWidget(_hsep())

        # BP guardrail
        bp_row = QtWidgets.QWidget()
        bp_h = QtWidgets.QHBoxLayout(bp_row)
        bp_h.setContentsMargins(0, 2, 0, 2)
        bp_h.setSpacing(6)

        bp_lbl = _label("BP used %", size=11)
        bp_lbl.setStyleSheet("color: #888;")
        bp_h.addWidget(bp_lbl)

        self._bp_spinbox = QtWidgets.QDoubleSpinBox()
        self._bp_spinbox.setRange(0.0, 100.0)
        self._bp_spinbox.setSingleStep(5.0)
        self._bp_spinbox.setDecimals(0)
        self._bp_spinbox.setValue(0.0)
        self._bp_spinbox.setFixedWidth(70)
        self._bp_spinbox.setStyleSheet(
            "background: #222; color: #ccc; border: 1px solid #444;"
        )
        self._bp_spinbox.valueChanged.connect(self._on_bp_changed)
        bp_h.addWidget(self._bp_spinbox)
        bp_h.addStretch()
        inner.addWidget(bp_row)

        self._bp_warning = _label("\u26a0 BP > 50% — reduce exposure", size=11)
        self._bp_warning.setStyleSheet(f"color: {RED};")
        self._bp_warning.hide()
        inner.addWidget(self._bp_warning)

        outer.addWidget(self._content)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_drift(
        self,
        tier: DriftTier,
        underlyings: list[DriftUnderlyingStatus],
    ) -> None:
        """Refresh the section with new tier and underlying data."""
        tier_colour = _DRIFT_TIER_COLOURS.get(tier.name, DIM)
        self._tier_label.setText(tier.name)
        self._tier_label.setStyleSheet(
            f"color: #111; background: {tier_colour}; "
            f"border-radius: 3px; padding: 1px 6px; font-weight: bold;"
        )
        self._toggle_btn.setStyleSheet(
            f"color: {tier_colour}; text-align: left; padding: 0;"
        )

        self._tier_info.setText(
            f"DD {tier.drawdown_pct:.1f}%  ·  BP rec {tier.bp_pct}%  ·  "
            f"{tier.structure}  ·  DTE {tier.dte_range}"
        )

        # Rebuild underlying rows
        while self._underlying_layout.count():
            item = self._underlying_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for status in underlyings:
            row = _UnderlyingRow()
            row.update_data(status)
            self._underlying_layout.addWidget(row)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_toggle(self, checked: bool) -> None:
        arrow = "\u25bc" if checked else "\u25b6"  # ▼ or ▶
        # Preserve current colour in toggle button text
        current_style = self._toggle_btn.styleSheet()
        colour_match = current_style.split("color:")[1].split(";")[0].strip() if "color:" in current_style else "#888"
        self._toggle_btn.setText(f"{arrow} DRIFT")
        if checked:
            self._content.show()
        else:
            self._content.hide()

    def _on_bp_changed(self, value: float) -> None:
        if value > _BP_WARNING_THRESHOLD:
            self._bp_warning.show()
        else:
            self._bp_warning.hide()


# ---------------------------------------------------------------------------
# SwingRegimePanel — assembles everything
# ---------------------------------------------------------------------------

class SwingRegimePanel(QtWidgets.QWidget):
    """Left panel: SPY/QQQ trend indicators, VIX, GO/NO-GO banner,
    macro event warning, events calendar, and Claude market summary.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        header = QtWidgets.QLabel("SWING REGIME")
        header.setFont(QtGui.QFont("Segoe UI", 14, QtGui.QFont.Weight.Bold))
        header.setStyleSheet("color: #ccc;")
        layout.addWidget(header)

        self.spy_row = TrendRow()
        self.qqq_row = TrendRow()
        layout.addWidget(self.spy_row)
        layout.addWidget(self.qqq_row)

        layout.addWidget(_hsep())

        self.vix_row = VixRow()
        layout.addWidget(self.vix_row)

        layout.addWidget(_hsep())

        self.go_nogo = GoNoGoBar()
        layout.addWidget(self.go_nogo)

        layout.addWidget(_hsep())

        self.macro_label = MacroEventLabel()
        layout.addWidget(self.macro_label)

        layout.addWidget(_hsep())

        self.events_widget = EventsCalendarWidget()
        layout.addWidget(self.events_widget)

        layout.addWidget(_hsep())

        self.summary_widget = MarketSummaryWidget()
        layout.addWidget(self.summary_widget)

        layout.addWidget(_hsep())

        self.drift_section = DriftSection()
        layout.addWidget(self.drift_section)

        layout.addStretch()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_indicators(
        self,
        spy: TrendStatus | None,
        qqq: TrendStatus | None,
        vix: VixStatus | None,
        status: str,
    ) -> None:
        if spy:
            self.spy_row.update_data(spy)
        else:
            self.spy_row.clear_data()

        if qqq:
            self.qqq_row.update_data(qqq)
        else:
            self.qqq_row.clear_data()

        if vix:
            self.vix_row.update_data(vix)
        else:
            self.vix_row.clear_data()

        self.go_nogo.set_status(status)

    def update_events(self, event_dicts: list[dict]) -> None:
        """Update macro warning label and events calendar from cache/pipeline data."""
        self.macro_label.set_events(event_dicts)
        self.events_widget.set_events(event_dicts)

    def update_market_summary(self, summary: dict | None) -> None:
        """Push a new market summary dict (or None to clear) to the summary widget."""
        self.summary_widget.set_summary(summary)

    def update_drift(
        self,
        tier: DriftTier,
        underlyings: list[DriftUnderlyingStatus],
    ) -> None:
        """Push updated DRIFT tier and underlying eligibility to the section."""
        self.drift_section.update_drift(tier, underlyings)


# ---------------------------------------------------------------------------
# MarketSummaryWidget — Claude-generated regime brief
# ---------------------------------------------------------------------------


class MarketSummaryWidget(QtWidgets.QWidget):
    """
    Displays the Claude-generated market summary in three states:

    - *empty*: placeholder text ("No summary yet")
    - *generating*: spinner-style label ("Generating summary…")
    - *ready*: regime badge + scrollable text brief
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        header = _label("MARKET BRIEF", size=11)
        header.setStyleSheet("color: #888;")
        layout.addWidget(header)

        self._regime_badge = QtWidgets.QLabel()
        self._regime_badge.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self._regime_badge.setFont(QtGui.QFont("Segoe UI", 12, QtGui.QFont.Weight.Bold))
        self._regime_badge.hide()
        layout.addWidget(self._regime_badge)

        self._text = QtWidgets.QTextBrowser()
        self._text.setReadOnly(True)
        self._text.setOpenExternalLinks(False)
        self._text.setFixedHeight(180)
        self._text.setStyleSheet(
            "background: #1a1a1a; border: 1px solid #333; "
            "color: #ccc; font-size: 11px; padding: 4px;"
        )
        layout.addWidget(self._text)

        self._set_placeholder("No summary yet")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_generating(self) -> None:
        """Show a 'Generating…' state while Claude is running."""
        self._regime_badge.hide()
        self._text.setPlainText("Generating market summary…")
        self._text.setStyleSheet(
            "background: #1a1a1a; border: 1px solid #333; "
            "color: #888; font-size: 11px; padding: 4px;"
        )

    def set_error(self, message: str) -> None:
        """Show an error state (red text) when the summary cannot be generated."""
        self._regime_badge.hide()
        self._text.setPlainText(message)
        self._text.setStyleSheet(
            "background: #1a1a1a; border: 1px solid #333; "
            "color: #F44336; font-size: 11px; padding: 4px;"
        )

    def set_summary(self, summary: dict | None) -> None:
        """
        Display the Claude market summary dict.

        Passing ``None`` resets to the empty placeholder state.
        """
        if not summary:
            self._set_placeholder("No summary yet")
            return

        regime = summary.get("regime", "")
        if regime:
            bg, fg = _GONOGO_STYLES.get(regime, (DIM, "#fff"))
            self._regime_badge.setText(regime)
            self._regime_badge.setStyleSheet(
                f"background: {bg}; color: {fg}; border-radius: 4px; "
                f"padding: 2px 10px;"
            )
            self._regime_badge.show()
        else:
            self._regime_badge.hide()

        lines: list[str] = []
        reasoning = summary.get("regime_reasoning", "")
        if reasoning:
            lines.append(reasoning)

        for section, key in [
            ("Themes", "themes"),
            ("Movers", "movers"),
            ("Risks", "risks"),
            ("Actions", "action_items"),
        ]:
            items = summary.get(key) or []
            if items:
                lines.append(f"\n{section}:")
                lines.extend(f"  • {item}" for item in items)

        self._text.setPlainText("\n".join(lines))
        self._text.setStyleSheet(
            "background: #1a1a1a; border: 1px solid #333; "
            "color: #ccc; font-size: 11px; padding: 4px;"
        )

    # ------------------------------------------------------------------

    def _set_placeholder(self, text: str) -> None:
        self._regime_badge.hide()
        self._text.setPlainText(text)
        self._text.setStyleSheet(
            "background: #1a1a1a; border: 1px solid #333; "
            "color: #555; font-size: 11px; padding: 4px;"
        )
