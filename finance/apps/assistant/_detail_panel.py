"""
finance.apps.assistant._detail_panel
======================================
Right panel — candidate detail view.

Displays all scored fields for the selected watchlist row:
  - Header: symbol, direction, score, tags
  - Dimension breakdown: D1–D5 bars with hard-gate / partial indicators,
    expandable sub-component rows
  - Price & momentum: price, 5D%, 1M%, RVOL, ATR%
  - Catalyst context: IV%, P/C ratio, earnings surprise
  - Metadata: sector, market cap, earnings date, tags
  - AI reasoning: Claude-generated setup analysis (placeholder until TA-E5-S2)
"""
from __future__ import annotations

from dataclasses import dataclass

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

# ---------------------------------------------------------------------------
# Colours (shared with swing panel)
# ---------------------------------------------------------------------------

GREEN = "#4CAF50"
AMBER = "#FFA726"
RED   = "#F44336"
DIM   = "#555"

_SCORE_GREEN = 70.0
_SCORE_AMBER = 40.0

_DIM_LABELS = {
    1: ("D1", "Trend Template"),
    2: ("D2", "Relative Strength"),
    3: ("D3", "Base Quality"),
    4: ("D4", "Catalyst"),
    5: ("D5", "Risk"),
}

# Maximum possible weighted_score per dimension (= dimension weight).
_DIM_MAX = {1: 25, 2: 25, 3: 15, 4: 20, 5: 15}

_REASONING_PLACEHOLDER = "No reasoning available — click Analyze to generate"


def _score_colour(score: float, max_score: float = 100.0) -> str:
    """Colour based on score as a fraction of max_score."""
    pct = score / max_score * 100.0 if max_score else 0.0
    if pct >= _SCORE_GREEN:
        return GREEN
    if pct >= _SCORE_AMBER:
        return AMBER
    return RED


def _label(text: str = "", size: int = 12, bold: bool = False) -> QtWidgets.QLabel:
    lbl = QtWidgets.QLabel(text)
    weight = QtGui.QFont.Weight.Bold if bold else QtGui.QFont.Weight.Normal
    lbl.setFont(QtGui.QFont("Segoe UI", size, weight))
    lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)
    return lbl


def _mono(text: str = "", size: int = 12) -> QtWidgets.QLabel:
    lbl = QtWidgets.QLabel(text)
    lbl.setFont(QtGui.QFont("Roboto Mono", size))
    return lbl


def _hsep() -> QtWidgets.QFrame:
    line = QtWidgets.QFrame()
    line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
    line.setStyleSheet("color: #333;")
    return line


def _section_header(text: str) -> QtWidgets.QLabel:
    lbl = _label(text, size=10, bold=False)
    lbl.setStyleSheet("color: #888; letter-spacing: 1px;")
    return lbl


def _kv_row(layout: QtWidgets.QFormLayout, key: str, value_widget: QtWidgets.QWidget) -> None:
    key_lbl = _label(key, size=11)
    key_lbl.setStyleSheet("color: #777;")
    layout.addRow(key_lbl, value_widget)


# ---------------------------------------------------------------------------
# _DimRow — all widgets for one scoring dimension
# ---------------------------------------------------------------------------


@dataclass
class _DimRow:
    """Holds every widget associated with a single scoring dimension row."""
    score_label: QtWidgets.QLabel
    bar: QtWidgets.QProgressBar
    gate_label: QtWidgets.QLabel       # ⚠ — shown when hard_gate_fired
    partial_label: QtWidgets.QLabel    # ~  — shown when partial data
    expand_btn: QtWidgets.QToolButton  # ▶/▼ toggle for sub-components
    components_widget: QtWidgets.QWidget   # hidden container for sub-rows
    components_layout: QtWidgets.QVBoxLayout


# ---------------------------------------------------------------------------
# CandidateDetailPanel
# ---------------------------------------------------------------------------

class CandidateDetailPanel(QtWidgets.QWidget):
    """Right panel — shows full detail for the selected watchlist row."""

    analyze_requested = QtCore.Signal(dict)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._current_row: dict | None = None
        self._build_ui()
        self._show_empty()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_row(self, row: dict | None) -> None:
        """Populate the panel from *row* dict, or clear if None."""
        self._current_row = row
        self.load_reasoning(None)
        self.set_analyzing(False)
        if row is None:
            self._show_empty()
            return
        self._populate(row)

    def set_analyzing(self, analyzing: bool) -> None:
        """Enable or disable the Analyze button to reflect background work."""
        self._analyze_btn.setEnabled(not analyzing)
        self._analyze_btn.setText("Analyzing…" if analyzing else "Analyze")

    def load_reasoning(self, data: dict | None) -> None:
        """
        Display Claude-generated reasoning in the AI reasoning section.

        Parameters
        ----------
        data:
            Dict with keys: setup_type, profit_mechanism, thesis,
            entry, stop, target, confidence.
            Pass ``None`` to reset to the placeholder state.
        """
        if not data:
            self._reasoning_text.setPlainText(_REASONING_PLACEHOLDER)
            self._reasoning_text.setStyleSheet(
                "background: #1a1a1a; border: 1px solid #333; "
                "color: #555; font-size: 11px; padding: 4px;"
            )
            return

        lines = []
        if data.get("setup_type"):
            lines.append(f"Setup:  {data['setup_type']}")
        if data.get("profit_mechanism"):
            lines.append(f"PM:     {data['profit_mechanism']}")
        if data.get("thesis"):
            lines.append(f"\n{data['thesis']}")
        entry = data.get("entry")
        stop = data.get("stop")
        target = data.get("target")
        if entry is not None:
            lines.append(f"\nEntry:  {entry:.2f}")
        if stop is not None:
            lines.append(f"Stop:   {stop:.2f}")
        if target is not None:
            lines.append(f"Target: {target:.2f}")
        if entry and stop and target:
            risk = abs(entry - stop)
            reward = abs(target - entry)
            rr = reward / risk if risk else 0.0
            lines.append(f"R:R     {rr:.1f}:1")
        if data.get("confidence"):
            lines.append(f"\nConfidence: {data['confidence']}")

        self._reasoning_text.setPlainText("\n".join(lines))
        self._reasoning_text.setStyleSheet(
            "background: #1a1a1a; border: 1px solid #333; "
            "color: #ccc; font-size: 11px; padding: 4px;"
        )

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(0)

        # Scrollable content
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QtWidgets.QWidget()
        self._layout = QtWidgets.QVBoxLayout(content)
        self._layout.setContentsMargins(0, 0, 8, 0)
        self._layout.setSpacing(6)

        # --- Header ---
        self._lbl_symbol = _label("", size=20, bold=True)
        self._lbl_direction = _label("", size=12)
        self._lbl_direction.setStyleSheet("color: #aaa;")
        self._lbl_score = _label("", size=22, bold=True)

        header_row = QtWidgets.QHBoxLayout()
        header_row.addWidget(self._lbl_symbol)
        header_row.addWidget(self._lbl_direction)
        header_row.addStretch()
        header_row.addWidget(self._lbl_score)
        self._layout.addLayout(header_row)

        self._lbl_tags = _label("", size=11)
        self._lbl_tags.setWordWrap(True)
        self._lbl_tags.setStyleSheet("color: #888;")
        self._layout.addWidget(self._lbl_tags)

        self._layout.addWidget(_hsep())

        # --- Dimensions ---
        self._layout.addWidget(_section_header("SCORING DIMENSIONS"))
        self._dim_rows: dict[int, _DimRow] = {}
        for dim_num, (short, long_name) in _DIM_LABELS.items():
            self._dim_rows[dim_num] = self._build_dim_row(dim_num, short, long_name)

        self._layout.addWidget(_hsep())

        # --- Price & momentum ---
        self._layout.addWidget(_section_header("PRICE & MOMENTUM"))
        form1 = QtWidgets.QFormLayout()
        form1.setSpacing(4)
        form1.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)

        self._lbl_price     = _mono(size=11)
        self._lbl_chg5d     = _mono(size=11)
        self._lbl_chg1m     = _mono(size=11)
        self._lbl_rvol      = _mono(size=11)
        self._lbl_atr       = _mono(size=11)

        _kv_row(form1, "Price",    self._lbl_price)
        _kv_row(form1, "5D %",     self._lbl_chg5d)
        _kv_row(form1, "1M %",     self._lbl_chg1m)
        _kv_row(form1, "RVOL",     self._lbl_rvol)
        _kv_row(form1, "ATR %",    self._lbl_atr)
        self._layout.addLayout(form1)

        self._layout.addWidget(_hsep())

        # --- Catalyst ---
        self._layout.addWidget(_section_header("CATALYST"))
        form2 = QtWidgets.QFormLayout()
        form2.setSpacing(4)
        form2.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)

        self._lbl_iv        = _mono(size=11)
        self._lbl_pc        = _mono(size=11)
        self._lbl_surprise  = _mono(size=11)
        self._lbl_short     = _mono(size=11)
        self._lbl_earnings  = _mono(size=11)

        _kv_row(form2, "IV Pctl",  self._lbl_iv)
        _kv_row(form2, "P/C Vol",  self._lbl_pc)
        _kv_row(form2, "EPS Surp", self._lbl_surprise)
        _kv_row(form2, "Short %",  self._lbl_short)
        _kv_row(form2, "Earnings", self._lbl_earnings)
        self._layout.addLayout(form2)

        self._layout.addWidget(_hsep())

        # --- Metadata ---
        self._layout.addWidget(_section_header("METADATA"))
        form3 = QtWidgets.QFormLayout()
        form3.setSpacing(4)
        form3.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)

        self._lbl_sector    = _mono(size=11)
        self._lbl_mktcap    = _mono(size=11)

        _kv_row(form3, "Sector",   self._lbl_sector)
        _kv_row(form3, "Mkt Cap",  self._lbl_mktcap)
        self._layout.addLayout(form3)

        self._layout.addWidget(_hsep())

        # --- AI Reasoning ---
        reasoning_header_row = QtWidgets.QHBoxLayout()
        reasoning_header_row.addWidget(_section_header("AI REASONING"))
        reasoning_header_row.addStretch()

        self._analyze_btn = QtWidgets.QPushButton("Analyze")
        self._analyze_btn.setFixedHeight(22)
        self._analyze_btn.setStyleSheet(
            "QPushButton { background: #2a3a4a; color: #aaa; border: 1px solid #444;"
            " border-radius: 3px; font-size: 11px; padding: 0 8px; }"
            "QPushButton:hover { background: #3a4a5a; color: #ccc; }"
            "QPushButton:disabled { color: #555; }"
        )
        self._analyze_btn.clicked.connect(self._on_analyze_clicked)
        reasoning_header_row.addWidget(self._analyze_btn)
        self._layout.addLayout(reasoning_header_row)

        self._reasoning_text = QtWidgets.QTextBrowser()
        self._reasoning_text.setReadOnly(True)
        self._reasoning_text.setOpenExternalLinks(False)
        self._reasoning_text.setFixedHeight(120)
        self._layout.addWidget(self._reasoning_text)

        self._layout.addStretch()

        scroll.setWidget(content)
        outer.addWidget(scroll)

    def _build_dim_row(self, dim_num: int, short: str, long_name: str) -> _DimRow:
        """
        Build the widgets for one scoring dimension and add them to the layout.

        Each dimension occupies two layout rows:
          1. The main row: expand button, name+max, progress bar, score, gate/partial
          2. A hidden components container (shown on expand)
        """
        dim_max = _DIM_MAX[dim_num]

        # --- main row ---
        row_w = QtWidgets.QWidget()
        row_h = QtWidgets.QHBoxLayout(row_w)
        row_h.setContentsMargins(0, 1, 0, 1)
        row_h.setSpacing(4)

        expand_btn = QtWidgets.QToolButton()
        expand_btn.setText("▶")
        expand_btn.setFixedSize(18, 18)
        expand_btn.setStyleSheet(
            "QToolButton { color: #555; background: transparent; border: none; font-size: 9px; }"
            "QToolButton:hover { color: #aaa; }"
        )
        expand_btn.hide()
        row_h.addWidget(expand_btn)

        lbl_name = _label(f"{short}  {long_name}  /{dim_max}", size=11)
        lbl_name.setStyleSheet("color: #bbb;")
        lbl_name.setFixedWidth(192)
        row_h.addWidget(lbl_name)

        bar = QtWidgets.QProgressBar()
        bar.setRange(0, dim_max * 10)
        bar.setValue(0)
        bar.setTextVisible(False)
        bar.setFixedHeight(6)
        bar.setStyleSheet(
            "QProgressBar { background: #222; border-radius: 3px; }"
            "QProgressBar::chunk { background: #4CAF50; border-radius: 3px; }"
        )
        row_h.addWidget(bar, 1)

        lbl_score = _mono("—", size=11)
        lbl_score.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        lbl_score.setFixedWidth(40)
        row_h.addWidget(lbl_score)

        gate_label = _label("⚠", size=11)
        gate_label.setStyleSheet(f"color: {RED}; font-weight: bold;")
        gate_label.setToolTip("Hard gate fired — dimension score set to 0")
        gate_label.hide()
        row_h.addWidget(gate_label)

        partial_label = _label("~", size=11)
        partial_label.setStyleSheet("color: #888;")
        partial_label.setToolTip("Partial data — some components unavailable")
        partial_label.hide()
        row_h.addWidget(partial_label)

        self._layout.addWidget(row_w)

        # --- components container (hidden) ---
        comp_widget = QtWidgets.QWidget()
        comp_layout = QtWidgets.QVBoxLayout(comp_widget)
        comp_layout.setContentsMargins(22, 0, 0, 2)
        comp_layout.setSpacing(1)
        comp_widget.hide()
        self._layout.addWidget(comp_widget)

        dim_row = _DimRow(
            score_label=lbl_score,
            bar=bar,
            gate_label=gate_label,
            partial_label=partial_label,
            expand_btn=expand_btn,
            components_widget=comp_widget,
            components_layout=comp_layout,
        )

        expand_btn.clicked.connect(lambda _checked, dr=dim_row: self._toggle_components(dr))
        return dim_row

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _on_analyze_clicked(self) -> None:
        if self._current_row is not None:
            self.analyze_requested.emit(self._current_row)

    @staticmethod
    def _toggle_components(dim_row: _DimRow) -> None:
        show = dim_row.components_widget.isHidden()  # currently hidden → show it
        dim_row.components_widget.setVisible(show)
        dim_row.expand_btn.setText("▼" if show else "▶")

    def _show_empty(self) -> None:
        self._lbl_symbol.setText('<span style="color:#444;">Select a row</span>')
        self._lbl_direction.setText("")
        self._lbl_score.setText("")
        self._lbl_tags.setText("")
        for dr in self._dim_rows.values():
            dr.score_label.setText("—")
            dr.bar.setValue(0)
            dr.gate_label.hide()
            dr.partial_label.hide()
            dr.expand_btn.hide()
            dr.components_widget.hide()
        for w in (self._lbl_price, self._lbl_chg5d, self._lbl_chg1m,
                  self._lbl_rvol, self._lbl_atr, self._lbl_iv, self._lbl_pc,
                  self._lbl_surprise, self._lbl_short, self._lbl_earnings,
                  self._lbl_sector, self._lbl_mktcap):
            w.setText("—")

    def _populate(self, row: dict) -> None:
        """Fill all sections from the result row dict produced by build_result_row()."""
        symbol = row.get("symbol") or ""
        direction = row.get("direction") or ""
        score = row.get("score_total")
        tags = row.get("tags") or []

        # Header
        self._lbl_symbol.setText(f'<b>{symbol}</b>')
        dir_text = "LONG" if direction == "long" else "SHORT"
        dir_colour = GREEN if direction == "long" else RED
        self._lbl_direction.setText(f'<span style="color:{dir_colour};">{dir_text}</span>')

        if score is not None:
            colour = _score_colour(float(score))
            self._lbl_score.setText(f'<span style="color:{colour};">{score:.1f}</span>')
        else:
            self._lbl_score.setText("—")

        self._lbl_tags.setText("  ".join(f"[{t}]" for t in tags) if tags else "no tags")

        # Dimensions
        dim_dicts = {d["dimension"]: d for d in (row.get("dimensions") or [])}
        for dim_num, dr in self._dim_rows.items():
            d = dim_dicts.get(dim_num)
            if d is None:
                dr.score_label.setText("—")
                dr.bar.setValue(0)
                dr.gate_label.hide()
                dr.partial_label.hide()
                dr.expand_btn.hide()
                dr.components_widget.hide()
                continue

            ws = d.get("weighted_score")
            dim_max = _DIM_MAX[dim_num]
            if ws is not None:
                colour = _score_colour(float(ws), max_score=dim_max)
                dr.score_label.setText(f'<span style="color:{colour};">{ws:.1f}</span>')
                dr.bar.setValue(int(float(ws) * 10))
                dr.bar.setStyleSheet(
                    "QProgressBar { background: #222; border-radius: 3px; }"
                    f"QProgressBar::chunk {{ background: {colour}; border-radius: 3px; }}"
                )
            else:
                dr.score_label.setText("—")
                dr.bar.setValue(0)

            # Hard gate and partial indicators
            if d.get("hard_gate_fired"):
                dr.gate_label.show()
            else:
                dr.gate_label.hide()

            if d.get("partial"):
                dr.partial_label.show()
            else:
                dr.partial_label.hide()

            # Sub-components
            components = d.get("components") or []
            self._rebuild_components(dr, components)

        # Price & momentum
        def _fmt(v, fmt: str, suffix: str = "") -> str:
            return f"{v:{fmt}}{suffix}" if v is not None else "—"

        self._lbl_price.setText(_fmt(row.get("price"), ".2f"))
        self._lbl_chg5d.setText(_fmt(row.get("change_5d_pct"), "+.1f", "%"))
        self._lbl_chg1m.setText(_fmt(row.get("change_1m_pct"), "+.1f", "%"))
        self._lbl_rvol.setText(_fmt(row.get("rvol_20d"), ".1f", "x"))
        self._lbl_atr.setText(_fmt(row.get("atr_pct_20d"), ".1f", "%"))

        # Catalyst
        self._lbl_iv.setText(_fmt(row.get("iv_percentile"), ".0f", "%ile"))
        self._lbl_pc.setText(_fmt(row.get("put_call_vol_5d"), ".2f"))
        self._lbl_surprise.setText(_fmt(row.get("earnings_surprise_pct"), "+.1f", "%"))
        self._lbl_short.setText(_fmt(row.get("short_float"), ".1f", "%"))
        self._lbl_earnings.setText(str(row.get("latest_earnings") or "—"))

        # Metadata
        self._lbl_sector.setText(str(row.get("sector") or "—"))
        mktcap_k = row.get("market_cap_k")
        if mktcap_k is not None:
            b = mktcap_k / 1_000_000
            self._lbl_mktcap.setText(f"${b:.1f}B")
        else:
            self._lbl_mktcap.setText("—")

    def _rebuild_components(self, dr: _DimRow, components: list[dict]) -> None:
        """Clear and repopulate the sub-component rows for a dimension."""
        # Remove old rows
        while dr.components_layout.count():
            item = dr.components_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        dr.components_widget.hide()
        dr.expand_btn.setText("▶")
        dr.expand_btn.hide()

        if not components:
            return

        for comp in components:
            name = comp.get("name", "")
            raw = comp.get("raw_score")
            available = comp.get("available", True)

            row_w = QtWidgets.QWidget()
            row_h = QtWidgets.QHBoxLayout(row_w)
            row_h.setContentsMargins(0, 0, 0, 0)
            row_h.setSpacing(6)

            name_lbl = _label(name, size=10)
            if not available:
                name_lbl.setStyleSheet("color: #444;")
            else:
                name_lbl.setStyleSheet("color: #777;")
            row_h.addWidget(name_lbl, 1)

            if not available:
                val_lbl = _mono("n/a", size=10)
                val_lbl.setStyleSheet("color: #444;")
            elif raw is not None:
                pct = int(raw * 100)
                colour = _score_colour(float(pct))
                val_lbl = _mono(f"{pct}%", size=10)
                val_lbl.setStyleSheet(f"color: {colour};")
            else:
                val_lbl = _mono("—", size=10)
                val_lbl.setStyleSheet("color: #555;")

            val_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            val_lbl.setFixedWidth(36)
            row_h.addWidget(val_lbl)

            dr.components_layout.addWidget(row_w)

        dr.expand_btn.show()
