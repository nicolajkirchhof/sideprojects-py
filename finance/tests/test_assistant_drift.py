"""
Tests for TA-E3-S5 — DRIFT regime and eligibility.

Written test-first (TDD).

Covers:
  - compute_drift_tier: tier selection from SPY drawdown × VIX
  - compute_drift_underlying: IVP, price vs 200d SMA, structure recommendation
  - DriftSection widget (Qt display required)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

_has_display = (
    sys.platform == "win32"
    or os.environ.get("DISPLAY")
    or os.environ.get("WAYLAND_DISPLAY")
)


# ---------------------------------------------------------------------------
# Helpers — build synthetic OHLCV + IV/HV DataFrames
# ---------------------------------------------------------------------------

def _make_spy_df(
    n: int = 300,
    current_price: float = 500.0,
    peak_price: float | None = None,
) -> pd.DataFrame:
    """Build a minimal daily DataFrame for SPY with constant close and IV."""
    if peak_price is None:
        peak_price = current_price
    closes = [peak_price] * (n - 1) + [current_price]
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    iv_vals = [0.20] * n
    hv_vals = [0.18] * n
    return pd.DataFrame({"c": closes, "iv": iv_vals, "hv": hv_vals}, index=idx)


def _make_underlying_df(
    n: int = 300,
    price: float = 100.0,
    sma200_price: float | None = None,
    iv: float = 0.25,
    hv: float = 0.20,
    ivp_history_low: bool = False,
) -> pd.DataFrame:
    """
    Build a minimal daily DataFrame for a DRIFT underlying.

    ivp_history_low=True: fill the last 252 bars with iv*2 so today's iv is low.
    """
    if sma200_price is None:
        sma200_price = price
    closes = [sma200_price] * (n - 1) + [price]
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    # Set historical IV high so current IV ranks low
    if ivp_history_low:
        iv_series = [iv * 2.0] * (n - 1) + [iv]
    else:
        iv_series = [iv * 0.5] * (n - 1) + [iv]
    hv_series = [hv] * n
    # Intentionally leave last HV as NaN to test robustness
    hv_series[-1] = float("nan")
    return pd.DataFrame({"c": closes, "iv": iv_series, "hv": hv_series}, index=idx)


# ---------------------------------------------------------------------------
# compute_drift_tier — tier selection
# ---------------------------------------------------------------------------


def test_drift_tier_normal_when_shallow_drawdown_and_low_vix():
    """0–5% drawdown + VIX < 20 → Normal tier."""
    from finance.apps.assistant._data import compute_drift_tier
    from finance.apps.assistant._data import VixStatus

    df = _make_spy_df(current_price=497.0, peak_price=500.0)  # -0.6% drawdown
    vix = VixStatus(level=16.0, zone="low", direction="falling", is_spiking=False)
    tier = compute_drift_tier(df, vix)

    assert tier.name == "Normal"
    assert tier.bp_pct == 30
    assert tier.drawdown_pct < 5.0


def test_drift_tier_elevated_when_both_conditions_met():
    """-7% drawdown + VIX 25 → Elevated tier."""
    from finance.apps.assistant._data import compute_drift_tier
    from finance.apps.assistant._data import VixStatus

    df = _make_spy_df(current_price=465.0, peak_price=500.0)  # -7% drawdown
    vix = VixStatus(level=25.0, zone="elevated", direction="rising", is_spiking=False)
    tier = compute_drift_tier(df, vix)

    assert tier.name == "Elevated"
    assert tier.bp_pct == 40


def test_drift_tier_is_conservative_when_only_drawdown_elevated():
    """
    -7% drawdown (Elevated) but VIX 16 (Normal) → stays at Normal.
    Both conditions must be met to advance — conservative-min rule.
    """
    from finance.apps.assistant._data import compute_drift_tier
    from finance.apps.assistant._data import VixStatus

    df = _make_spy_df(current_price=465.0, peak_price=500.0)  # -7% drawdown
    vix = VixStatus(level=16.0, zone="low", direction="falling", is_spiking=False)
    tier = compute_drift_tier(df, vix)

    assert tier.name == "Normal"


def test_drift_tier_is_conservative_when_only_vix_elevated():
    """
    VIX 25 (Elevated range) but 0% drawdown (Normal) → stays at Normal.
    """
    from finance.apps.assistant._data import compute_drift_tier
    from finance.apps.assistant._data import VixStatus

    df = _make_spy_df(current_price=500.0, peak_price=500.0)  # 0% drawdown
    vix = VixStatus(level=25.0, zone="elevated", direction="rising", is_spiking=False)
    tier = compute_drift_tier(df, vix)

    assert tier.name == "Normal"


def test_drift_tier_correction_when_both_conditions_met():
    """-15% drawdown + VIX 32 → Correction tier."""
    from finance.apps.assistant._data import compute_drift_tier
    from finance.apps.assistant._data import VixStatus

    df = _make_spy_df(current_price=425.0, peak_price=500.0)  # -15% drawdown
    vix = VixStatus(level=32.0, zone="high", direction="spiking", is_spiking=True)
    tier = compute_drift_tier(df, vix)

    assert tier.name == "Correction"
    assert tier.bp_pct == 55


def test_drift_tier_defaults_to_normal_when_no_data():
    """Missing SPY data → safe default of Normal."""
    from finance.apps.assistant._data import compute_drift_tier

    tier = compute_drift_tier(None, None)

    assert tier.name == "Normal"
    assert tier.bp_pct == 30


def test_drift_tier_drawdown_pct_is_positive():
    """drawdown_pct field is always a non-negative percentage."""
    from finance.apps.assistant._data import compute_drift_tier
    from finance.apps.assistant._data import VixStatus

    df = _make_spy_df(current_price=450.0, peak_price=500.0)  # -10% drawdown
    vix = VixStatus(level=20.0, zone="elevated", direction="rising", is_spiking=False)
    tier = compute_drift_tier(df, vix)

    assert tier.drawdown_pct >= 0.0
    assert abs(tier.drawdown_pct - 10.0) < 0.5


# ---------------------------------------------------------------------------
# compute_drift_underlying — eligibility and structure
# ---------------------------------------------------------------------------


def test_drift_underlying_eligible_when_ivp_above_50():
    """High IV relative to history → IVP ≥ 50 → eligible."""
    from finance.apps.assistant._data import compute_drift_underlying

    df = _make_underlying_df(price=500.0, iv=0.25, ivp_history_low=False)
    status = compute_drift_underlying("SPY", df, "Directional", "Core")

    assert status.eligible is True
    assert status.ivp is not None
    assert status.ivp >= 50.0


def test_drift_underlying_not_eligible_when_ivp_below_50():
    """Low IV relative to history → IVP < 50 → not eligible."""
    from finance.apps.assistant._data import compute_drift_underlying

    df = _make_underlying_df(price=500.0, iv=0.25, ivp_history_low=True)
    status = compute_drift_underlying("SPY", df, "Directional", "Core")

    assert status.eligible is False
    assert "Wait" in status.structure or "IVP" in status.structure


def test_drift_underlying_structure_spread_when_below_200sma():
    """Eligible IVP but price below 200d SMA → spreads only."""
    from finance.apps.assistant._data import compute_drift_underlying

    # price=100, SMA200 ~ sma200_price=120 (all bars at 120 except last)
    df = _make_underlying_df(price=100.0, sma200_price=120.0, iv=0.30, ivp_history_low=False)
    status = compute_drift_underlying("SPY", df, "Directional", "Core")

    assert status.price_above_200 is False
    if status.eligible:
        assert "Spread" in status.structure or "spread" in status.structure


def test_drift_underlying_structure_short_put_when_above_200sma_and_eligible():
    """Eligible IVP + price above 200d SMA → short put / XYZ."""
    from finance.apps.assistant._data import compute_drift_underlying

    df = _make_underlying_df(price=110.0, sma200_price=100.0, iv=0.30, ivp_history_low=False)
    status = compute_drift_underlying("SPY", df, "Directional", "Core")

    assert status.price_above_200 is True
    if status.eligible:
        assert "Short put" in status.structure or "XYZ" in status.structure


def test_drift_underlying_ibkr_required_when_no_data():
    """No IBKR data available → structure is 'IBKR required'."""
    from finance.apps.assistant._data import compute_drift_underlying

    status = compute_drift_underlying("GLD", None, "Neutral", "Core")

    assert status.eligible is False
    assert "IBKR" in status.structure
    assert status.ivp is None
    assert status.price_above_200 is None


def test_drift_underlying_iv_gt_hv_uses_last_valid_hv():
    """iv_gt_hv uses last non-NaN HV value (last bar may be NaN)."""
    from finance.apps.assistant._data import compute_drift_underlying

    df = _make_underlying_df(price=100.0, iv=0.30, hv=0.20, ivp_history_low=False)
    # Last HV is NaN by _make_underlying_df design
    assert pd.isna(df["hv"].iloc[-1])

    status = compute_drift_underlying("QQQ", df, "Directional", "Core")

    assert status.iv_gt_hv is not None  # must still compute from penultimate bar


def test_drift_underlying_symbol_and_block_preserved():
    """Symbol and block fields are echoed back correctly."""
    from finance.apps.assistant._data import compute_drift_underlying

    status = compute_drift_underlying("GLD", None, "Neutral", "Core")

    assert status.symbol == "GLD"
    assert status.block == "Neutral"
    assert status.registry_tier == "Core"


# ---------------------------------------------------------------------------
# DriftSection widget (Qt display required)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_drift_section_loads_without_error():
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._swing_panel import DriftSection
    from finance.apps.assistant._data import DriftTier, DriftUnderlyingStatus

    ensure_qt_app()
    section = DriftSection()
    tier = DriftTier(name="Normal", drawdown_pct=1.0, bp_pct=30,
                     structure="XYZ 111, short puts", dte_range="45–60")
    underlyings = [
        DriftUnderlyingStatus("SPY", "Directional", "Core",
                              ivp=65.0, price_above_200=True, iv_gt_hv=True,
                              eligible=True, structure="Short put / XYZ"),
    ]
    section.update_drift(tier, underlyings)


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_drift_section_collapsed_by_default():
    """Content is hidden before the toggle is clicked."""
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._swing_panel import DriftSection

    ensure_qt_app()
    section = DriftSection()
    assert section._content.isHidden()


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_drift_section_toggle_expands_and_collapses():
    """Toggle button shows/hides the content area."""
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._swing_panel import DriftSection

    ensure_qt_app()
    section = DriftSection()
    section._toggle_btn.click()
    assert not section._content.isHidden()
    section._toggle_btn.click()
    assert section._content.isHidden()


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_drift_section_shows_tier_name():
    """Tier name is visible in the header after update_drift."""
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._swing_panel import DriftSection
    from finance.apps.assistant._data import DriftTier

    ensure_qt_app()
    section = DriftSection()
    tier = DriftTier(name="Elevated", drawdown_pct=7.0, bp_pct=40,
                     structure="XYZ 111, short puts", dte_range="45–90")
    section.update_drift(tier, [])
    assert "Elevated" in section._tier_label.text()


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_drift_section_bp_warning_shown_above_50():
    """BP input > 50% shows the warning label."""
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._swing_panel import DriftSection

    ensure_qt_app()
    section = DriftSection()
    section._bp_spinbox.setValue(55.0)
    assert not section._bp_warning.isHidden()


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_drift_section_bp_warning_hidden_below_50():
    """BP input ≤ 50% hides the warning label."""
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._swing_panel import DriftSection

    ensure_qt_app()
    section = DriftSection()
    section._bp_spinbox.setValue(30.0)
    assert section._bp_warning.isHidden()


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_swing_regime_panel_has_drift_section():
    """SwingRegimePanel exposes a drift_section attribute."""
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._swing_panel import DriftSection, SwingRegimePanel

    ensure_qt_app()
    panel = SwingRegimePanel()
    assert hasattr(panel, "drift_section")
    assert isinstance(panel.drift_section, DriftSection)
