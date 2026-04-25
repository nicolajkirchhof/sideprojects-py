import numpy as np
import pandas as pd
import pytest

from finance.apps.assistant._data import (
    SLOPE_LOOKBACK,
    SLOPE_THRESHOLD,
    SMA_FAST,
    SMA_LONG,
    SMA_SHORT,
    TrendStatus,
    VixStatus,
    classify_slope,
    compute_go_nogo,
    compute_trend_status,
    compute_vix_status,
)


def _make_daily(closes: list[float]) -> pd.DataFrame:
    """Build a minimal daily DataFrame with a DatetimeIndex and 'c' column."""
    dates = pd.date_range(end="2025-06-01", periods=len(closes), freq="D")
    return pd.DataFrame({"c": closes}, index=dates)


def _rising_series(n: int = 250, start: float = 400.0, step: float = 0.5) -> list[float]:
    """Steadily rising prices — will produce rising SMAs."""
    return [start + i * step for i in range(n)]


def _falling_series(n: int = 250, start: float = 600.0, step: float = 0.5) -> list[float]:
    """Steadily falling prices — will produce falling SMAs."""
    return [start - i * step for i in range(n)]


def _flat_series(n: int = 250, value: float = 500.0) -> list[float]:
    """Flat prices — will produce flat SMAs."""
    rng = np.random.default_rng(42)
    return [value + rng.normal(0, 0.01) for _ in range(n)]


# ---------------------------------------------------------------------------
# classify_slope
# ---------------------------------------------------------------------------

class TestClassifySlope:
    def test_rising(self):
        sma = pd.Series([100.0 + i * 0.1 for i in range(SLOPE_LOOKBACK)])
        assert classify_slope(sma) == "rising"

    def test_falling(self):
        sma = pd.Series([100.0 - i * 0.1 for i in range(SLOPE_LOOKBACK)])
        assert classify_slope(sma) == "falling"

    def test_flat(self):
        sma = pd.Series([100.0] * SLOPE_LOOKBACK)
        assert classify_slope(sma) == "flat"

    def test_barely_above_threshold_is_rising(self):
        base = 100.0
        change = base * (SLOPE_THRESHOLD + 0.0001)
        sma = pd.Series(np.linspace(base, base + change, SLOPE_LOOKBACK))
        assert classify_slope(sma) == "rising"

    def test_too_short_returns_flat(self):
        sma = pd.Series([100.0, 101.0])
        assert classify_slope(sma) == "flat"


# ---------------------------------------------------------------------------
# compute_trend_status
# ---------------------------------------------------------------------------

class TestComputeTrendStatus:
    def test_above_both_smas_rising(self):
        closes = _rising_series(250)
        # Push last price well above both SMAs
        closes[-1] = closes[-1] + 50
        df = _make_daily(closes)
        status = compute_trend_status("SPY", df)

        assert isinstance(status, TrendStatus)
        assert status.symbol == "SPY"
        assert status.price_above_50
        assert status.price_above_200
        assert status.sma_50_slope == "rising"
        assert status.sma_200_slope == "rising"

    def test_below_both_smas_falling(self):
        closes = _falling_series(250)
        closes[-1] = closes[-1] - 50
        df = _make_daily(closes)
        status = compute_trend_status("SPY", df)

        assert not status.price_above_50
        assert not status.price_above_200
        assert status.sma_200_slope == "falling"

    def test_last_price_is_most_recent_close(self):
        closes = _rising_series(250)
        df = _make_daily(closes)
        status = compute_trend_status("SPY", df)
        assert status.last_price == pytest.approx(closes[-1])

    def test_sma_values_are_correct(self):
        closes = _rising_series(250)
        df = _make_daily(closes)
        status = compute_trend_status("SPY", df)

        expected_20 = np.mean(closes[-SMA_FAST:])
        expected_50 = np.mean(closes[-SMA_SHORT:])
        expected_200 = np.mean(closes[-SMA_LONG:])
        assert status.sma_20 == pytest.approx(expected_20, rel=1e-6)
        assert status.sma_50 == pytest.approx(expected_50, rel=1e-6)
        assert status.sma_200 == pytest.approx(expected_200, rel=1e-6)

    def test_insufficient_data_returns_none(self):
        df = _make_daily([100.0] * 50)
        assert compute_trend_status("SPY", df) is None

    def test_empty_dataframe_returns_none(self):
        df = pd.DataFrame({"c": []}, index=pd.DatetimeIndex([]))
        assert compute_trend_status("SPY", df) is None


# ---------------------------------------------------------------------------
# compute_vix_status
# ---------------------------------------------------------------------------

class TestComputeVixStatus:
    def _vix_df(self, closes: list[float]) -> pd.DataFrame:
        dates = pd.date_range(end="2025-06-01", periods=len(closes), freq="D")
        return pd.DataFrame({"c": closes}, index=dates)

    def test_low_zone(self):
        closes = [15.0] * 20
        status = compute_vix_status(self._vix_df(closes))
        assert status.zone == "low"
        assert status.level == pytest.approx(15.0)

    def test_elevated_zone(self):
        closes = [25.0] * 20
        status = compute_vix_status(self._vix_df(closes))
        assert status.zone == "elevated"

    def test_high_zone(self):
        closes = [35.0] * 20
        status = compute_vix_status(self._vix_df(closes))
        assert status.zone == "high"

    def test_boundary_20_is_elevated(self):
        closes = [20.0] * 20
        status = compute_vix_status(self._vix_df(closes))
        assert status.zone == "elevated"

    def test_boundary_30_is_high(self):
        closes = [30.0] * 20
        status = compute_vix_status(self._vix_df(closes))
        assert status.zone == "high"

    def test_falling_direction(self):
        closes = [25.0 - i * 0.5 for i in range(20)]
        status = compute_vix_status(self._vix_df(closes))
        assert status.direction == "falling"

    def test_rising_direction(self):
        closes = [15.0 + i * 0.5 for i in range(20)]
        status = compute_vix_status(self._vix_df(closes))
        assert status.direction == "rising"

    def test_spiking_over_20_pct_in_5_days(self):
        closes = [15.0] * 15 + [15.0, 15.5, 16.0, 17.0, 19.0]
        # 19 / 15 - 1 = 26.7% > 20%
        status = compute_vix_status(self._vix_df(closes))
        assert status.direction == "spiking"
        assert status.is_spiking

    def test_not_spiking_under_20_pct(self):
        closes = [15.0] * 15 + [15.0, 15.2, 15.4, 15.6, 17.0]
        # 17 / 15 - 1 = 13.3% < 20%
        status = compute_vix_status(self._vix_df(closes))
        assert not status.is_spiking

    def test_empty_returns_none(self):
        df = pd.DataFrame({"c": []}, index=pd.DatetimeIndex([]))
        assert compute_vix_status(df) is None


# ---------------------------------------------------------------------------
# compute_go_nogo
# ---------------------------------------------------------------------------

class TestComputeGoNogo:
    def _trend(self, above_50=True, above_200=True, slope_200="rising") -> TrendStatus:
        return TrendStatus(
            symbol="SPY",
            last_price=500.0,
            sma_20=495.0,
            sma_50=490.0 if above_50 else 510.0,
            sma_200=480.0 if above_200 else 520.0,
            price_above_20=True,
            price_above_50=above_50,
            price_above_200=above_200,
            sma_20_slope="rising",
            sma_50_slope="rising",
            sma_200_slope=slope_200,
        )

    def _vix(self, zone="low", is_spiking=False) -> VixStatus:
        return VixStatus(
            level=15.0 if zone == "low" else 25.0 if zone == "elevated" else 35.0,
            zone=zone,
            direction="falling",
            is_spiking=is_spiking,
        )

    def test_go_all_green(self):
        assert compute_go_nogo(self._trend(), self._vix()) == "GO"

    def test_nogo_spy_below_both(self):
        trend = self._trend(above_50=False, above_200=False, slope_200="falling")
        assert compute_go_nogo(trend, self._vix()) == "NO-GO"

    def test_nogo_vix_high(self):
        assert compute_go_nogo(self._trend(), self._vix(zone="high")) == "NO-GO"

    def test_nogo_vix_spiking(self):
        assert compute_go_nogo(self._trend(), self._vix(is_spiking=True)) == "NO-GO"

    def test_caution_mixed_sma(self):
        trend = self._trend(above_50=True, above_200=False)
        assert compute_go_nogo(trend, self._vix()) == "CAUTION"

    def test_caution_200_flat(self):
        trend = self._trend(slope_200="flat")
        assert compute_go_nogo(trend, self._vix()) == "CAUTION"

    def test_caution_200_falling_but_above(self):
        trend = self._trend(slope_200="falling")
        assert compute_go_nogo(trend, self._vix()) == "CAUTION"

    def test_caution_vix_elevated(self):
        trend = self._trend()
        assert compute_go_nogo(trend, self._vix(zone="elevated")) == "CAUTION"

    def test_nogo_takes_precedence_over_caution(self):
        trend = self._trend(above_50=False, above_200=False, slope_200="falling")
        vix = self._vix(zone="high", is_spiking=True)
        assert compute_go_nogo(trend, vix) == "NO-GO"

    def test_none_trend_returns_nogo(self):
        assert compute_go_nogo(None, self._vix()) == "NO-GO"

    def test_none_vix_returns_caution(self):
        assert compute_go_nogo(self._trend(), None) == "CAUTION"
