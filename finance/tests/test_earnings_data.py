"""Tests for finance.utils.earnings_data — SUE computation and earnings helpers."""
import numpy as np
import pandas as pd
import pytest

from finance.utils.earnings_data import compute_earnings_fields


class TestComputeEarningsFields:
    """Verify SUE, surprise_dir, and consecutive_beats computation."""

    def _make_earnings(self, eps_vals, est_vals, dates=None) -> pd.DataFrame:
        n = len(eps_vals)
        if dates is None:
            dates = pd.date_range("2020-01-15", periods=n, freq="91D")
        return pd.DataFrame({
            "symbol": ["TEST"] * n,
            "date": dates,
            "when": ["post"] * n,
            "period_end_date": dates - pd.Timedelta(days=30),
            "eps": eps_vals,
            "eps_est": est_vals,
        })

    # -- SUE computation --

    def test_sue_positive_for_beat(self):
        df = self._make_earnings([1.50], [1.20])
        result = compute_earnings_fields(df)
        assert result.iloc[0]["sue"] == pytest.approx(0.25)  # (1.50-1.20)/|1.20|

    def test_sue_negative_for_miss(self):
        df = self._make_earnings([0.80], [1.00])
        result = compute_earnings_fields(df)
        assert result.iloc[0]["sue"] == pytest.approx(-0.20)

    def test_sue_zero_for_inline(self):
        df = self._make_earnings([1.00], [1.00])
        result = compute_earnings_fields(df)
        assert result.iloc[0]["sue"] == pytest.approx(0.0)

    def test_sue_nan_when_estimate_is_zero(self):
        df = self._make_earnings([0.50], [0.0])
        result = compute_earnings_fields(df)
        assert pd.isna(result.iloc[0]["sue"])

    def test_sue_nan_when_estimate_is_nan(self):
        df = self._make_earnings([1.00], [np.nan])
        result = compute_earnings_fields(df)
        assert pd.isna(result.iloc[0]["sue"])

    def test_sue_nan_when_eps_is_nan(self):
        df = self._make_earnings([np.nan], [1.00])
        result = compute_earnings_fields(df)
        assert pd.isna(result.iloc[0]["sue"])

    def test_sue_handles_negative_estimate(self):
        # Company expected to lose -0.50, actually lost -0.30 (beat)
        df = self._make_earnings([-0.30], [-0.50])
        result = compute_earnings_fields(df)
        assert result.iloc[0]["sue"] == pytest.approx(0.40)  # (-0.30-(-0.50))/|-0.50|
        assert result.iloc[0]["surprise_dir"] == "beat"

    # -- surprise_dir --

    def test_surprise_dir_beat(self):
        df = self._make_earnings([1.50], [1.20])
        result = compute_earnings_fields(df)
        assert result.iloc[0]["surprise_dir"] == "beat"

    def test_surprise_dir_miss(self):
        df = self._make_earnings([0.80], [1.00])
        result = compute_earnings_fields(df)
        assert result.iloc[0]["surprise_dir"] == "miss"

    def test_surprise_dir_inline(self):
        df = self._make_earnings([1.00], [1.00])
        result = compute_earnings_fields(df)
        assert result.iloc[0]["surprise_dir"] == "inline"

    def test_surprise_dir_unknown_when_missing_data(self):
        df = self._make_earnings([np.nan], [np.nan])
        result = compute_earnings_fields(df)
        assert result.iloc[0]["surprise_dir"] == "unknown"

    # -- consecutive_beats --

    def test_consecutive_beats_counts_streak(self):
        # beat, beat, beat = 1, 2, 3
        df = self._make_earnings([1.5, 1.5, 1.5], [1.0, 1.0, 1.0])
        result = compute_earnings_fields(df)
        assert list(result["consecutive_beats"]) == [1, 2, 3]

    def test_consecutive_beats_resets_on_miss(self):
        # beat, miss, beat = 1, 0, 1
        df = self._make_earnings([1.5, 0.8, 1.5], [1.0, 1.0, 1.0])
        result = compute_earnings_fields(df)
        assert list(result["consecutive_beats"]) == [1, 0, 1]

    def test_consecutive_beats_resets_on_inline(self):
        # beat, inline, beat = 1, 0, 1
        df = self._make_earnings([1.5, 1.0, 1.5], [1.0, 1.0, 1.0])
        result = compute_earnings_fields(df)
        assert list(result["consecutive_beats"]) == [1, 0, 1]

    def test_consecutive_beats_zero_when_no_data(self):
        df = self._make_earnings([np.nan], [np.nan])
        result = compute_earnings_fields(df)
        assert result.iloc[0]["consecutive_beats"] == 0

    # -- empty input --

    def test_empty_dataframe_returns_empty(self):
        df = pd.DataFrame(columns=["symbol", "date", "when", "period_end_date", "eps", "eps_est"])
        result = compute_earnings_fields(df)
        assert result.empty
        assert "sue" in result.columns
        assert "surprise_dir" in result.columns
        assert "consecutive_beats" in result.columns

    # -- column preservation --

    def test_preserves_original_columns(self):
        df = self._make_earnings([1.50], [1.20])
        result = compute_earnings_fields(df)
        for col in ["symbol", "date", "when", "period_end_date", "eps", "eps_est"]:
            assert col in result.columns
