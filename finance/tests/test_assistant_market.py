"""
Tests for finance.apps.assistant._market — market instrument loading.

All tests use synthetic data; no live IBKR connection required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from finance.apps.assistant._market import (
    MARKET_INSTRUMENTS,
    _swing_row_to_archive,
    load_all_market,
    load_market_instrument,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df_day(**overrides) -> pd.DataFrame:
    """Return a synthetic single-row df_day from swing_indicators."""
    row = {
        "o": 500.0, "h": 505.0, "l": 497.0, "c": 503.0, "v": 80_000_000.0,
        "pct": 0.6,
        "gappct": 0.1,
        "rvol20": 1.15,
        "atrp20": 1.2,
        "ma50": 490.0,
        "ma50_slope": 0.49,       # → slope_50d_sma = (0.49 / 490) * 100 ≈ 0.1
        "ma200": 460.0,
        "ma200_slope": 0.92,      # → slope_200d_sma = (0.92 / 460) * 100 = 0.2
        "ma50_dist": 2.65,
        "bb_upper": 515.0,
        "bb_lower": 478.0,
        "bb_basis": 496.0,
        "squeeze_on": False,
        "hv20": 13.5,
        "iv_pct": 42.0,
        "1M_chg": 5.2,
        "3M_chg": 12.1,
        "6M_chg": 18.3,
        "12M_chg": 24.7,
    }
    row.update(overrides)
    # Extend with enough history for 5d change
    dates = pd.date_range("2026-01-01", periods=10, freq="B")
    closes = [490.0, 492.0, 493.0, 494.0, 496.0, 498.0, 499.0, 500.0, 502.0, row["c"]]
    df = pd.DataFrame({"c": closes}, index=dates)
    # Add all other row columns to last row only (single-row approach for last row extraction)
    for k, v in row.items():
        if k != "c":
            df[k] = [None] * 9 + [v]
    return df


# ---------------------------------------------------------------------------
# 1. MARKET_INSTRUMENTS registry
# ---------------------------------------------------------------------------

class TestMarketInstruments:
    def test_has_13_instruments(self) -> None:
        assert len(MARKET_INSTRUMENTS) == 13

    def test_all_have_required_keys(self) -> None:
        for instr in MARKET_INSTRUMENTS:
            assert "symbol" in instr
            assert "category" in instr
            assert "stk_symbol" in instr

    def test_categories_are_valid(self) -> None:
        valid = {"Indices", "Volatility", "Commodities-Energy", "Commodities-Metals", "Bonds", "Forex"}
        for instr in MARKET_INSTRUMENTS:
            assert instr["category"] in valid, f"Unknown category: {instr['category']}"

    def test_spy_qqq_iwm_present(self) -> None:
        symbols = {i["symbol"] for i in MARKET_INSTRUMENTS}
        assert {"SPY", "QQQ", "IWM"}.issubset(symbols)

    def test_vix_uses_dollar_prefix(self) -> None:
        vix = next(i for i in MARKET_INSTRUMENTS if i["symbol"] == "VIX")
        assert vix["stk_symbol"] == "$VIX"

    def test_forex_use_caret_prefix(self) -> None:
        forex = [i for i in MARKET_INSTRUMENTS if i["category"] == "Forex"]
        for instr in forex:
            assert instr["stk_symbol"].startswith("^"), (
                f"Forex {instr['symbol']} stk_symbol should start with ^"
            )


# ---------------------------------------------------------------------------
# 2. _swing_row_to_archive — column mapping
# ---------------------------------------------------------------------------

class TestSwingRowToArchive:
    def _last_row(self, **overrides) -> tuple[pd.Series, pd.DataFrame]:
        df = _make_df_day(**overrides)
        return df.iloc[-1], df

    def test_price_mapped_from_c(self) -> None:
        last, df = self._last_row(c=503.0)
        row = _swing_row_to_archive("SPY", "Indices", last, df)
        assert row["price"] == pytest.approx(503.0)

    def test_change_pct_mapped_from_pct(self) -> None:
        last, df = self._last_row(pct=0.6)
        row = _swing_row_to_archive("SPY", "Indices", last, df)
        assert row["change_pct"] == pytest.approx(0.6)

    def test_change_1m_pct_mapped(self) -> None:
        last, df = self._last_row(**{"1M_chg": 5.2})
        row = _swing_row_to_archive("SPY", "Indices", last, df)
        assert row["change_1m_pct"] == pytest.approx(5.2)

    def test_change_52w_pct_mapped_from_12m(self) -> None:
        last, df = self._last_row(**{"12M_chg": 24.7})
        row = _swing_row_to_archive("SPY", "Indices", last, df)
        assert row["change_52w_pct"] == pytest.approx(24.7)

    def test_rvol_mapped_from_rvol20(self) -> None:
        last, df = self._last_row(rvol20=1.15)
        row = _swing_row_to_archive("SPY", "Indices", last, df)
        assert row["rvol_20d"] == pytest.approx(1.15)

    def test_atr_mapped_from_atrp20(self) -> None:
        last, df = self._last_row(atrp20=1.2)
        row = _swing_row_to_archive("SPY", "Indices", last, df)
        assert row["atr_pct_20d"] == pytest.approx(1.2)

    def test_pct_from_50d_sma_from_ma50_dist(self) -> None:
        last, df = self._last_row(ma50_dist=2.65)
        row = _swing_row_to_archive("SPY", "Indices", last, df)
        assert row["pct_from_50d_sma"] == pytest.approx(2.65)

    def test_iv_percentile_from_iv_pct(self) -> None:
        last, df = self._last_row(iv_pct=42.0)
        row = _swing_row_to_archive("SPY", "Indices", last, df)
        assert row["iv_percentile"] == pytest.approx(42.0)

    def test_hv20_populated(self) -> None:
        last, df = self._last_row(hv20=13.5)
        row = _swing_row_to_archive("SPY", "Indices", last, df)
        assert row["hv20"] == pytest.approx(13.5)

    def test_row_type_is_market(self) -> None:
        last, df = self._last_row()
        row = _swing_row_to_archive("SPY", "Indices", last, df)
        assert row["row_type"] == "market"

    def test_category_set(self) -> None:
        last, df = self._last_row()
        row = _swing_row_to_archive("GLD", "Commodities-Metals", last, df)
        assert row["category"] == "Commodities-Metals"


# ---------------------------------------------------------------------------
# 3. Slope normalization
# ---------------------------------------------------------------------------

class TestSlopeNormalization:
    def test_slope_50d_normalized(self) -> None:
        # ma50_slope = 0.49, ma50 = 490 → (0.49/490)*100 = 0.1
        last, df = self._last_row(ma50_slope=0.49, ma50=490.0)
        row = _swing_row_to_archive("SPY", "Indices", last, df)
        assert row["slope_50d_sma"] == pytest.approx(0.1, rel=1e-3)

    def test_slope_200d_normalized(self) -> None:
        # ma200_slope = 0.92, ma200 = 460 → (0.92/460)*100 = 0.2
        last, df = self._last_row(ma200_slope=0.92, ma200=460.0)
        row = _swing_row_to_archive("SPY", "Indices", last, df)
        assert row["slope_200d_sma"] == pytest.approx(0.2, rel=1e-3)

    def test_slope_zero_ma_gives_na(self) -> None:
        last, df = self._last_row(ma50_slope=0.5, ma50=0.0)
        row = _swing_row_to_archive("SPY", "Indices", last, df)
        assert pd.isna(row["slope_50d_sma"])

    def _last_row(self, **overrides) -> tuple[pd.Series, pd.DataFrame]:
        df = _make_df_day(**overrides)
        return df.iloc[-1], df


# ---------------------------------------------------------------------------
# 4. BB% computation
# ---------------------------------------------------------------------------

class TestBBPct:
    def _last_row(self, **overrides) -> tuple[pd.Series, pd.DataFrame]:
        df = _make_df_day(**overrides)
        return df.iloc[-1], df

    def test_bb_pct_computed_correctly(self) -> None:
        # c=503, bb_lower=478, bb_upper=515 → (503-478)/(515-478)*100 = 25/37*100 ≈ 67.57
        last, df = self._last_row(c=503.0, bb_lower=478.0, bb_upper=515.0)
        row = _swing_row_to_archive("SPY", "Indices", last, df)
        assert row["bb_pct"] == pytest.approx(25.0 / 37.0 * 100, rel=1e-3)

    def test_bb_pct_zero_width_gives_na(self) -> None:
        # bb_upper == bb_lower → division by zero → NA
        last, df = self._last_row(bb_upper=500.0, bb_lower=500.0)
        row = _swing_row_to_archive("SPY", "Indices", last, df)
        assert pd.isna(row["bb_pct"])


# ---------------------------------------------------------------------------
# 5. TTM squeeze mapping
# ---------------------------------------------------------------------------

class TestTtmSqueezeMapping:
    def _last_row(self, **overrides) -> tuple[pd.Series, pd.DataFrame]:
        df = _make_df_day(**overrides)
        return df.iloc[-1], df

    def test_squeeze_on_true_gives_on(self) -> None:
        last, df = self._last_row(squeeze_on=True)
        row = _swing_row_to_archive("SPY", "Indices", last, df)
        assert row["ttm_squeeze"] == "On"

    def test_squeeze_on_false_gives_off(self) -> None:
        last, df = self._last_row(squeeze_on=False)
        row = _swing_row_to_archive("SPY", "Indices", last, df)
        assert row["ttm_squeeze"] == "Off"

    def test_squeeze_missing_gives_na(self) -> None:
        df = _make_df_day()
        df.loc[df.index[-1], "squeeze_on"] = None
        last = df.iloc[-1]
        row = _swing_row_to_archive("SPY", "Indices", last, df)
        assert pd.isna(row["ttm_squeeze"])


# ---------------------------------------------------------------------------
# 6. VIX-specific validation
# ---------------------------------------------------------------------------

class TestVIXValidation:
    def test_vix_volume_na_is_valid(self) -> None:
        """VIX has no trading volume — NA is expected and must not error."""
        df = _make_df_day(v=None)
        df.loc[df.index[-1], "v"] = None
        last = df.iloc[-1]
        row = _swing_row_to_archive("VIX", "Volatility", last, df)
        assert pd.isna(row["volume"])

    def test_vix_zero_price_returns_none_from_loader(self) -> None:
        """Price == 0 is invalid for VIX; instrument load should return None."""
        mock_data = MagicMock()
        mock_data.empty = False
        df = _make_df_day(c=0.0)
        mock_data.df_day = df
        instr = {"symbol": "VIX", "category": "Volatility", "stk_symbol": "$VIX"}
        with patch("finance.apps.assistant._market.SwingTradingData", return_value=mock_data):
            result = load_market_instrument(instr)
        assert result is None


# ---------------------------------------------------------------------------
# 7. Failure isolation in load_all_market
# ---------------------------------------------------------------------------

class TestLoadAllMarket:
    def test_one_failure_does_not_prevent_others(self) -> None:
        """If one instrument fails, the rest should still be returned."""
        call_count = 0

        def _fake_loader(instr: dict, **_kwargs) -> dict | None:
            nonlocal call_count
            call_count += 1
            if instr["symbol"] == "QQQ":
                return None  # simulate failure
            return {
                "symbol": instr["symbol"],
                "row_type": "market",
                "category": instr["category"],
                "price": 100.0,
                **{col: pd.NA for col in []},  # rest NA
            }

        with patch("finance.apps.assistant._market.load_market_instrument", side_effect=_fake_loader):
            df = load_all_market()

        assert call_count == 13  # all instruments attempted
        assert "QQQ" not in df["symbol"].tolist()
        assert "SPY" in df["symbol"].tolist()

    def test_returns_dataframe(self) -> None:
        """load_all_market returns a DataFrame even when some instruments fail."""
        with patch("finance.apps.assistant._market.load_market_instrument", return_value=None):
            df = load_all_market()
        assert isinstance(df, pd.DataFrame)
