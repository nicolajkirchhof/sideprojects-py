"""Tests for finance.utils.fundamentals — accruals anomaly and F-Score computation."""
import numpy as np
import pandas as pd
import pytest

from finance.utils.fundamentals import compute_accruals, compute_fscore


class TestComputeAccruals:
    """Verify accruals ratio computation: (net_income - CFO) / total_assets."""

    def _make_df(self, net_income, cfo, total_assets, n=1):
        return pd.DataFrame({
            "act_symbol": ["TEST"] * n,
            "date": pd.date_range("2024-03-31", periods=n, freq="QE"),
            "net_income": net_income if isinstance(net_income, list) else [net_income] * n,
            "net_cash_from_operating_activities": cfo if isinstance(cfo, list) else [cfo] * n,
            "total_assets": total_assets if isinstance(total_assets, list) else [total_assets] * n,
        })

    def test_positive_accruals_when_income_exceeds_cash(self):
        df = self._make_df(net_income=100, cfo=60, total_assets=1000)
        result = compute_accruals(df)
        assert result.iloc[0]["accruals_ratio"] == pytest.approx(0.04)  # (100-60)/1000

    def test_negative_accruals_when_cash_exceeds_income(self):
        df = self._make_df(net_income=60, cfo=100, total_assets=1000)
        result = compute_accruals(df)
        assert result.iloc[0]["accruals_ratio"] == pytest.approx(-0.04)

    def test_zero_accruals_when_equal(self):
        df = self._make_df(net_income=100, cfo=100, total_assets=1000)
        result = compute_accruals(df)
        assert result.iloc[0]["accruals_ratio"] == pytest.approx(0.0)

    def test_nan_when_total_assets_is_zero(self):
        df = self._make_df(net_income=100, cfo=60, total_assets=0)
        result = compute_accruals(df)
        assert pd.isna(result.iloc[0]["accruals_ratio"])

    def test_nan_when_fields_missing(self):
        df = pd.DataFrame({"act_symbol": ["TEST"], "date": [pd.Timestamp("2024-03-31")]})
        result = compute_accruals(df)
        assert pd.isna(result.iloc[0]["accruals_ratio"])

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["act_symbol", "date", "net_income",
                                    "net_cash_from_operating_activities", "total_assets"])
        result = compute_accruals(df)
        assert result.empty
        assert "accruals_ratio" in result.columns


class TestComputeFscore:
    """Verify Piotroski F-Score computation (9 binary signals)."""

    def _make_quarter(self, symbol="TEST", date="2024-03-31", **overrides):
        """Create a single quarter row with defaults that produce F-Score = 9."""
        defaults = {
            "act_symbol": symbol,
            "date": pd.Timestamp(date),
            # Income statement
            "net_income": 100,
            "sales": 1000,
            "cost_of_goods": 600,
            "gross_profit": 400,
            "average_shares": 100,
            # Cash flow
            "net_cash_from_operating_activities": 120,
            # Balance sheet assets
            "total_assets": 5000,
            "total_current_assets": 2000,
            # Balance sheet liabilities
            "long_term_debt": 500,
            "total_current_liabilities": 800,
            # Prior quarter for deltas (embedded as _prev columns)
            "prev_net_income": 80,
            "prev_total_assets": 4800,
            "prev_net_cash_from_operating_activities": 90,
            "prev_long_term_debt": 600,
            "prev_total_current_assets": 1800,
            "prev_total_current_liabilities": 900,
            "prev_gross_profit": 350,
            "prev_sales": 950,
            "prev_average_shares": 100,
        }
        defaults.update(overrides)
        return defaults

    def _make_df(self, quarters):
        return pd.DataFrame(quarters)

    def test_perfect_score_is_9(self):
        df = self._make_df([self._make_quarter()])
        result = compute_fscore(df)
        assert result.iloc[0]["fscore"] == 9

    def test_negative_roa_reduces_score(self):
        df = self._make_df([self._make_quarter(net_income=-10)])
        result = compute_fscore(df)
        assert result.iloc[0]["fscore"] < 9
        assert result.iloc[0]["f_roa_positive"] == 0

    def test_negative_cfo_reduces_score(self):
        df = self._make_df([self._make_quarter(net_cash_from_operating_activities=-10)])
        result = compute_fscore(df)
        assert result.iloc[0]["f_cfo_positive"] == 0

    def test_increasing_debt_reduces_score(self):
        # Current LTD > prev LTD
        df = self._make_df([self._make_quarter(long_term_debt=700, prev_long_term_debt=500)])
        result = compute_fscore(df)
        assert result.iloc[0]["f_delta_ltd"] == 0

    def test_dilution_reduces_score(self):
        df = self._make_df([self._make_quarter(average_shares=120, prev_average_shares=100)])
        result = compute_fscore(df)
        assert result.iloc[0]["f_no_dilution"] == 0

    def test_score_zero_when_all_negative(self):
        q = self._make_quarter(
            net_income=-100,
            net_cash_from_operating_activities=-50,
            prev_net_income=-80,   # ROA worsening
            long_term_debt=800, prev_long_term_debt=500,  # debt increasing
            total_current_assets=1500, total_current_liabilities=1200,  # current ratio declining
            prev_total_current_assets=1800, prev_total_current_liabilities=900,
            average_shares=120, prev_average_shares=100,  # dilution
            gross_profit=300, prev_gross_profit=400,  # margin declining
            sales=900, cost_of_goods=700,  # turnover declining
            prev_sales=950,
        )
        df = self._make_df([q])
        result = compute_fscore(df)
        assert result.iloc[0]["fscore"] <= 2

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = compute_fscore(df)
        assert result.empty
        assert "fscore" in result.columns

    def test_preserves_symbol_and_date(self):
        df = self._make_df([self._make_quarter(symbol="AAPL", date="2024-06-30")])
        result = compute_fscore(df)
        assert result.iloc[0]["act_symbol"] == "AAPL"
