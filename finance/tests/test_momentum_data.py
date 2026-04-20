"""Tests for finance.utils.momentum_data — extended columns and earnings helpers."""
import numpy as np
import pandas as pd
import pytest

from finance.utils.momentum_data import (
    _required_columns,
    load_earnings_events,
    load_and_prep_data,
    _DATASET_DIR,
)


class TestRequiredColumns:
    """Verify _required_columns includes all columns needed for backtesting."""

    def setup_method(self):
        self.cols = _required_columns()
        self.col_set = set(self.cols)

    # -- Extended daily range (cpct26..60) --

    def test_includes_cpct_up_to_60(self):
        for i in range(1, 61):
            assert f"cpct{i}" in self.col_set, f"cpct{i} missing"

    def test_includes_daily_dist_metrics_up_to_60(self):
        for metric in ["ma5_dist", "ma10_dist", "ma20_dist", "ma50_dist"]:
            for i in range(1, 61):
                assert f"{metric}{i}" in self.col_set, f"{metric}{i} missing"

    def test_includes_daily_slope_and_vol_metrics_up_to_60(self):
        for metric in ["ma5_slope", "ma10_slope", "ma20_slope", "ma50_slope",
                        "rvol20", "hv20", "atrp20"]:
            for i in range(1, 61):
                assert f"{metric}{i}" in self.col_set, f"{metric}{i} missing"

    # -- Extended weekly range (w_cpct9..12) --

    def test_includes_weekly_cpct_up_to_12(self):
        for i in range(1, 13):
            assert f"w_cpct{i}" in self.col_set, f"w_cpct{i} missing"

    def test_includes_weekly_dist_metrics_up_to_12(self):
        for metric in ["w_ma5_dist", "w_ma10_dist", "w_ma20_dist", "w_ma50_dist"]:
            for i in range(1, 13):
                assert f"{metric}{i}" in self.col_set, f"{metric}{i} missing"

    def test_includes_weekly_slope_and_vol_metrics_up_to_12(self):
        for metric in ["w_ma5_slope", "w_ma10_slope", "w_ma20_slope", "w_ma50_slope",
                        "w_rvol20", "w_hv20", "w_atrp20"]:
            for i in range(1, 13):
                assert f"{metric}{i}" in self.col_set, f"{metric}{i} missing"

    # -- New earnings / event columns --

    def test_includes_earnings_pead_columns(self):
        for col in ["sue", "surprise_dir", "close_in_range", "consecutive_beats"]:
            assert col in self.col_set, f"{col} missing"

    def test_includes_new_event_type_flags(self):
        for col in ["evt_episodic_pivot", "evt_pre_earnings",
                     "evt_ema_reclaim", "evt_selloff"]:
            assert col in self.col_set, f"{col} missing"

    # -- Backward compat: original columns still present --

    def test_still_includes_original_core_columns(self):
        for col in ["date", "original_price", "c0", "cpct0", "atrp200",
                     "is_earnings", "is_etf", "spy0", "spy5", "market_cap_class",
                     "evt_atrp_breakout", "evt_green_line_breakout", "evt_bb_lower_touch",
                     "1M_chg", "3M_chg", "6M_chg", "12M_chg"]:
            assert col in self.col_set, f"{col} missing"

    def test_includes_gappct(self):
        assert "gappct" in self.col_set

    def test_columns_are_sorted(self):
        assert self.cols == sorted(self.cols)


class TestLoadEarningsEvents:
    """Verify load_earnings_events filters correctly on SUE and direction."""

    @pytest.fixture()
    def earnings_parquet(self, tmp_path):
        """Write a minimal yearly parquet file and patch _DATASET_DIR."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-03-01", "2024-06-01", "2024-09-01"]),
            "c0": [100.0, 110.0, 90.0],
            "original_price": [100.0, 110.0, 90.0],
            "cpct0": [5.0, -3.0, 0.0],
            "atrp200": [2.0, 2.0, 2.0],
            "is_earnings": [True, True, True],
            "is_etf": [False, False, False],
            "spy0": [450.0, 455.0, 440.0],
            "spy5": [452.0, 453.0, 442.0],
            "eps": [1.50, 0.80, 1.00],
            "eps_est": [1.20, 1.00, 1.00],
            "sue": [0.25, -0.20, 0.0],
            "surprise_dir": ["beat", "miss", "inline"],
        })
        out = tmp_path / "all_2024.parquet"
        df.to_parquet(out, index=False)
        return tmp_path

    def test_filters_beats_only(self, earnings_parquet, monkeypatch):
        monkeypatch.setattr("finance.utils.momentum_data._DATASET_DIR", str(earnings_parquet))
        result = load_earnings_events(range(2024, 2025), direction="beat")
        assert len(result) == 1
        assert result.iloc[0]["surprise_dir"] == "beat"

    def test_filters_misses_only(self, earnings_parquet, monkeypatch):
        monkeypatch.setattr("finance.utils.momentum_data._DATASET_DIR", str(earnings_parquet))
        result = load_earnings_events(range(2024, 2025), direction="miss")
        assert len(result) == 1
        assert result.iloc[0]["surprise_dir"] == "miss"

    def test_filters_min_sue(self, earnings_parquet, monkeypatch):
        monkeypatch.setattr("finance.utils.momentum_data._DATASET_DIR", str(earnings_parquet))
        result = load_earnings_events(range(2024, 2025), min_sue=0.0)
        # sue=0.25 (beat) and sue=0.0 (inline) pass; sue=-0.20 (miss) fails
        assert len(result) == 2

    def test_filters_max_sue(self, earnings_parquet, monkeypatch):
        monkeypatch.setattr("finance.utils.momentum_data._DATASET_DIR", str(earnings_parquet))
        result = load_earnings_events(range(2024, 2025), max_sue=0.0)
        # sue=-0.20 (miss) and sue=0.0 (inline) pass; sue=0.25 (beat) fails
        assert len(result) == 2

    def test_returns_empty_for_no_matching_year(self, earnings_parquet, monkeypatch):
        monkeypatch.setattr("finance.utils.momentum_data._DATASET_DIR", str(earnings_parquet))
        result = load_earnings_events(range(2020, 2021))
        assert result.empty


class TestBacktestReport:
    """Tests for backtest_report.generate_report."""

    def _make_df(self, n: int = 100) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=n, freq="B"),
            "is_earnings": [True] * n,
            "market_cap_class": rng.choice(["Large", "Mid", "Small"], size=n),
            "spy_class": rng.choice(["Supporting", "Neutral", "Non-Supporting"], size=n),
            "cpct1": rng.normal(0.2, 2.0, n),
            "cpct5": rng.normal(0.5, 3.0, n),
            "cpct10": rng.normal(1.0, 4.0, n),
            "cpct20": rng.normal(1.5, 5.0, n),
            "cpct40": rng.normal(2.0, 6.0, n),
            "cpct60": rng.normal(2.5, 7.0, n),
        })

    def test_report_has_correct_structure(self):
        from finance.swing_pm.backtests.backtest_report import generate_report
        df = self._make_df()
        report = generate_report(df=df, label="Test Report", horizons=[1, 5, 10])
        assert report.label == "Test Report"
        assert report.total_events == 100
        assert len(report.horizons) == 3
        assert report.horizons[0].horizon == 1
        assert report.horizons[1].horizon == 5
        assert report.horizons[2].horizon == 10

    def test_win_rate_is_percentage(self):
        from finance.swing_pm.backtests.backtest_report import generate_report
        df = self._make_df()
        report = generate_report(df=df, label="Test", horizons=[1])
        wr = report.horizons[0].win_rate
        assert 0 <= wr <= 100

    def test_cost_adjusted_mean_is_less_than_mean(self):
        from finance.swing_pm.backtests.backtest_report import generate_report
        df = self._make_df()
        report = generate_report(df=df, label="Test", horizons=[5])
        h = report.horizons[0]
        assert h.cost_adjusted_mean < h.mean_return

    def test_segments_populated_when_requested(self):
        from finance.swing_pm.backtests.backtest_report import generate_report
        df = self._make_df()
        report = generate_report(
            df=df, label="Test", horizons=[1, 5],
            segment_cols=["market_cap_class"],
        )
        assert len(report.segments) == 1
        assert report.segments[0].segment_col == "market_cap_class"
        assert len(report.segments[0].rows) == 3  # Large, Mid, Small

    def test_markdown_contains_key_sections(self):
        from finance.swing_pm.backtests.backtest_report import generate_report
        df = self._make_df()
        report = generate_report(
            df=df, label="My Label", horizons=[1, 5],
            segment_cols=["spy_class"],
        )
        assert "# Backtest Report: My Label" in report.markdown
        assert "## Forward Returns by Horizon" in report.markdown
        assert "## By spy_class" in report.markdown
        assert "Net Mean%" in report.markdown

    def test_empty_df_produces_zero_metrics(self):
        from finance.swing_pm.backtests.backtest_report import generate_report
        df = pd.DataFrame(columns=["date", "cpct1", "cpct5"])
        report = generate_report(df=df, label="Empty", horizons=[1, 5])
        assert report.total_events == 0
        assert all(h.n_events == 0 for h in report.horizons)

    def test_missing_return_column_produces_zero_metrics(self):
        from finance.swing_pm.backtests.backtest_report import generate_report
        df = self._make_df()
        report = generate_report(df=df, label="Test", horizons=[99])
        assert report.horizons[0].n_events == 0
        assert report.horizons[0].win_rate == 0

    def test_save_creates_file(self, tmp_path):
        from finance.swing_pm.backtests.backtest_report import generate_report
        df = self._make_df(10)
        report = generate_report(df=df, label="Save Test", horizons=[1])
        out = str(tmp_path / "sub" / "report.md")
        report.save(out)
        import os
        assert os.path.exists(out)
        content = open(out).read()
        assert "Save Test" in content
