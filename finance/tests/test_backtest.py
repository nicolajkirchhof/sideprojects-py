from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from finance.utils.backtest import InstrumentClass, cost_per_trade, net_pnl
from finance.utils.swing_backtest import (
    TradeEntry,
    load_trade_entries,
    reconstruct_candidate,
    reconstruct_candidate_from_event_row,
    run_event_scoring,
    run_scoring_backtest,
)


class TestCostPerTrade:
    def test_stock_cost_is_commission_plus_slippage(self):
        # notional=10_000: slippage = 10_000 * 0.05% * 2 = $10.00, commission = $1.00
        cost = cost_per_trade(InstrumentClass.STOCK, notional=10_000)
        assert cost == pytest.approx(11.00)

    def test_stock_cost_scales_with_notional(self):
        cost_small = cost_per_trade(InstrumentClass.STOCK, notional=1_000)
        cost_large = cost_per_trade(InstrumentClass.STOCK, notional=100_000)
        assert cost_large > cost_small

    def test_stock_cost_is_positive_for_any_notional(self):
        assert cost_per_trade(InstrumentClass.STOCK, notional=100) > 0

    def test_future_cost_is_spread_plus_commission_per_contract(self):
        # tick_spread=$12.50, commission=$10.00, 1 contract = $22.50
        cost = cost_per_trade(InstrumentClass.FUTURE, notional=12.50, n_contracts=1)
        assert cost == pytest.approx(22.50)

    def test_future_cost_scales_with_contracts(self):
        cost_1 = cost_per_trade(InstrumentClass.FUTURE, notional=12.50, n_contracts=1)
        cost_2 = cost_per_trade(InstrumentClass.FUTURE, notional=12.50, n_contracts=2)
        assert cost_2 == pytest.approx(cost_1 * 2)

    def test_option_cost_is_fraction_of_bid_ask_width(self):
        # bid-ask width=$2.00, 20% per side, 1 contract = $0.40
        cost = cost_per_trade(InstrumentClass.OPTION, notional=2.00, n_contracts=1)
        assert cost == pytest.approx(0.40)

    def test_option_cost_scales_with_contracts(self):
        cost_1 = cost_per_trade(InstrumentClass.OPTION, notional=2.00, n_contracts=1)
        cost_5 = cost_per_trade(InstrumentClass.OPTION, notional=2.00, n_contracts=5)
        assert cost_5 == pytest.approx(cost_1 * 5)

    def test_all_instrument_classes_return_positive_cost(self):
        assert cost_per_trade(InstrumentClass.STOCK, notional=5_000) > 0
        assert cost_per_trade(InstrumentClass.FUTURE, notional=12.50) > 0
        assert cost_per_trade(InstrumentClass.OPTION, notional=1.50) > 0

    def test_stock_raises_when_n_contracts_is_not_one(self):
        with pytest.raises(ValueError):
            cost_per_trade(InstrumentClass.STOCK, notional=10_000, n_contracts=5)


class TestNetPnl:
    def test_net_pnl_is_less_than_gross(self):
        gross = 500.0
        net = net_pnl(gross, InstrumentClass.STOCK, notional=10_000)
        assert net < gross

    def test_net_pnl_deducts_exact_cost(self):
        gross = 500.0
        cost = cost_per_trade(InstrumentClass.FUTURE, notional=12.50)
        net = net_pnl(gross, InstrumentClass.FUTURE, notional=12.50)
        assert net == pytest.approx(gross - cost)

    def test_net_pnl_can_be_negative_when_gross_is_small(self):
        net = net_pnl(0.50, InstrumentClass.STOCK, notional=10_000)
        assert net < 0

    def test_net_pnl_preserves_sign_for_large_winner(self):
        net = net_pnl(10_000.0, InstrumentClass.FUTURE, notional=12.50)
        assert net > 0


# ---------------------------------------------------------------------------
# Helpers — synthetic IBKR parquet
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 300, start_price: float = 100.0) -> pd.DataFrame:
    """Synthetic daily OHLCV parquet with realistic price series."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0005, 0.015, n)
    closes = start_price * np.exp(np.cumsum(returns))
    dates = pd.date_range("2023-01-02", periods=n, freq="B")

    highs = closes * (1 + np.abs(rng.normal(0, 0.005, n)))
    lows = closes * (1 - np.abs(rng.normal(0, 0.005, n)))
    opens = closes * (1 + rng.normal(0, 0.003, n))
    volumes = rng.integers(500_000, 5_000_000, n).astype(float)

    df = pd.DataFrame({
        "o": opens, "h": highs, "l": lows, "c": closes, "v": volumes,
    }, index=dates)
    return df


# ---------------------------------------------------------------------------
# TestTradeEntry — dataclass structure
# ---------------------------------------------------------------------------

class TestTradeEntry:
    def test_fields(self) -> None:
        e = TradeEntry(
            trade_id=1, symbol="AAPL", entry_date=date(2024, 3, 15),
            direction="long", pnl=250.0,
        )
        assert e.symbol == "AAPL"
        assert e.direction == "long"
        assert e.pnl == 250.0


# ---------------------------------------------------------------------------
# TestLoadTradeEntries — API parsing
# ---------------------------------------------------------------------------

class TestLoadTradeEntries:
    def _raw_trade(self, **overrides) -> dict:
        base = {
            "id": 1,
            "symbol": "AAPL",
            "entryDate": "2024-03-15",
            "directional": "Long",
            "pnl": 250.0,
        }
        base.update(overrides)
        return base

    @patch("finance.utils.swing_backtest.fetch_trades_for_review")
    @patch("finance.utils.swing_backtest.load_config")
    def test_parses_basic_trade(self, mock_config, mock_fetch) -> None:
        mock_config.return_value = MagicMock(tradelog=MagicMock())
        mock_fetch.return_value = [self._raw_trade()]
        entries = load_trade_entries()
        assert len(entries) == 1
        e = entries[0]
        assert e.symbol == "AAPL"
        assert e.entry_date == date(2024, 3, 15)
        assert e.direction == "long"
        assert e.pnl == pytest.approx(250.0)

    @patch("finance.utils.swing_backtest.fetch_trades_for_review")
    @patch("finance.utils.swing_backtest.load_config")
    def test_short_direction_parsed(self, mock_config, mock_fetch) -> None:
        mock_config.return_value = MagicMock(tradelog=MagicMock())
        mock_fetch.return_value = [self._raw_trade(directional="Short")]
        entries = load_trade_entries()
        assert entries[0].direction == "short"

    @patch("finance.utils.swing_backtest.fetch_trades_for_review")
    @patch("finance.utils.swing_backtest.load_config")
    def test_unparseable_date_skipped(self, mock_config, mock_fetch) -> None:
        mock_config.return_value = MagicMock(tradelog=MagicMock())
        mock_fetch.return_value = [self._raw_trade(entryDate="not-a-date")]
        entries = load_trade_entries()
        assert len(entries) == 0

    @patch("finance.utils.swing_backtest.fetch_trades_for_review")
    @patch("finance.utils.swing_backtest.load_config")
    def test_missing_symbol_skipped(self, mock_config, mock_fetch) -> None:
        mock_config.return_value = MagicMock(tradelog=MagicMock())
        mock_fetch.return_value = [self._raw_trade(symbol=None)]
        entries = load_trade_entries()
        assert len(entries) == 0

    @patch("finance.utils.swing_backtest.fetch_trades_for_review")
    @patch("finance.utils.swing_backtest.load_config")
    def test_null_pnl_allowed(self, mock_config, mock_fetch) -> None:
        mock_config.return_value = MagicMock(tradelog=MagicMock())
        mock_fetch.return_value = [self._raw_trade(pnl=None)]
        entries = load_trade_entries()
        assert len(entries) == 1
        assert entries[0].pnl is None


# ---------------------------------------------------------------------------
# TestReconstructCandidate
# ---------------------------------------------------------------------------

class TestReconstructCandidate:
    @patch("finance.utils.swing_backtest._load_parquet")
    def test_returns_none_when_no_parquet(self, mock_load) -> None:
        mock_load.return_value = None
        result = reconstruct_candidate("NOTFOUND", date(2024, 3, 15))
        assert result is None

    @patch("finance.utils.swing_backtest._load_parquet")
    def test_returns_none_when_too_few_bars(self, mock_load) -> None:
        mock_load.return_value = _make_ohlcv(20)  # < 30 bars
        result = reconstruct_candidate("AAPL", date(2023, 1, 25))
        assert result is None

    @patch("finance.utils.swing_backtest._load_parquet")
    def test_returns_enriched_candidate_with_sufficient_data(self, mock_load) -> None:
        df = _make_ohlcv(300)
        mock_load.return_value = df
        entry_date = df.index[-1].date()
        result = reconstruct_candidate("AAPL", entry_date)
        assert result is not None
        assert result.candidate.symbol == "AAPL"
        assert result.candidate.price is not None and result.candidate.price > 0
        assert result.data_available is True

    @patch("finance.utils.swing_backtest._load_parquet")
    def test_technicals_populated(self, mock_load) -> None:
        df = _make_ohlcv(300)
        mock_load.return_value = df
        entry_date = df.index[-1].date()
        result = reconstruct_candidate("AAPL", entry_date)
        assert result is not None
        t = result.technicals
        assert t is not None
        assert t.sma_50 is not None
        assert t.sma_200 is not None
        assert t.atr_14 is not None

    @patch("finance.utils.swing_backtest._load_parquet")
    def test_cutoff_respected(self, mock_load) -> None:
        """Slicing at an earlier date should give different indicators than using all data."""
        df = _make_ohlcv(300)
        mock_load.return_value = df
        early_date = df.index[150].date()
        late_date = df.index[-1].date()
        result_early = reconstruct_candidate("AAPL", early_date)
        result_late = reconstruct_candidate("AAPL", late_date)
        assert result_early is not None and result_late is not None
        # Prices should differ
        assert result_early.candidate.price != result_late.candidate.price


# ---------------------------------------------------------------------------
# TestRunScoringBacktest
# ---------------------------------------------------------------------------

class TestRunScoringBacktest:
    def _make_entries(self) -> list[TradeEntry]:
        return [
            TradeEntry(1, "AAPL", date(2024, 6, 1), "long", 400.0),
            TradeEntry(2, "MSFT", date(2024, 6, 5), "long", -150.0),
            TradeEntry(3, "NVDA", date(2024, 6, 10), "short", 300.0),
        ]

    @patch("finance.utils.swing_backtest.reconstruct_candidate")
    def test_empty_when_no_data(self, mock_rc) -> None:
        mock_rc.return_value = None
        entries = self._make_entries()
        df = run_scoring_backtest(entries)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    @patch("finance.utils.swing_backtest._load_parquet")
    def test_returns_one_row_per_scored_trade(self, mock_load) -> None:
        df_ohlcv = _make_ohlcv(300)
        mock_load.return_value = df_ohlcv
        entries = [TradeEntry(1, "AAPL", df_ohlcv.index[-1].date(), "long", 400.0)]
        result = run_scoring_backtest(entries)
        assert len(result) == 1

    @patch("finance.utils.swing_backtest._load_parquet")
    def test_output_columns_present(self, mock_load) -> None:
        df_ohlcv = _make_ohlcv(300)
        mock_load.return_value = df_ohlcv
        entries = [TradeEntry(1, "AAPL", df_ohlcv.index[-1].date(), "long", 400.0)]
        result = run_scoring_backtest(entries)
        for col in ["trade_id", "symbol", "entry_date", "direction", "pnl", "win",
                    "score_total", "score_d1", "score_d2", "score_d3", "score_d4", "score_d5"]:
            assert col in result.columns, f"Missing column: {col}"

    @patch("finance.utils.swing_backtest._load_parquet")
    def test_win_flag_correct(self, mock_load) -> None:
        df_ohlcv = _make_ohlcv(300)
        mock_load.return_value = df_ohlcv
        entries = [
            TradeEntry(1, "AAPL", df_ohlcv.index[-1].date(), "long", 400.0),
            TradeEntry(2, "AAPL", df_ohlcv.index[-2].date(), "long", -100.0),
        ]
        result = run_scoring_backtest(entries)
        wins = result.set_index("trade_id")["win"]
        if 1 in wins.index:
            assert wins[1] == True  # noqa: E712
        if 2 in wins.index:
            assert wins[2] == False  # noqa: E712

    @patch("finance.utils.swing_backtest._load_parquet")
    def test_score_in_valid_range(self, mock_load) -> None:
        df_ohlcv = _make_ohlcv(300)
        mock_load.return_value = df_ohlcv
        entries = [TradeEntry(1, "AAPL", df_ohlcv.index[-1].date(), "long", 0.0)]
        result = run_scoring_backtest(entries)
        if len(result) > 0:
            assert 0.0 <= result["score_total"].iloc[0] <= 112.0


# ---------------------------------------------------------------------------
# Helpers — synthetic event row
# ---------------------------------------------------------------------------

def _make_event_row(**overrides) -> pd.Series:
    """Synthetic momentum earnings event row at T=0."""
    base = {
        "symbol": "AAPL",
        "date": pd.Timestamp("2024-01-15"),
        "c0": 150.0,
        "1M_chg": 12.0,
        "3M_chg": 20.0,
        "6M_chg": 30.0,
        "12M_chg": 45.0,
        "ma50_dist0": 5.0,
        "ma200_dist0": 15.0,
        "ma50_slope0": 0.25,    # price-units/day
        "ma200_slope0": 0.10,
        "atrp200": 3.5,
        "atrp140": 3.2,
        "rvol200": 1.8,
        "sue": 0.5,
        "spy0": 10.0,
        "spy-21": 7.0,
        "spy-50": 5.0,
        "spy-60": 4.0,
        "cpct10": 8.5,
        "is_earnings": True,
    }
    base.update(overrides)
    return pd.Series(base)


# ---------------------------------------------------------------------------
# TestReconstructCandidateFromEventRow
# ---------------------------------------------------------------------------

class TestReconstructCandidateFromEventRow:
    def test_returns_none_when_c0_missing(self) -> None:
        row = _make_event_row(c0=None)
        assert reconstruct_candidate_from_event_row(row) is None

    def test_returns_none_when_c0_zero(self) -> None:
        row = _make_event_row(c0=0.0)
        assert reconstruct_candidate_from_event_row(row) is None

    def test_returns_enriched_candidate(self) -> None:
        row = _make_event_row()
        ec = reconstruct_candidate_from_event_row(row)
        assert ec is not None
        assert ec.candidate.symbol == "AAPL"
        assert ec.candidate.price == pytest.approx(150.0)
        assert ec.data_available is True

    def test_momentum_fields_populated(self) -> None:
        row = _make_event_row()
        ec = reconstruct_candidate_from_event_row(row)
        assert ec is not None
        c = ec.candidate
        assert c.change_1m_pct == pytest.approx(12.0)
        assert c.change_3m_pct == pytest.approx(20.0)
        assert c.change_6m_pct == pytest.approx(30.0)
        assert c.change_52w_pct == pytest.approx(45.0)

    def test_slope_converted_to_pct_per_day(self) -> None:
        # ma50_slope0=0.25, c0=150 → slope_50d_sma = 0.25 / 1.50 ≈ 0.1667 %/day
        row = _make_event_row(c0=150.0, ma50_slope0=0.25)
        ec = reconstruct_candidate_from_event_row(row)
        assert ec is not None
        assert ec.candidate.slope_50d_sma == pytest.approx(0.25 / (150.0 * 0.01), rel=1e-4)

    def test_sma_back_computed_from_price_and_dist(self) -> None:
        # price=150, ma50_dist0=5.0 → sma_50 = 150 / 1.05 ≈ 142.857
        row = _make_event_row(c0=150.0, ma50_dist0=5.0)
        ec = reconstruct_candidate_from_event_row(row)
        assert ec is not None
        assert ec.technicals is not None
        assert ec.technicals.sma_50 == pytest.approx(150.0 / 1.05, rel=1e-4)

    def test_atr_14_in_price_units(self) -> None:
        # atrp140=3.2, c0=150 → atr_14 = 3.2 * 150 / 100 = 4.8
        row = _make_event_row(c0=150.0, atrp140=3.2)
        ec = reconstruct_candidate_from_event_row(row)
        assert ec is not None
        assert ec.technicals is not None
        assert ec.technicals.atr_14 == pytest.approx(4.8)

    def test_relative_perf_vs_spy_computed(self) -> None:
        # 1M_chg=12, spy0=10, spy-21=7 → perf_1m = 12 - (10 - 7) = 9
        row = _make_event_row(**{"1M_chg": 12.0, "spy0": 10.0, "spy-21": 7.0})
        ec = reconstruct_candidate_from_event_row(row)
        assert ec is not None
        assert ec.candidate.perf_vs_market_1m == pytest.approx(9.0)

    def test_relative_perf_none_when_spy_missing(self) -> None:
        row = _make_event_row(**{"spy0": None})
        ec = reconstruct_candidate_from_event_row(row)
        assert ec is not None
        assert ec.candidate.perf_vs_market_1m is None
        assert ec.candidate.perf_vs_market_3m is None

    def test_sue_used_as_earnings_surprise(self) -> None:
        row = _make_event_row(sue=0.75)
        ec = reconstruct_candidate_from_event_row(row)
        assert ec is not None
        assert ec.candidate.earnings_surprise_pct == pytest.approx(0.75)

    def test_bb_width_unavailable(self) -> None:
        row = _make_event_row()
        ec = reconstruct_candidate_from_event_row(row)
        assert ec is not None
        assert ec.technicals is not None
        assert ec.technicals.bb_width is None
        assert ec.technicals.bb_width_avg_20 is None

    def test_sma_slope_categorical_derived(self) -> None:
        # Positive slope → "rising"; negative slope → "falling"
        row_rising = _make_event_row(c0=150.0, ma50_slope0=0.5)
        row_falling = _make_event_row(c0=150.0, ma50_slope0=-0.5)
        ec_r = reconstruct_candidate_from_event_row(row_rising)
        ec_f = reconstruct_candidate_from_event_row(row_falling)
        assert ec_r is not None and ec_f is not None
        assert ec_r.technicals is not None and ec_f.technicals is not None
        assert ec_r.technicals.sma_50_slope == "rising"
        assert ec_f.technicals.sma_50_slope == "falling"


# ---------------------------------------------------------------------------
# TestRunEventScoring
# ---------------------------------------------------------------------------

class TestRunEventScoring:
    def _make_events_df(self, n: int = 5) -> pd.DataFrame:
        rows = [_make_event_row(symbol=f"SYM{i}", cpct10=float(i * 2 - 4)) for i in range(n)]
        return pd.DataFrame(rows)

    def test_returns_dataframe(self) -> None:
        df = self._make_events_df()
        result = run_event_scoring(df)
        assert isinstance(result, pd.DataFrame)

    def test_output_columns_present(self) -> None:
        df = self._make_events_df()
        result = run_event_scoring(df)
        for col in ["symbol", "date", "score_total", "score_d1", "score_d2",
                    "score_d3", "score_d4", "score_d5", "fwd_return", "win", "spy_regime"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_one_row_per_valid_event(self) -> None:
        df = self._make_events_df(5)
        result = run_event_scoring(df)
        assert len(result) == 5

    def test_skips_rows_with_missing_price(self) -> None:
        df = self._make_events_df(3)
        df.loc[1, "c0"] = None
        result = run_event_scoring(df)
        assert len(result) == 2

    def test_score_in_valid_range(self) -> None:
        df = self._make_events_df()
        result = run_event_scoring(df)
        assert (result["score_total"] >= 0).all()
        assert (result["score_total"] <= 112).all()

    def test_win_flag_correct_for_long(self) -> None:
        rows = [
            _make_event_row(symbol="WIN", cpct10=5.0),
            _make_event_row(symbol="LOSE", cpct10=-3.0),
        ]
        result = run_event_scoring(pd.DataFrame(rows), fwd_col="cpct10", direction="long")
        wins = result.set_index("symbol")["win"]
        assert wins["WIN"] == True   # noqa: E712
        assert wins["LOSE"] == False  # noqa: E712

    def test_spy_regime_go_when_spy_rising(self) -> None:
        # spy-50 < spy0 → "go"
        row = _make_event_row(**{"spy0": 10.0, "spy-50": 5.0})
        result = run_event_scoring(pd.DataFrame([row]))
        assert result["spy_regime"].iloc[0] == "go"

    def test_spy_regime_nogo_when_spy_falling(self) -> None:
        # spy-50 > spy0 → "no-go"
        row = _make_event_row(**{"spy0": 5.0, "spy-50": 10.0})
        result = run_event_scoring(pd.DataFrame([row]))
        assert result["spy_regime"].iloc[0] == "no-go"

    def test_spy_regime_none_when_spy_missing(self) -> None:
        row = _make_event_row(**{"spy0": None, "spy-50": None})
        result = run_event_scoring(pd.DataFrame([row]))
        assert result["spy_regime"].iloc[0] is None

    def test_empty_df_returns_empty_with_schema(self) -> None:
        result = run_event_scoring(pd.DataFrame())
        assert isinstance(result, pd.DataFrame)
        for col in ["symbol", "score_total", "fwd_return", "win", "spy_regime"]:
            assert col in result.columns

    def test_direction_column_in_output(self) -> None:
        df = self._make_events_df(2)
        result = run_event_scoring(df)
        assert "direction" in result.columns

    def test_win_flag_correct_for_short(self) -> None:
        # For shorts: negative fwd return = win
        rows = [
            _make_event_row(symbol="WIN", cpct10=-3.0),
            _make_event_row(symbol="LOSE", cpct10=5.0),
        ]
        result = run_event_scoring(pd.DataFrame(rows), fwd_col="cpct10", direction="short")
        wins = result.set_index("symbol")["win"]
        assert wins["WIN"] == True   # noqa: E712
        assert wins["LOSE"] == False  # noqa: E712

    def test_direction_col_overrides_default(self) -> None:
        # Rows with direction_col="short" should score as short
        rows = [
            _make_event_row(symbol="A", cpct10=-5.0),
            _make_event_row(symbol="B", cpct10=5.0),
        ]
        df = pd.DataFrame(rows)
        df["evt_direction"] = "short"
        result = run_event_scoring(df, direction="long", direction_col="evt_direction")
        assert (result["direction"] == "short").all()
        # short + negative return = win
        wins = result.set_index("symbol")["win"]
        assert wins["A"] == True   # noqa: E712
        assert wins["B"] == False  # noqa: E712

    def test_direction_col_mixed_directions(self) -> None:
        # One long row, one short row — win flag evaluated per direction
        rows = [
            _make_event_row(symbol="LONG_WIN", cpct10=5.0),
            _make_event_row(symbol="SHORT_WIN", cpct10=-5.0),
        ]
        df = pd.DataFrame(rows)
        df["evt_direction"] = ["long", "short"]
        result = run_event_scoring(df, direction_col="evt_direction")
        wins = result.set_index("symbol")["win"]
        assert wins["LONG_WIN"] == True   # noqa: E712
        assert wins["SHORT_WIN"] == True  # noqa: E712
