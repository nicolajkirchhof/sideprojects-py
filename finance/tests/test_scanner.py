"""Tests for the scanner CSV parser."""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from finance.apps.analyst._config import ScannerConfig
from finance.apps.analyst._models import Candidate
from finance.apps.analyst._scanner import deduplicate, parse_csv, parse_multiple
from finance.apps.assistant._tags import _tag_ttm_fired


@pytest.fixture
def config() -> ScannerConfig:
    return ScannerConfig(
        column_mapping={
            "Symbol": "symbol",
            "Last": "price",
            "Volume": "volume",
            "5 Day % Chg": "change_5d_pct",
            "1 Month % Chg": "change_1m_pct",
            "% Off High": "high_52w_distance_pct",
            "Sector": "sector",
        },
        percent_columns=["change_5d_pct", "change_1m_pct", "high_52w_distance_pct"],
    )


def _write_csv(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(dedent(content).strip())
    return p


class TestParseCsv:
    def test_basic_parsing(self, tmp_path: Path, config: ScannerConfig) -> None:
        csv = _write_csv(tmp_path, "scan.csv", """\
            Symbol,Last,Volume,5 Day % Chg,1 Month % Chg,% Off High,Sector
            AAPL,185.50,50000000,3.5%,8.2%,-5.1%,Technology
            NVDA,920.00,30000000,7.1%,15.3%,-2.0%,Technology
        """)
        result = parse_csv(csv, config)
        assert len(result) == 2
        assert result[0].symbol == "AAPL"
        assert result[0].price == 185.50
        assert result[0].change_5d_pct == 3.5
        assert result[0].sector == "Technology"
        assert result[1].symbol == "NVDA"

    def test_missing_columns_graceful(self, tmp_path: Path, config: ScannerConfig) -> None:
        csv = _write_csv(tmp_path, "minimal.csv", """\
            Symbol,Last
            TSLA,250.00
        """)
        result = parse_csv(csv, config)
        assert len(result) == 1
        assert result[0].symbol == "TSLA"
        assert result[0].price == 250.00
        assert result[0].volume is None
        assert result[0].change_5d_pct is None

    def test_no_symbol_column_returns_empty(self, tmp_path: Path, config: ScannerConfig) -> None:
        csv = _write_csv(tmp_path, "bad.csv", """\
            Ticker,Price
            AAPL,185
        """)
        result = parse_csv(csv, config)
        assert result == []

    def test_empty_csv(self, tmp_path: Path, config: ScannerConfig) -> None:
        csv = _write_csv(tmp_path, "empty.csv", """\
            Symbol,Last,Volume
        """)
        result = parse_csv(csv, config)
        assert result == []

    def test_percent_suffix_stripped(self, tmp_path: Path, config: ScannerConfig) -> None:
        csv = _write_csv(tmp_path, "pct.csv", """\
            Symbol,Last,5 Day % Chg
            SPY,450,2.5%
        """)
        result = parse_csv(csv, config)
        assert result[0].change_5d_pct == 2.5

    def test_symbol_uppercased(self, tmp_path: Path, config: ScannerConfig) -> None:
        csv = _write_csv(tmp_path, "case.csv", """\
            Symbol,Last
            aapl,185
        """)
        result = parse_csv(csv, config)
        assert result[0].symbol == "AAPL"

    def test_nonexistent_file_returns_empty(self, config: ScannerConfig) -> None:
        result = parse_csv(Path("/nonexistent.csv"), config)
        assert result == []

    def test_barchart_footer_row_is_filtered(self, tmp_path: Path, config: ScannerConfig) -> None:
        """Barchart appends a footer row whose 'symbol' contains spaces — must be dropped."""
        csv = _write_csv(tmp_path, "with_footer.csv", """\
            Symbol,Last,Volume
            AAPL,185.50,50000000
            DOWNLOADED FROM BARCHART.COM AS OF 04-22-2026 02:38AM CDT,,
        """)
        result = parse_csv(csv, config)
        assert len(result) == 1
        assert result[0].symbol == "AAPL"


class TestDeduplicate:
    def test_keeps_first_occurrence(self) -> None:
        candidates = [
            Candidate(symbol="AAPL", price=185),
            Candidate(symbol="NVDA", price=920),
            Candidate(symbol="AAPL", price=186),
        ]
        result = deduplicate(candidates)
        assert len(result) == 2
        assert result[0].price == 185  # first AAPL kept


@pytest.fixture
def barchart_config() -> ScannerConfig:
    """Config using the exact Barchart column names defined in BarchartScreeners.md."""
    return ScannerConfig(
        column_mapping={
            "Symbol": "symbol",
            "Latest": "price",
            "%Change": "change_pct",
            "Volume": "volume",
            "5D %Chg": "change_5d_pct",
            "1M %Chg": "change_1m_pct",
            "52W %/High": "high_52w_distance_pct",
            "Trend Seeker Signal": "trend_seeker_signal",
            "Weighted Alpha": "weighted_alpha",
            "Perf vs Market 5D": "perf_vs_market_5d",
            "Perf vs Market 1M": "perf_vs_market_1m",
            "3M % Change from Index": "perf_vs_market_3m",
            "20D RelVol": "rvol_20d",
            "20D ATRP": "atr_pct_20d",
            "20D ADR%": "adr_pct_20d",
            "% 50D MA": "pct_from_50d_sma",
            "Slope of 50D SMA": "slope_50d_sma",
            "Slope of 200D SMA": "slope_200d_sma",
            "TTM Squeeze": "ttm_squeeze",
            "Bollinger Bands Rank": "bb_rank",
            "Gap Up %": "gap_up_pct",
            "%Chg(Pre)": "change_pre_pct",
            "Daily Closing Range": "daily_closing_range",
            "Short Float": "short_float",
            "Short Interest, K": "short_interest_k",
            "Earnings Surprise%": "earnings_surprise_pct",
            "Earnings Surprise% 1-Qtr Ago": "earnings_surprise_q1",
            "Earnings Surprise% 2-Qtrs Ago": "earnings_surprise_q2",
            "Earnings Surprise% 3-Qtrs Ago": "earnings_surprise_q3",
            "5D P/C Vol": "put_call_vol_5d",
            "1M Put/Call Vol": "put_call_vol_1m",
            "IV Pctl": "iv_percentile",
            "5D IV Chg": "iv_chg_5d",
            "1M IV Chg": "iv_chg_1m",
            "1M Total Vol": "options_vol_1m",
            "1M Total OI": "options_oi_1m",
            "Total Volume/OI Ratio": "vol_oi_ratio",
            "Short Interest, K": "short_interest_k",
            "Short Int %Chg": "short_interest_chg_pct",
            "Days to Cover": "days_to_cover",
            "Market Cap, $K": "market_cap_k",
            "Latest Earnings": "latest_earnings",
            "Sector": "sector",
        },
        percent_columns=[
            "change_pct", "change_5d_pct", "change_1m_pct", "high_52w_distance_pct",
            "adr_pct_20d", "atr_pct_20d", "pct_from_50d_sma", "slope_50d_sma",
            "slope_200d_sma", "gap_up_pct", "change_pre_pct", "short_float",
            "earnings_surprise_pct", "earnings_surprise_q1", "earnings_surprise_q2",
            "earnings_surprise_q3", "iv_percentile", "iv_chg_5d", "iv_chg_1m",
            "short_interest_chg_pct",
        ],
    )


class TestBarchartColumnNames:
    """Column names as defined in BarchartScreeners.md §Pipeline Integration."""

    def test_standard_view_trend_and_momentum(
        self, tmp_path: Path, barchart_config: ScannerConfig
    ) -> None:
        csv = _write_csv(tmp_path, "long-universe.csv", """\
            Symbol,Latest,Trend Seeker Signal,Weighted Alpha,Perf vs Market 5D,Perf vs Market 1M,3M % Change from Index
            AAPL,185.50,Buy,18.5,3.2,5.1,12.3
        """)
        result = parse_csv(csv, barchart_config)
        assert len(result) == 1
        c = result[0]
        assert c.trend_seeker_signal == "Buy"
        assert c.weighted_alpha == 18.5
        assert c.perf_vs_market_5d == 3.2
        assert c.perf_vs_market_1m == 5.1
        assert c.perf_vs_market_3m == 12.3

    def test_standard_view_slope_and_bb_rank(
        self, tmp_path: Path, barchart_config: ScannerConfig
    ) -> None:
        csv = _write_csv(tmp_path, "long-universe.csv", """\
            Symbol,Latest,Slope of 50D SMA,Slope of 200D SMA,Bollinger Bands Rank,20D ADR%
            NVDA,850.00,0.52%,-0.10%,74.5,3.8%
        """)
        result = parse_csv(csv, barchart_config)
        c = result[0]
        assert c.slope_50d_sma == 0.52
        assert c.slope_200d_sma == -0.10
        assert c.bb_rank == 74.5
        assert c.adr_pct_20d == 3.8

    def test_pead_ep_view_earnings_columns(
        self, tmp_path: Path, barchart_config: ScannerConfig
    ) -> None:
        csv = _write_csv(tmp_path, "pead-scanner.csv", """\
            Symbol,Latest,Gap Up %,%Chg(Pre),Earnings Surprise%,Earnings Surprise% 1-Qtr Ago,Earnings Surprise% 2-Qtrs Ago,Earnings Surprise% 3-Qtrs Ago
            META,510.00,5.2%,4.8%,12.5%,8.3%,6.1%,-2.0%
        """)
        result = parse_csv(csv, barchart_config)
        c = result[0]
        assert c.gap_up_pct == 5.2
        assert c.change_pre_pct == 4.8
        assert c.earnings_surprise_pct == 12.5
        assert c.earnings_surprise_q1 == 8.3
        assert c.earnings_surprise_q2 == 6.1
        assert c.earnings_surprise_q3 == -2.0

    def test_intraday_view_daily_closing_range(
        self, tmp_path: Path, barchart_config: ScannerConfig
    ) -> None:
        csv = _write_csv(tmp_path, "intraday.csv", """\
            Symbol,Latest,Daily Closing Range
            TSLA,250.00,78.5
        """)
        result = parse_csv(csv, barchart_config)
        assert result[0].daily_closing_range == 78.5

    def test_options_flow_view_vol_and_oi(
        self, tmp_path: Path, barchart_config: ScannerConfig
    ) -> None:
        csv = _write_csv(tmp_path, "high-put-ratio.csv", (
            'Symbol,Latest,1M Total Vol,1M Total OI,Total Volume/OI Ratio,'
            '"Short Interest, K",Days to Cover,"Market Cap, $K"\n'
            'AMZN,190.00,250000,180000,1.39,45000,3.2,1950000\n'
        ))
        result = parse_csv(csv, barchart_config)
        c = result[0]
        assert c.options_vol_1m == 250000
        assert c.options_oi_1m == 180000
        assert c.vol_oi_ratio == 1.39
        assert c.short_interest_k == 45000
        assert c.days_to_cover == 3.2
        assert c.market_cap_k == 1950000


class TestTtmFiredBbRankFallback:
    """_ttm_fired should use bb_rank when bb_pct is unavailable."""

    def test_fired_via_bb_rank(self) -> None:
        c = Candidate(
            symbol="AAPL",
            ttm_squeeze="Long",
            bb_pct=None,
            bb_rank=82.0,
            rvol_20d=1.5,
            atr_pct_20d=4.0,
        )
        assert _tag_ttm_fired(c) is True

    def test_not_fired_when_bb_rank_low(self) -> None:
        c = Candidate(
            symbol="AAPL",
            ttm_squeeze="Long",
            bb_pct=None,
            bb_rank=60.0,
            rvol_20d=1.5,
            atr_pct_20d=4.0,
        )
        assert _tag_ttm_fired(c) is False

    def test_bb_pct_takes_priority_over_bb_rank(self) -> None:
        """bb_pct wins when both are present — bb_pct > 80 fires regardless of bb_rank."""
        c = Candidate(
            symbol="AAPL",
            ttm_squeeze="Long",
            bb_pct=85.0,
            bb_rank=50.0,  # low rank but bb_pct wins
            rvol_20d=1.5,
            atr_pct_20d=4.0,
        )
        assert _tag_ttm_fired(c) is True


class TestParseMultiple:
    def test_deduplicates_across_files(self, tmp_path: Path, config: ScannerConfig) -> None:
        csv1 = _write_csv(tmp_path, "scan1.csv", """\
            Symbol,Last
            AAPL,185
            NVDA,920
        """)
        csv2 = _write_csv(tmp_path, "scan2.csv", """\
            Symbol,Last
            NVDA,921
            TSLA,250
        """)
        result = parse_multiple([csv1, csv2], config)
        symbols = [c.symbol for c in result]
        assert symbols == ["AAPL", "NVDA", "TSLA"]
        # NVDA from first file (price 920) should be kept
        assert result[1].price == 920
