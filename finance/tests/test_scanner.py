"""Tests for the scanner CSV parser."""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from finance.apps.analyst._config import ScannerConfig
from finance.apps.analyst._models import Candidate
from finance.apps.analyst._scanner import deduplicate, parse_csv, parse_multiple


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
