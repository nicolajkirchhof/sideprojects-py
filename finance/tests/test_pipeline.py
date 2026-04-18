"""Tests for pipeline CSV resolution and trading day logic."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from textwrap import dedent

import pytest

from finance.apps.analyst._pipeline import _last_trading_day, _resolve_csv_paths
from finance.apps.analyst._config import AnalystConfig, ScannerConfig


class TestLastTradingDay:
    def test_weekday_returns_same_day(self) -> None:
        # Wednesday
        assert _last_trading_day(date(2026, 4, 15)) == date(2026, 4, 15)

    def test_friday_returns_friday(self) -> None:
        assert _last_trading_day(date(2026, 4, 17)) == date(2026, 4, 17)

    def test_saturday_returns_friday(self) -> None:
        assert _last_trading_day(date(2026, 4, 18)) == date(2026, 4, 17)

    def test_sunday_returns_friday(self) -> None:
        assert _last_trading_day(date(2026, 4, 19)) == date(2026, 4, 17)

    def test_monday_returns_monday(self) -> None:
        assert _last_trading_day(date(2026, 4, 20)) == date(2026, 4, 20)


class TestResolveCsvPaths:
    def test_explicit_paths_override(self, tmp_path: Path) -> None:
        csv = tmp_path / "custom.csv"
        csv.write_text("Symbol\nAAPL")
        config = AnalystConfig()
        result = _resolve_csv_paths([str(csv)], config)
        assert result == [csv]

    def test_explicit_nonexistent_filtered(self) -> None:
        config = AnalystConfig()
        result = _resolve_csv_paths(["/nonexistent.csv"], config)
        assert result == []

    def test_finds_screener_by_date(self, tmp_path: Path) -> None:
        # Create a screener CSV for Friday April 17 2026
        csv = tmp_path / "screener-momentum_04-17-2026.csv"
        csv.write_text("Symbol\nAAPL")

        config = AnalystConfig(
            scanner=ScannerConfig(csv_directory=str(tmp_path)),
        )
        # Saturday → should find Friday's files
        from unittest.mock import patch
        with patch("finance.apps.analyst._pipeline._last_trading_day", return_value=date(2026, 4, 17)):
            result = _resolve_csv_paths(None, config)
        assert result == [csv]

    def test_finds_multiple_screeners_same_date(self, tmp_path: Path) -> None:
        csv1 = tmp_path / "screener-5d-chg_04-17-2026.csv"
        csv2 = tmp_path / "screener-momentum_04-17-2026.csv"
        csv1.write_text("Symbol\nAAPL")
        csv2.write_text("Symbol\nNVDA")

        config = AnalystConfig(
            scanner=ScannerConfig(csv_directory=str(tmp_path)),
        )
        from unittest.mock import patch
        with patch("finance.apps.analyst._pipeline._last_trading_day", return_value=date(2026, 4, 17)):
            result = _resolve_csv_paths(None, config)
        assert len(result) == 2

    def test_ignores_other_dates(self, tmp_path: Path) -> None:
        # CSV from a different date
        csv = tmp_path / "screener-momentum_04-16-2026.csv"
        csv.write_text("Symbol\nAAPL")

        config = AnalystConfig(
            scanner=ScannerConfig(csv_directory=str(tmp_path)),
        )
        from unittest.mock import patch
        with patch("finance.apps.analyst._pipeline._last_trading_day", return_value=date(2026, 4, 17)):
            result = _resolve_csv_paths(None, config)
        assert result == []
