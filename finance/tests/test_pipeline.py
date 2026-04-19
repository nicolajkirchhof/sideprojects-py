"""Tests for pipeline trading day logic."""
from __future__ import annotations

from datetime import date

from finance.apps.analyst._pipeline import _last_trading_day


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
