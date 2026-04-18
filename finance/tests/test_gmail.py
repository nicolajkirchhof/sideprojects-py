"""Tests for Gmail email parsing, classification, and state management."""
from __future__ import annotations

import base64
import json
from datetime import date, datetime
from pathlib import Path

import pytest

from finance.apps.analyst._gmail import (
    EmailMessage,
    _build_query,
    _classify,
    _extract_body,
    _load_last_run_date,
    _parse_date,
    _save_last_run_date,
)


class TestBuildQuery:
    def test_label_only(self) -> None:
        assert _build_query("market", None) == "label:market"

    def test_label_with_date(self) -> None:
        result = _build_query("market", date(2026, 4, 17))
        assert result == "label:market after:2026-04-17"


class TestClassify:
    def test_csv_attachment_is_scanner(self) -> None:
        email = EmailMessage(
            message_id="1", sender="alerts@barchart.com",
            subject="Screener", date=datetime.now(),
            body_text="", attachments=[Path("scan.csv")],
        )
        assert _classify(email, []) == "scanner"

    def test_matching_sender_rule(self) -> None:
        email = EmailMessage(
            message_id="2", sender="newsletter@briefing.com",
            subject="Morning Brief", date=datetime.now(),
            body_text="Market summary...",
        )
        rules = [{"sender": "@briefing.com", "category": "market_commentary"}]
        assert _classify(email, rules) == "market_commentary"

    def test_body_text_defaults_to_commentary(self) -> None:
        email = EmailMessage(
            message_id="3", sender="unknown@example.com",
            subject="Daily Update", date=datetime.now(),
            body_text="Some market content here",
        )
        assert _classify(email, []) == "market_commentary"

    def test_empty_email_is_unknown(self) -> None:
        email = EmailMessage(
            message_id="4", sender="unknown@example.com",
            subject="Empty", date=datetime.now(),
            body_text="",
        )
        assert _classify(email, []) == "unknown"

    def test_attachment_takes_priority_over_rules(self) -> None:
        email = EmailMessage(
            message_id="5", sender="alerts@barchart.com",
            subject="Screener", date=datetime.now(),
            body_text="Here is your report",
            attachments=[Path("screener.csv")],
        )
        rules = [{"sender": "@barchart.com", "category": "market_commentary"}]
        # Attachment presence should override sender rule
        assert _classify(email, rules) == "scanner"


class TestExtractBody:
    def test_plain_text_body(self) -> None:
        payload = {
            "mimeType": "text/plain",
            "body": {"data": base64.urlsafe_b64encode(b"Hello world").decode()},
        }
        assert _extract_body(payload) == "Hello world"

    def test_multipart_extracts_plain(self) -> None:
        payload = {
            "mimeType": "multipart/alternative",
            "parts": [
                {"mimeType": "text/html", "body": {"data": base64.urlsafe_b64encode(b"<p>Hi</p>").decode()}},
                {"mimeType": "text/plain", "body": {"data": base64.urlsafe_b64encode(b"Hi plain").decode()}},
            ],
        }
        assert _extract_body(payload) == "Hi plain"

    def test_empty_payload(self) -> None:
        assert _extract_body({"mimeType": "text/plain", "body": {}}) == ""


class TestParseDate:
    def test_standard_format(self) -> None:
        result = _parse_date("Fri, 17 Apr 2026 09:30:00 +0000")
        assert result.year == 2026
        assert result.month == 4
        assert result.day == 17

    def test_with_timezone_name(self) -> None:
        result = _parse_date("Fri, 17 Apr 2026 09:30:00 +0000 (UTC)")
        assert result.year == 2026

    def test_unparseable_returns_now(self) -> None:
        result = _parse_date("not a date")
        assert isinstance(result, datetime)


class TestStateFile:
    def test_save_and_load(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        state_file = tmp_path / "_state.json"
        monkeypatch.setattr("finance.apps.analyst._gmail._STATE_FILE", state_file)

        _save_last_run_date(date(2026, 4, 17))
        result = _load_last_run_date()
        assert result == date(2026, 4, 17)

    def test_load_nonexistent_returns_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        state_file = tmp_path / "nope.json"
        monkeypatch.setattr("finance.apps.analyst._gmail._STATE_FILE", state_file)
        assert _load_last_run_date() is None

    def test_load_corrupt_returns_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        state_file = tmp_path / "_state.json"
        state_file.write_text("not json")
        monkeypatch.setattr("finance.apps.analyst._gmail._STATE_FILE", state_file)
        assert _load_last_run_date() is None
