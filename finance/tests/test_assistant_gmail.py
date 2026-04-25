"""
Tests for finance.apps.assistant._gmail — TA Gmail screener fetch.

All tests mock the Gmail API service; no real network calls.
"""
from __future__ import annotations

import base64
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_service(messages: list[dict], attachments: dict[str, bytes]) -> MagicMock:
    """
    Build a minimal mock Gmail service.

    Parameters
    ----------
    messages:
        List of message metadata dicts (each must have an "id" key).
    attachments:
        Mapping of attachmentId -> raw bytes content.
    """
    service = MagicMock()

    # users().messages().list().execute()
    service.users().messages().list().execute.return_value = {"messages": messages}

    def _get_message_execute(message_id):
        mock = MagicMock()
        # Build a full message payload with one CSV attachment
        att_id = f"att-{message_id}"
        mock.execute.return_value = {
            "id": message_id,
            "payload": {
                "headers": [
                    {"name": "From", "value": "noreply@barchart.com"},
                    {"name": "Subject", "value": "stocks-screener-long-universe-04-23-2026.csv"},
                    {"name": "Date", "value": "Wed, 23 Apr 2026 18:00:00 +0000"},
                ],
                "parts": [
                    {
                        "mimeType": "text/plain",
                        "body": {"data": base64.urlsafe_b64encode(b"body text").decode()},
                    },
                    {
                        "mimeType": "text/csv",
                        "filename": f"stocks-screener-long-universe-04-23-2026.csv",
                        "body": {"attachmentId": att_id},
                    },
                ],
            },
        }
        return mock

    service.users().messages().get.side_effect = (
        lambda userId, id, format: _get_message_execute(id)
    )

    def _get_attachment_execute(att_id):
        mock = MagicMock()
        raw = attachments.get(att_id, b"col1,col2\nval1,val2")
        mock.execute.return_value = {"data": base64.urlsafe_b64encode(raw).decode()}
        return mock

    service.users().messages().attachments().get.side_effect = (
        lambda userId, messageId, id: _get_attachment_execute(id)
    )

    return service


# ---------------------------------------------------------------------------
# fetch_screener_csvs tests
# ---------------------------------------------------------------------------


def test_fetch_returns_empty_when_no_messages(tmp_path):
    """No emails → empty list, no files written."""
    from finance.apps.analyst._config import GmailConfig
    from finance.apps.assistant._gmail import fetch_screener_csvs

    service = _make_service(messages=[], attachments={})
    with patch("finance.apps.assistant._gmail._get_gmail_service", return_value=service):
        result = fetch_screener_csvs(
            GmailConfig(label="TradeAnalyst"),
            staging_dir=tmp_path,
            trade_date=date(2026, 4, 23),
        )

    assert result == []
    assert list(tmp_path.iterdir()) == []


def test_fetch_downloads_csv_attachment(tmp_path):
    """One email with one CSV attachment → file written, path returned."""
    from finance.apps.analyst._config import GmailConfig
    from finance.apps.assistant._gmail import fetch_screener_csvs

    csv_bytes = b"Symbol,Latest\nAAPL,185.0\n"
    service = _make_service(
        messages=[{"id": "msg1"}],
        attachments={"att-msg1": csv_bytes},
    )
    with patch("finance.apps.assistant._gmail._get_gmail_service", return_value=service):
        result = fetch_screener_csvs(
            GmailConfig(label="TradeAnalyst"),
            staging_dir=tmp_path,
            trade_date=date(2026, 4, 23),
        )

    assert len(result) == 1
    assert result[0].suffix == ".csv"
    assert result[0].read_bytes() == csv_bytes


def test_fetch_creates_staging_dir(tmp_path):
    """fetch_screener_csvs creates staging_dir if it does not exist."""
    from finance.apps.analyst._config import GmailConfig
    from finance.apps.assistant._gmail import fetch_screener_csvs

    staging = tmp_path / "sub" / "deep"
    service = _make_service(messages=[], attachments={})
    with patch("finance.apps.assistant._gmail._get_gmail_service", return_value=service):
        fetch_screener_csvs(
            GmailConfig(label="TradeAnalyst"),
            staging_dir=staging,
            trade_date=date(2026, 4, 23),
        )
    assert staging.exists()


def test_fetch_raises_when_service_unavailable(tmp_path):
    """If _get_gmail_service raises, the error propagates (halts pipeline)."""
    from finance.apps.analyst._config import GmailConfig
    from finance.apps.assistant._gmail import fetch_screener_csvs

    with patch(
        "finance.apps.assistant._gmail._get_gmail_service",
        side_effect=RuntimeError("OAuth failed"),
    ):
        with pytest.raises(RuntimeError, match="OAuth failed"):
            fetch_screener_csvs(
                GmailConfig(label="TradeAnalyst"),
                staging_dir=tmp_path,
                trade_date=date(2026, 4, 23),
            )


def test_fetch_uses_label_in_query(tmp_path):
    """The Gmail list call must include the configured label."""
    from finance.apps.analyst._config import GmailConfig
    from finance.apps.assistant._gmail import fetch_screener_csvs

    service = _make_service(messages=[], attachments={})
    with patch("finance.apps.assistant._gmail._get_gmail_service", return_value=service):
        fetch_screener_csvs(
            GmailConfig(label="MyLabel"),
            staging_dir=tmp_path,
            trade_date=date(2026, 4, 23),
        )

    call_kwargs = service.users().messages().list.call_args
    assert "MyLabel" in call_kwargs.kwargs.get("q", "")


def test_fetch_uses_previous_day_in_query(tmp_path):
    """Query must use trade_date - 1 day so evening-before emails are included."""
    from finance.apps.analyst._config import GmailConfig
    from finance.apps.assistant._gmail import fetch_screener_csvs

    service = _make_service(messages=[], attachments={})
    with patch("finance.apps.assistant._gmail._get_gmail_service", return_value=service):
        fetch_screener_csvs(
            GmailConfig(label="TradeAnalyst"),
            staging_dir=tmp_path,
            trade_date=date(2026, 4, 23),
        )

    call_kwargs = service.users().messages().list.call_args
    q = call_kwargs.kwargs.get("q", "")
    # Screener emails arrive on Apr 22 evening; query must use after:2026/04/22
    assert "after:2026/04/22" in q
