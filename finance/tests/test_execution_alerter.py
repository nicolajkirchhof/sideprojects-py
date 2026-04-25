from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from finance.execution.alerter import TelegramAlerter


def _make_mock_response(status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = "error detail"
    return resp


def _patch_client(post_return: MagicMock | None = None, raise_exc: Exception | None = None):
    """Return a context manager that patches httpx.AsyncClient on TelegramAlerter's instance."""
    mock_post = AsyncMock(return_value=post_return) if post_return is not None else AsyncMock(side_effect=raise_exc)
    mock_client = MagicMock()
    mock_client.post = mock_post
    return mock_client, mock_post


class TestTelegramAlerterMessageFormat:
    @pytest.mark.anyio
    async def test_message_includes_mode_prefix(self):
        alerter = TelegramAlerter(bot_token="tok", chat_id="123", mode="paper")
        mock_client, mock_post = _patch_client(post_return=_make_mock_response())
        alerter._client = mock_client

        await alerter.send("engine started")

        payload = mock_post.call_args[1]["json"]
        assert payload["text"].startswith("[PAPER]")

    @pytest.mark.anyio
    async def test_live_mode_prefix(self):
        alerter = TelegramAlerter(bot_token="tok", chat_id="123", mode="live")
        mock_client, mock_post = _patch_client(post_return=_make_mock_response())
        alerter._client = mock_client

        await alerter.send("engine started")

        payload = mock_post.call_args[1]["json"]
        assert payload["text"].startswith("[LIVE]")

    @pytest.mark.anyio
    async def test_known_emoji_key_prepended(self):
        alerter = TelegramAlerter(bot_token="tok", chat_id="123", mode="paper")
        mock_client, mock_post = _patch_client(post_return=_make_mock_response())
        alerter._client = mock_client

        await alerter.send("position filled", emoji_key="fill")

        payload = mock_post.call_args[1]["json"]
        assert "\U0001f7e2" in payload["text"]

    @pytest.mark.anyio
    async def test_unknown_emoji_key_does_not_crash(self):
        alerter = TelegramAlerter(bot_token="tok", chat_id="123", mode="paper")
        mock_client, mock_post = _patch_client(post_return=_make_mock_response())
        alerter._client = mock_client

        await alerter.send("test", emoji_key="nonexistent_key")

        mock_post.assert_called_once()

    @pytest.mark.anyio
    async def test_chat_id_sent_in_payload(self):
        alerter = TelegramAlerter(bot_token="tok", chat_id="456", mode="paper")
        mock_client, mock_post = _patch_client(post_return=_make_mock_response())
        alerter._client = mock_client

        await alerter.send("test")

        payload = mock_post.call_args[1]["json"]
        assert payload["chat_id"] == "456"


class TestTelegramAlerterGracefulDegradation:
    @pytest.mark.anyio
    async def test_http_error_status_does_not_raise(self):
        alerter = TelegramAlerter(bot_token="tok", chat_id="123", mode="paper")
        mock_client, _ = _patch_client(post_return=_make_mock_response(status_code=400))
        alerter._client = mock_client

        await alerter.send("test")  # must not raise

    @pytest.mark.anyio
    async def test_network_exception_does_not_raise(self):
        alerter = TelegramAlerter(bot_token="tok", chat_id="123", mode="paper")
        mock_client, _ = _patch_client(raise_exc=ConnectionError("timeout"))
        alerter._client = mock_client

        await alerter.send("test")  # must not raise

    @pytest.mark.anyio
    async def test_http_error_logged_as_warning(self, caplog):
        alerter = TelegramAlerter(bot_token="tok", chat_id="123", mode="paper")
        mock_client, _ = _patch_client(post_return=_make_mock_response(status_code=500))
        alerter._client = mock_client

        with caplog.at_level(logging.WARNING, logger="finance.execution.alerter"):
            await alerter.send("test")

        assert any("500" in r.message for r in caplog.records)

    @pytest.mark.anyio
    async def test_network_exception_logged_as_warning(self, caplog):
        alerter = TelegramAlerter(bot_token="tok", chat_id="123", mode="paper")
        mock_client, _ = _patch_client(raise_exc=ConnectionError("network down"))
        alerter._client = mock_client

        with caplog.at_level(logging.WARNING, logger="finance.execution.alerter"):
            await alerter.send("test")

        assert any("network down" in r.message for r in caplog.records)
