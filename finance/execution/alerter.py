from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"

_EMOJI = {
    "fill": "\U0001f7e2",        # green circle
    "halted": "\U0001f534",      # red circle
    "error": "\u26a0\ufe0f",     # warning
    "startup": "\u25b6\ufe0f",   # play
    "shutdown": "\u23f9\ufe0f",  # stop
    "missed": "\u23f0",          # alarm clock
    "blocked": "\u26d4",         # no entry
}


class TelegramAlerter:
    """Sends alert messages to a Telegram chat via the Bot API.

    Failures are logged as warnings and never propagate — the engine must
    continue running even if Telegram is unreachable.

    Call ``aclose()`` on engine shutdown to release the underlying HTTP connection pool.
    """

    def __init__(self, bot_token: str, chat_id: str, mode: str = "paper") -> None:
        self._chat_id = chat_id
        self._mode = mode.upper()
        self._url = _TELEGRAM_API.format(token=bot_token)
        self._client = httpx.AsyncClient(timeout=10.0)

    async def send(self, message: str, emoji_key: str = "") -> None:
        """Send a message to the configured chat.

        Args:
            message: Human-readable alert text.
            emoji_key: Optional key into the emoji map for a leading emoji.
        """
        prefix = _EMOJI.get(emoji_key, "")
        text = f"[{self._mode}] {prefix} {message}".strip()
        try:
            resp = await self._client.post(
                self._url,
                json={"chat_id": self._chat_id, "text": text},
            )
            if resp.status_code != 200:
                logger.warning("Telegram send failed (HTTP %s): %s", resp.status_code, resp.text)
        except Exception as exc:
            logger.warning("Telegram send error (suppressed): %s", exc)

    async def aclose(self) -> None:
        """Release the underlying HTTP connection pool."""
        await self._client.aclose()
