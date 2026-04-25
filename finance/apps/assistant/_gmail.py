"""
finance.apps.assistant._gmail
================================
Gmail API client for the Trading Assistant.

Provides two fetch functions:
  fetch_screener_csvs  — downloads Barchart screener CSV attachments
  fetch_email_bodies   — returns text bodies of market commentary emails

Both authenticate via OAuth2 using credentials in _credentials/.

Error policy: any failure raises and propagates to halt the pipeline.
"""
from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from finance.apps.analyst._config import GmailConfig

log = logging.getLogger(__name__)

_CREDENTIALS_DIR = Path(__file__).parent / "_credentials"


# ---------------------------------------------------------------------------
# EmailBody — lightweight email struct for Claude market summary
# ---------------------------------------------------------------------------

@dataclass
class EmailBody:
    """Text body of a single email, used as input to summarize_market()."""
    subject: str
    sender: str
    date: str       # ISO date string, e.g. "2026-04-23"
    body_text: str
_CLIENT_SECRET_FILE = _CREDENTIALS_DIR / "client_secret.json"
_TOKEN_FILE = _CREDENTIALS_DIR / "gmail_token.json"
_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def fetch_screener_csvs(
    config: GmailConfig,
    staging_dir: Path,
    trade_date: date,
) -> list[Path]:
    """Fetch Barchart screener CSV attachments from Gmail for *trade_date*.

    Searches the configured label for emails received on or after
    *trade_date*, downloads every CSV attachment to *staging_dir*, and
    returns the list of written paths.

    Parameters
    ----------
    config:
        Gmail configuration (label name).
    staging_dir:
        Directory to write downloaded CSV files. Created if absent.
    trade_date:
        Session date — emails before this date are ignored.

    Returns
    -------
    list[Path]
        Paths of downloaded CSV files (may be empty if no emails found).

    Raises
    ------
    RuntimeError
        If authentication fails or the Gmail API returns an error.
    """
    service = _get_gmail_service()
    staging_dir.mkdir(parents=True, exist_ok=True)

    query = _build_query(config.label, trade_date)
    log.info("Fetching screener emails: label=%s, since=%s", config.label, trade_date)

    messages = _list_messages(service, query)
    if not messages:
        log.info("No screener emails found for %s", trade_date)
        return []

    log.info("Found %d email(s) — downloading CSV attachments", len(messages))
    paths: list[Path] = []
    for msg_meta in messages:
        msg = _get_message(service, msg_meta["id"])
        downloaded = _download_csv_attachments(service, msg["id"], msg["payload"], staging_dir)
        paths.extend(downloaded)
        log.info("  msg=%s → %d CSV(s)", msg_meta["id"], len(downloaded))

    log.info("Downloaded %d CSV file(s) total", len(paths))
    return paths


def last_trading_day(d: date) -> date:
    """Return the previous weekday before *d* (Mon→Fri, Tue-Fri→yesterday).

    No holiday handling — weekends only.
    """
    from datetime import timedelta
    prev = d - timedelta(days=1)
    while prev.weekday() >= 5:   # 5=Sat, 6=Sun
        prev -= timedelta(days=1)
    return prev


def fetch_email_bodies(
    config: GmailConfig,
    trade_date: date,
    *,
    max_emails: int = 10,
) -> list[EmailBody]:
    """Fetch text bodies of market commentary emails for *trade_date*.

    Searches the configured label for emails from the last two days
    (screener emails often arrive the evening before), extracts plain-text
    bodies, and returns up to *max_emails* results.

    Unlike fetch_screener_csvs, this function does NOT filter by
    ``has:attachment`` so it picks up text-only market newsletters too.

    Parameters
    ----------
    config:
        Gmail configuration (label name).
    trade_date:
        Session date — emails before ``trade_date - 1d`` are excluded.
    max_emails:
        Cap on the number of emails returned (avoids huge prompts).

    Returns
    -------
    list[EmailBody]
        Text bodies, newest first.  May be empty if no emails found.

    Raises
    ------
    RuntimeError
        If authentication fails or the Gmail API returns an error.
    """
    service = _get_gmail_service()

    after = last_trading_day(trade_date).strftime("%Y/%m/%d")
    query = f"label:{config.label} after:{after}"
    log.info("Fetching email bodies: label=%s, since=%s", config.label, trade_date)

    messages = _list_messages(service, query)
    if not messages:
        log.info("No market emails found for %s", trade_date)
        return []

    bodies: list[EmailBody] = []
    for msg_meta in messages[:max_emails]:
        msg = _get_message(service, msg_meta["id"])
        body = _extract_text_body(msg["payload"])
        if not body.strip():
            continue
        headers = {h["name"]: h["value"] for h in msg["payload"].get("headers", [])}
        bodies.append(EmailBody(
            subject=headers.get("Subject", "(no subject)"),
            sender=headers.get("From", ""),
            date=headers.get("Date", ""),
            body_text=body[:3000],  # cap per email to keep prompt size manageable
        ))

    log.info("Fetched %d email body(ies)", len(bodies))
    return bodies


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_gmail_service():
    """Authenticate and return a Gmail API service object.

    Raises RuntimeError if dependencies are missing or authentication fails.
    """
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError(
            "Gmail dependencies not installed. "
            "Run: uv add google-auth-oauthlib google-api-python-client"
        ) from exc

    if not _CLIENT_SECRET_FILE.exists():
        raise RuntimeError(
            f"Gmail client secret not found at {_CLIENT_SECRET_FILE}. "
            "Place your OAuth2 client_secret.json there."
        )

    creds = None
    if _TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(_TOKEN_FILE), _SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            log.info("Refreshing Gmail token...")
            creds.refresh(Request())
        else:
            log.info("Opening browser for Gmail authentication...")
            flow = InstalledAppFlow.from_client_secrets_file(str(_CLIENT_SECRET_FILE), _SCOPES)
            creds = flow.run_local_server(port=0)
        _TOKEN_FILE.write_text(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def _build_query(label: str, since: date) -> str:
    """Build a Gmail search query string.

    Screener emails arrive the evening before the session date, so search
    from one day prior to ensure they are included.
    """
    from datetime import timedelta
    after = (since - timedelta(days=1)).strftime("%Y/%m/%d")
    return f"label:{label} after:{after} has:attachment"


def _list_messages(service, query: str) -> list[dict]:
    """Return message metadata dicts matching *query*."""
    response = service.users().messages().list(
        userId="me", q=query, maxResults=50,
    ).execute()
    return response.get("messages", [])


def _get_message(service, message_id: str) -> dict:
    """Fetch a full message payload by ID."""
    return service.users().messages().get(
        userId="me", id=message_id, format="full",
    ).execute()


def _download_csv_attachments(
    service,
    message_id: str,
    payload: dict,
    staging_dir: Path,
) -> list[Path]:
    """Download all CSV attachments from *payload* to *staging_dir*."""
    downloaded: list[Path] = []
    for part in payload.get("parts", []):
        filename = part.get("filename", "")
        if not filename or not filename.lower().endswith(".csv"):
            continue
        attachment_id = part.get("body", {}).get("attachmentId")
        if not attachment_id:
            continue

        att = service.users().messages().attachments().get(
            userId="me", messageId=message_id, id=attachment_id,
        ).execute()
        data = base64.urlsafe_b64decode(att["data"])

        out_path = staging_dir / filename
        out_path.write_bytes(data)
        downloaded.append(out_path)
        log.debug("Downloaded: %s (%d bytes)", filename, len(data))

    return downloaded


def _extract_text_body(payload: dict) -> str:
    """Recursively extract the plain-text body from a Gmail message payload.

    Walks the MIME part tree and returns the first ``text/plain`` part found.
    Falls back to stripping HTML tags from ``text/html`` if no plain text exists.
    """
    mime_type = payload.get("mimeType", "")

    if mime_type == "text/plain":
        data = payload.get("body", {}).get("data", "")
        if data:
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")

    if mime_type == "text/html":
        data = payload.get("body", {}).get("data", "")
        if data:
            html = base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
            # Strip tags with a simple regex — good enough for newsletter content
            import re
            return re.sub(r"<[^>]+>", " ", html)

    # Recurse into parts
    for part in payload.get("parts", []):
        text = _extract_text_body(part)
        if text.strip():
            return text

    return ""
