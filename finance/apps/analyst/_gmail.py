"""
finance.apps.analyst._gmail
==============================
Gmail API client for fetching market emails and extracting attachments.

Authenticates via OAuth2, fetches emails by label, classifies them,
and downloads CSV attachments to a staging directory.
"""
from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from finance.apps.analyst._config import GmailConfig

log = logging.getLogger(__name__)

_CREDENTIALS_DIR = Path(__file__).parent / "_credentials"
_CLIENT_SECRET_FILE = _CREDENTIALS_DIR / "client_secret.json"
_TOKEN_FILE = _CREDENTIALS_DIR / "gmail_token.json"
_STATE_FILE = Path(__file__).parent / "_state.json"
_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


@dataclass
class EmailMessage:
    """A fetched email with extracted content."""
    message_id: str
    sender: str
    subject: str
    date: datetime
    body_text: str
    attachments: list[Path] = field(default_factory=list)
    category: str = "unknown"  # scanner | market_commentary | research | unknown


def fetch_and_classify(
    config: GmailConfig,
    staging_dir: Path,
    since_date: date | None = None,
) -> list[EmailMessage]:
    """Fetch emails from Gmail label, download attachments, classify.

    Args:
        config: Gmail configuration (label, classification rules).
        staging_dir: Directory to save CSV attachments.
        since_date: Only fetch emails after this date. If None, uses last run date from state file.

    Returns:
        List of EmailMessage objects with attachments downloaded and category assigned.
    """
    service = _get_gmail_service()
    if service is None:
        return []

    since = since_date or _load_last_run_date()
    query = _build_query(config.label, since)

    log.info("Fetching emails: label=%s, since=%s", config.label, since)
    messages = _list_messages(service, query)
    if not messages:
        log.info("No new emails found")
        _save_last_run_date(date.today())
        return []

    log.info("Found %d email(s) to process", len(messages))
    staging_dir.mkdir(parents=True, exist_ok=True)

    results: list[EmailMessage] = []
    for msg_meta in messages:
        msg = _get_message(service, msg_meta["id"])
        if msg is None:
            continue

        email = _parse_message(msg, service, staging_dir)
        email.category = _classify(email, config.classification)
        results.append(email)
        log.info("  %s | %s | %s | %d attachment(s)",
                 email.category, email.sender, email.subject[:50], len(email.attachments))

    _save_last_run_date(date.today())
    log.info("Processed %d email(s): %d scanner, %d commentary, %d other",
             len(results),
             sum(1 for e in results if e.category == "scanner"),
             sum(1 for e in results if e.category == "market_commentary"),
             sum(1 for e in results if e.category not in ("scanner", "market_commentary")))
    return results


def _get_gmail_service():
    """Authenticate and return a Gmail API service object."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        log.error("Gmail dependencies not installed. Run: uv add google-auth-oauthlib google-api-python-client")
        return None

    if not _CLIENT_SECRET_FILE.exists():
        log.error("Gmail client secret not found at %s. See docs/SETUP-GMAIL-API.md", _CLIENT_SECRET_FILE)
        return None

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


def _build_query(label: str, since: date | None) -> str:
    """Build Gmail search query."""
    parts = [f"label:{label}"]
    if since:
        parts.append(f"after:{since.isoformat()}")
    return " ".join(parts)


def _list_messages(service, query: str) -> list[dict]:
    """List message IDs matching the query."""
    try:
        response = service.users().messages().list(
            userId="me", q=query, maxResults=50,
        ).execute()
        return response.get("messages", [])
    except Exception:
        log.exception("Failed to list Gmail messages")
        return []


def _get_message(service, message_id: str) -> dict | None:
    """Fetch a full message by ID."""
    try:
        return service.users().messages().get(
            userId="me", id=message_id, format="full",
        ).execute()
    except Exception:
        log.exception("Failed to fetch message %s", message_id)
        return None


def _parse_message(
    msg: dict, service, staging_dir: Path,
) -> EmailMessage:
    """Extract headers, body text, and attachments from a Gmail message."""
    headers = {h["name"].lower(): h["value"] for h in msg["payload"].get("headers", [])}

    sender = headers.get("from", "")
    subject = headers.get("subject", "")
    msg_date = _parse_date(headers.get("date", ""))
    body = _extract_body(msg["payload"])
    attachments = _download_attachments(service, msg["id"], msg["payload"], staging_dir)

    return EmailMessage(
        message_id=msg["id"],
        sender=sender,
        subject=subject,
        date=msg_date,
        body_text=body,
        attachments=attachments,
    )


def _extract_body(payload: dict) -> str:
    """Extract plain text body from a message payload (handles multipart)."""
    if payload.get("mimeType") == "text/plain" and "body" in payload:
        data = payload["body"].get("data", "")
        if data:
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")

    for part in payload.get("parts", []):
        if part.get("mimeType") == "text/plain":
            data = part.get("body", {}).get("data", "")
            if data:
                return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
        # Recurse into multipart
        if part.get("parts"):
            result = _extract_body(part)
            if result:
                return result

    return ""


def _download_attachments(
    service, message_id: str, payload: dict, staging_dir: Path,
) -> list[Path]:
    """Download CSV attachments from a message."""
    attachments: list[Path] = []

    for part in payload.get("parts", []):
        filename = part.get("filename", "")
        if not filename or not filename.lower().endswith(".csv"):
            continue

        attachment_id = part.get("body", {}).get("attachmentId")
        if not attachment_id:
            continue

        try:
            att = service.users().messages().attachments().get(
                userId="me", messageId=message_id, id=attachment_id,
            ).execute()
            data = base64.urlsafe_b64decode(att["data"])

            out_path = staging_dir / filename
            out_path.write_bytes(data)
            attachments.append(out_path)
            log.debug("Downloaded attachment: %s", filename)
        except Exception:
            log.exception("Failed to download attachment %s", filename)

    return attachments


def _classify(email: EmailMessage, rules: list[dict[str, str]]) -> str:
    """Classify an email by sender/subject matching rules."""
    # If it has CSV attachments, it's a scanner email
    if email.attachments:
        return "scanner"

    sender_lower = email.sender.lower()
    for rule in rules:
        pattern = rule.get("sender", "").lower()
        if pattern and pattern in sender_lower:
            return rule.get("category", "unknown")

    # Default: if it has body text, treat as market commentary
    if email.body_text.strip():
        return "market_commentary"

    return "unknown"


def _parse_date(date_str: str) -> datetime:
    """Parse an email Date header into a datetime."""
    # Email dates have many formats; try common ones
    for fmt in (
        "%a, %d %b %Y %H:%M:%S %z",
        "%d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S",
    ):
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue

    # Fallback: strip timezone name suffix like " (UTC)" and retry
    cleaned = re.sub(r"\s*\([^)]*\)\s*$", "", date_str.strip())
    for fmt in (
        "%a, %d %b %Y %H:%M:%S %z",
        "%d %b %Y %H:%M:%S %z",
    ):
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue

    log.debug("Could not parse email date: %s", date_str)
    return datetime.now()


def _load_last_run_date() -> date | None:
    """Load the last successful run date from state file."""
    if not _STATE_FILE.exists():
        return None
    try:
        state = json.loads(_STATE_FILE.read_text())
        return date.fromisoformat(state["last_run_date"])
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def _save_last_run_date(d: date) -> None:
    """Save the current run date to state file."""
    state = {}
    if _STATE_FILE.exists():
        try:
            state = json.loads(_STATE_FILE.read_text())
        except json.JSONDecodeError:
            pass
    state["last_run_date"] = d.isoformat()
    _STATE_FILE.write_text(json.dumps(state, indent=2))
