"""
finance.apps.assistant._claude
================================
Claude API client for the Trading Assistant.

Provides:
  summarize_market    — generate a MarketSummary from email bodies (TA-E3-S4)
  analyze_candidate   — generate a CandidateAnalysis for a single watchlist row (TA-E5-S2)

Error policy: API failures are logged and return empty/default objects —
callers decide whether to surface errors or silently skip.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from finance.apps.assistant._gmail import EmailBody
from finance.apps.assistant._models import CandidateAnalysis, MarketSummary

if TYPE_CHECKING:
    from finance.apps.assistant._data import TrendStatus, VixStatus

log = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "_prompts"

_MAX_EMAIL_CHARS: int = 2000  # per-email truncation before sending to Claude


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def summarize_market(
    emails: list[EmailBody],
    model: str,
    *,
    market_context: str = "",
) -> MarketSummary:
    """
    Generate a MarketSummary from *emails* using Claude.

    Parameters
    ----------
    emails:
        Text bodies of market commentary emails to summarise.
    model:
        Claude model ID to use (e.g. ``"claude-sonnet-4-6"``).
    market_context:
        Optional web-researched market snippet injected before the emails.

    Returns
    -------
    MarketSummary
        Populated on success; empty (all defaults) when *emails* is empty.

    Raises
    ------
    RuntimeError
        When the Claude API call fails (empty API key, network error, etc.).
    """
    if not emails:
        return MarketSummary()

    email_text = _format_emails(emails)
    prompt_template = _load_prompt("market_summary.md")

    if market_context:
        context_section = f"# Market Research\n{market_context}\n"
    else:
        context_section = ""
    user_prompt = (
        prompt_template
        .replace("{market_context_section}", context_section)
        .replace("{emails}", email_text)
    )

    raw = _call_claude(user_prompt=user_prompt, model=model, purpose="market summary")
    if not raw:
        raise RuntimeError("Claude API call failed — check ANTHROPIC_API_KEY")

    parsed = _parse_json(raw)
    if not parsed:
        return MarketSummary(raw_response=raw)

    return MarketSummary(
        regime=parsed.get("regime", ""),
        regime_reasoning=parsed.get("regime_reasoning", ""),
        themes=parsed.get("themes") or [],
        movers=parsed.get("movers") or [],
        risks=parsed.get("risks") or [],
        action_items=parsed.get("action_items") or [],
        raw_response=raw,
    )


def analyze_candidate(
    row: dict,
    model: str,
) -> CandidateAnalysis:
    """
    Generate a trade analysis for a single watchlist candidate using Claude.

    Parameters
    ----------
    row:
        A result row dict as produced by build_result_row().
    model:
        Claude model ID to use (e.g. ``"claude-sonnet-4-6"``).

    Returns
    -------
    CandidateAnalysis
        Populated on success; empty (all defaults) when the API call fails
        or the response cannot be parsed.
    """
    prompt_template = _load_prompt("candidate_analysis.md")
    user_prompt = _format_candidate_prompt(prompt_template, row)

    raw = _call_claude(
        user_prompt=user_prompt,
        model=model,
        purpose=f"candidate analysis for {row.get('symbol', '?')}",
    )
    if not raw:
        return CandidateAnalysis()

    parsed = _parse_json(raw)
    if not parsed:
        return CandidateAnalysis(raw_response=raw)

    return CandidateAnalysis(
        setup_type=parsed.get("setup_type", ""),
        profit_mechanism=parsed.get("profit_mechanism", ""),
        thesis=parsed.get("thesis", ""),
        entry=_to_float(parsed.get("entry")),
        stop=_to_float(parsed.get("stop")),
        target=_to_float(parsed.get("target")),
        confidence=parsed.get("confidence", ""),
        raw_response=raw,
    )


def review_trade(
    trade: dict,
    alerts: list,
    model: str,
    *,
    regime_at_entry: str = "",
) -> dict:
    """
    Generate a narrative trade review for a closed trade using Claude.

    Parameters
    ----------
    trade:
        Closed trade dict as returned by ``fetch_closed_trades()``.
    alerts:
        List of Alert objects produced by ``evaluate_position()`` for this trade.
    model:
        Claude model ID to use.
    regime_at_entry:
        GO/NO-GO regime string at the time of entry (best-effort).

    Returns
    -------
    dict
        Parsed review with keys: verdict, summary, what_went_well,
        what_to_improve, key_lesson, raw_response.
        On API failure raises RuntimeError; on parse failure returns
        ``{"summary": raw_response, "raw_response": raw_response}``.
    """
    prompt_template = _load_prompt("trade_review.md")
    user_prompt = _format_trade_review_prompt(prompt_template, trade, alerts, regime_at_entry)

    raw = _call_claude(
        user_prompt=user_prompt,
        model=model,
        purpose=f"trade review for {trade.get('symbol', '?')}",
    )
    if not raw:
        raise RuntimeError("Claude API returned empty response — check ANTHROPIC_API_KEY")

    parsed = _parse_json(raw)
    if not parsed:
        return {"summary": raw, "raw_response": raw}

    parsed["raw_response"] = raw
    return parsed


# ---------------------------------------------------------------------------
# Local regime context
# ---------------------------------------------------------------------------

_SLOPE_ARROW = {"rising": "↗", "falling": "↘", "flat": "→"}


def format_regime_context(
    spy: TrendStatus | None,
    qqq: TrendStatus | None,
    vix: VixStatus | None,
    regime_status: str = "",
) -> str:
    """
    Format locally-computed SPY/QQQ/VIX regime data into a markdown block.

    Returns ``""`` if all inputs are ``None``.
    """
    if spy is None and qqq is None and vix is None:
        return ""

    def _position(ts: TrendStatus) -> str:
        if ts.price_above_50 and ts.price_above_200:
            return "above both SMAs"
        if ts.price_above_50:
            return "above 50d, below 200d"
        if ts.price_above_200:
            return "below 50d, above 200d"
        return "below both SMAs"

    lines = ["# Local Market Data"]
    for ts in (spy, qqq):
        if ts is None:
            continue
        a50 = _SLOPE_ARROW.get(ts.sma_50_slope, "→")
        a200 = _SLOPE_ARROW.get(ts.sma_200_slope, "→")
        lines.append(
            f"{ts.symbol:<4} {ts.last_price:>8.2f}  "
            f"50d {ts.sma_50:.1f} {a50}  "
            f"200d {ts.sma_200:.1f} {a200}  "
            f"{_position(ts)}"
        )
    if vix is not None:
        lines.append(
            f"VIX  {vix.level:>6.2f}   zone: {vix.zone}   direction: {vix.direction}"
        )
    if regime_status:
        lines.append(f"Regime: {regime_status}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Web market context
# ---------------------------------------------------------------------------


def fetch_market_context(trade_date: date, *, timeout: float = 15.0) -> str:
    """
    Fetch 2–3 DuckDuckGo search snippets for current market context.

    Best-effort: returns ``""`` on any network or parsing error so that
    callers are never blocked by a failed web fetch.

    Parameters
    ----------
    trade_date:
        Session date used to constrain the search queries.
    timeout:
        Per-request timeout in seconds.
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        log.warning("requests/beautifulsoup4 not installed — skipping web market context")
        return ""

    date_str = trade_date.strftime("%B %d %Y")
    queries = [
        f"stock market SPY QQQ today {date_str}",
        f"VIX market volatility sector movers {date_str}",
    ]

    snippets: list[str] = []
    headers = {"User-Agent": "Mozilla/5.0 (compatible; market-research-bot/1.0)"}

    for query in queries:
        try:
            url = "https://html.duckduckgo.com/html/"
            resp = requests.get(url, params={"q": query}, headers=headers, timeout=timeout)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for el in soup.select(".result__snippet")[:4]:
                text = el.get_text(separator=" ", strip=True)
                if text:
                    snippets.append(text)
        except Exception:
            log.debug("Web context query failed: %r", query, exc_info=True)

    if not snippets:
        return ""

    lines = "\n".join(f"- {s}" for s in snippets)
    web_block = f"Web snippets for {date_str}:\n{lines}\n"

    news_block = _fetch_barchart_news()
    return "\n\n".join(filter(None, [web_block, news_block]))


def _fetch_barchart_news(*, timeout: float = 10.0) -> str:
    """
    Fetch up to 5 headlines from Barchart's public RSS feed.

    Best-effort: returns ``""`` on any network or parsing error.
    """
    try:
        import xml.etree.ElementTree as ET

        import requests
    except ImportError:
        return ""

    try:
        resp = requests.get(
            "https://www.barchart.com/headlines/rss/headlines",
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; market-research-bot/1.0)"},
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        items = root.findall(".//item")[:5]
        if not items:
            return ""
        parts: list[str] = []
        for item in items:
            title = (item.findtext("title") or "").strip()
            desc = (item.findtext("description") or "").strip()
            if title:
                parts.append(f"**{title}**")
            if desc:
                parts.append(desc)
        if not parts:
            return ""
        return "# Barchart News\n" + "\n".join(parts) + "\n"
    except Exception:
        log.debug("Barchart RSS fetch failed", exc_info=True)
        return ""


# ---------------------------------------------------------------------------
# Claude API
# ---------------------------------------------------------------------------


def _call_claude(*, user_prompt: str, model: str, purpose: str) -> str:
    """Call the Claude API and return the response text, or "" on failure."""
    try:
        import anthropic
    except ImportError:
        log.error("anthropic package not installed. Run: uv add anthropic")
        return ""

    api_key = _get_api_key()
    if not api_key:
        return ""

    system_prompt = _load_prompt("system.md")

    log.info("Calling Claude (%s) for %s…", model, purpose)
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=[{
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_prompt}],
        )
        result = response.content[0].text
        log.info("  → %d in + %d out tokens",
                 response.usage.input_tokens, response.usage.output_tokens)
        return result
    except Exception:
        log.exception("Claude API error for %s", purpose)
        return ""


def _get_api_key() -> str:
    import os
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return key

    # Fall back to config.local.yaml next to config.yaml
    local_cfg = Path(__file__).parent / "config.local.yaml"
    if local_cfg.exists():
        try:
            import yaml
            raw = yaml.safe_load(local_cfg.read_text(encoding="utf-8")) or {}
            key = raw.get("claude", {}).get("api_key", "")
            if key:
                return key
        except Exception:
            log.debug("Failed to read api_key from %s", local_cfg, exc_info=True)

    log.error(
        "ANTHROPIC_API_KEY not set. Set the env var or add it to %s:\n"
        "  claude:\n"
        "    api_key: sk-ant-...",
        local_cfg,
    )
    return ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_prompt(filename: str) -> str:
    path = _PROMPTS_DIR / filename
    if not path.exists():
        log.error("Prompt template not found: %s", path)
        return ""
    return path.read_text(encoding="utf-8")


def _format_emails(emails: list[EmailBody]) -> str:
    parts = []
    for e in emails:
        body = e.body_text[:_MAX_EMAIL_CHARS]
        parts.append(f"## {e.subject}\nFrom: {e.sender}\nDate: {e.date}\n\n{body}")
    return "\n\n---\n\n".join(parts)


def _format_candidate_prompt(template: str, row: dict) -> str:
    """Fill the candidate_analysis.md template with values from *row*."""
    dims = row.get("dimensions") or []
    dim_lines = "\n".join(
        f"  D{d.get('dimension', '?')} {d.get('name', '')}: {d.get('weighted_score', 0):.1f}"
        for d in dims
    )
    tags = ", ".join(row.get("tags") or []) or "none"

    def _v(key: str, fmt: str = "") -> str:
        v = row.get(key)
        if v is None:
            return "n/a"
        return f"{v:{fmt}}" if fmt else str(v)

    return (
        template
        .replace("{symbol}", _v("symbol"))
        .replace("{direction}", _v("direction"))
        .replace("{price}", _v("price", ".2f"))
        .replace("{score_total}", _v("score_total", ".1f"))
        .replace("{change_5d_pct}", _v("change_5d_pct", "+.1f"))
        .replace("{change_1m_pct}", _v("change_1m_pct", "+.1f"))
        .replace("{rvol_20d}", _v("rvol_20d", ".1f"))
        .replace("{atr_pct_20d}", _v("atr_pct_20d", ".1f"))
        .replace("{iv_percentile}", _v("iv_percentile", ".0f"))
        .replace("{put_call_vol_5d}", _v("put_call_vol_5d", ".2f"))
        .replace("{earnings_surprise_pct}", _v("earnings_surprise_pct", "+.1f"))
        .replace("{latest_earnings}", _v("latest_earnings"))
        .replace("{sector}", _v("sector"))
        .replace("{tags}", tags)
        .replace("{dimension_scores}", dim_lines)
    )


def _format_trade_review_prompt(
    template: str,
    trade: dict,
    alerts: list,
    regime_at_entry: str,
) -> str:
    """Fill the trade_review.md template with values from *trade* and *alerts*."""
    pnl_d = float(trade.get("pnl") or trade.get("pnlDollars") or 0)
    initial_risk = float(trade.get("initialRisk") or trade.get("xAtrMove") or 1)
    pnl_r = pnl_d / initial_risk if initial_risk else 0

    flag_lines = "\n".join(
        f"- [{a.severity.upper()}] {a.rule}: {a.message}"
        for a in alerts
    ) or "None"

    return (
        template
        .replace("{symbol}", str(trade.get("symbol", "?")))
        .replace("{direction}", str(trade.get("direction", "?")))
        .replace("{position_type}", str(trade.get("positionType") or trade.get("type", "stock")))
        .replace("{entry_price}", str(trade.get("entryPrice") or trade.get("openPrice") or "n/a"))
        .replace("{exit_price}", str(trade.get("exitPrice") or trade.get("closePrice") or "n/a"))
        .replace("{pnl_dollars}", f"{pnl_d:+.0f}")
        .replace("{pnl_r:.2f}", f"{pnl_r:.2f}")
        .replace("{days_held}", str(trade.get("daysHeld") or "n/a"))
        .replace("{initial_risk}", f"{initial_risk:.2f}")
        .replace("{open_date}", str(trade.get("openDate") or "n/a"))
        .replace("{close_date}", str(trade.get("closeDate") or "n/a"))
        .replace("{rule_flags}", flag_lines)
        .replace("{regime_at_entry}", regime_at_entry or "unknown")
    )


def _to_float(value: object) -> float | None:
    """Convert *value* to float, returning None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_json(text: str) -> dict | None:
    """Extract and parse JSON from Claude's response text."""
    # Direct parse
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        pass

    # Extract from ```json ... ``` code fence
    match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(1))
            return result if isinstance(result, dict) else None
        except json.JSONDecodeError:
            pass

    # Find first { … } block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            result = json.loads(text[start:end + 1])
            return result if isinstance(result, dict) else None
        except json.JSONDecodeError:
            pass

    log.warning("Could not parse JSON from Claude response (%d chars)", len(text))
    return None
