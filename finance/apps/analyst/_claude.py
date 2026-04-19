"""
finance.apps.analyst._claude
================================
Claude API client for trade analysis.

Handles three analysis types:
- Market context summarization (E3-S1)
- Trade candidate reasoning (E3-S2)
- Historical trade compliance review (E3-S3)
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from finance.apps.analyst._config import ClaudeConfig
from finance.apps.analyst._gmail import EmailMessage
from finance.apps.analyst._web import WebArticle
from finance.apps.analyst._models import (
    ComplianceAggregate,
    MarketSummary,
    ScoredCandidate,
    TradeAnalysisResult,
    TradeRecommendation,
)

log = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "_prompts"


def summarize_market(
    emails: list[EmailMessage],
    config: ClaudeConfig,
    web_articles: list[WebArticle] | None = None,
) -> MarketSummary:
    """Summarize market emails and web articles into a structured brief (E3-S1)."""
    if not emails and not web_articles:
        return MarketSummary()

    email_text = _format_emails(emails) if emails else ""
    articles_text = _format_web_articles(web_articles) if web_articles else ""

    all_content = ""
    if email_text:
        all_content += email_text
    if articles_text:
        if all_content:
            all_content += "\n\n---\n\n"
        all_content += articles_text

    prompt_template = _load_prompt("market_summary.md")
    user_prompt = prompt_template.replace("{emails}", all_content)

    raw = _call_claude(
        user_prompt=user_prompt,
        model=config.model_scanner,
        purpose="market summary",
    )
    if not raw:
        return MarketSummary()

    parsed = _parse_json(raw)
    if not parsed:
        return MarketSummary(raw_response=raw)

    return MarketSummary(
        regime=parsed.get("regime", ""),
        regime_reasoning=parsed.get("regime_reasoning", ""),
        themes=parsed.get("themes", []),
        movers=parsed.get("movers", []),
        risks=parsed.get("risks", []),
        action_items=parsed.get("action_items", []),
        raw_response=raw,
    )


def analyze_candidates(
    scored: list[ScoredCandidate],
    market_summary: MarketSummary,
    config: ClaudeConfig,
) -> list[TradeRecommendation]:
    """Evaluate top candidates against the playbook (E3-S2)."""
    top = scored[:config.max_candidates]
    if not top:
        return []

    candidates_text = _format_candidates(top)
    context_text = _format_market_context(market_summary)
    prompt_template = _load_prompt("trade_reasoning.md")

    user_prompt = (
        prompt_template
        .replace("{candidates}", candidates_text)
        .replace("{market_context}", context_text)
    )

    raw = _call_claude(
        user_prompt=user_prompt,
        model=config.model_scanner,
        purpose="trade reasoning",
    )
    if not raw:
        return []

    parsed = _parse_json(raw)
    if not parsed or not isinstance(parsed, list):
        log.warning("Trade reasoning response was not a JSON array")
        return []

    return [
        TradeRecommendation(
            symbol=r.get("symbol", ""),
            setup_type=r.get("setup_type", ""),
            profit_mechanism=r.get("profit_mechanism", ""),
            thesis=r.get("thesis", ""),
            catalyst_assessment=r.get("catalyst_assessment", ""),
            recommended_structure=r.get("recommended_structure", ""),
            entry=r.get("entry"),
            stop=r.get("stop"),
            target=r.get("target"),
            risk_reward=r.get("risk_reward", ""),
            confidence=r.get("confidence", ""),
            reasoning=r.get("reasoning", ""),
        )
        for r in parsed
        if isinstance(r, dict)
    ]


def review_compliance(
    trades: list[dict[str, Any]],
    market_context: str,
    config: ClaudeConfig,
) -> tuple[list[TradeAnalysisResult], ComplianceAggregate | None]:
    """Review closed trades for playbook compliance (E3-S3).

    Args:
        trades: List of trade dicts from the Tradelog API export.
        market_context: Formatted market conditions at time of trades.
        config: Claude configuration.

    Returns:
        Tuple of (per-trade analyses, aggregate insights).
    """
    batch = trades[:config.max_trade_reviews]
    if not batch:
        return [], None

    trades_text = _format_trades(batch)
    prompt_template = _load_prompt("compliance.md")

    user_prompt = (
        prompt_template
        .replace("{trades}", trades_text)
        .replace("{market_context}", market_context)
    )

    raw = _call_claude(
        user_prompt=user_prompt,
        model=config.model_review,
        purpose="compliance review",
    )
    if not raw:
        return [], None

    parsed = _parse_json(raw)
    if not parsed:
        return [], None

    # Parse individual trade reviews
    reviews_data = parsed if isinstance(parsed, list) else parsed.get("reviews", [])
    reviews = [
        TradeAnalysisResult(
            trade_id=r.get("trade_id", 0),
            symbol=r.get("symbol", ""),
            score=r.get("score", 0),
            analysis=r.get("analysis", ""),
        )
        for r in reviews_data
        if isinstance(r, dict) and "trade_id" in r
    ]

    # Parse aggregate (may be in the response or as a separate key)
    agg_data = parsed.get("aggregate") if isinstance(parsed, dict) else None
    aggregate = None
    if agg_data and isinstance(agg_data, dict):
        aggregate = ComplianceAggregate(
            avg_score=agg_data.get("avg_score", 0),
            patterns=agg_data.get("patterns", []),
            top_improvement=agg_data.get("top_improvement", ""),
            refinements=agg_data.get("refinements", []),
        )

    return reviews, aggregate


# --- Claude API ---

def _call_claude(*, user_prompt: str, model: str, purpose: str) -> str:
    """Call the Claude API with the system prompt + user prompt."""
    try:
        import anthropic
    except ImportError:
        log.error("anthropic package not installed. Run: uv add anthropic")
        return ""

    api_key = _get_api_key()
    if not api_key:
        return ""

    system_prompt = _load_prompt("system.md")

    log.info("Calling Claude (%s) for %s...", model, purpose)
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
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        log.info("  → %d input + %d output tokens", input_tokens, output_tokens)
        return result

    except anthropic.APIError:
        log.exception("Claude API error for %s", purpose)
        return ""


def _get_api_key() -> str:
    """Get the Anthropic API key from environment."""
    import os
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        log.error("ANTHROPIC_API_KEY environment variable not set")
    return key


# --- Prompt helpers ---

def _load_prompt(filename: str) -> str:
    """Load a prompt template from the _prompts directory."""
    path = _PROMPTS_DIR / filename
    if not path.exists():
        log.error("Prompt template not found: %s", path)
        return ""
    return path.read_text(encoding="utf-8")


def _parse_json(text: str) -> Any:
    """Extract and parse JSON from Claude's response.

    Handles responses that may have markdown code fences around the JSON.
    """
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` code fence
    match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding the first { or [ and parse from there
    # Prefer whichever appears first in the text
    candidates_to_try: list[tuple[int, str]] = []
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end > start:
            candidates_to_try.append((start, text[start:end + 1]))
    candidates_to_try.sort(key=lambda x: x[0])

    for _, fragment in candidates_to_try:
        try:
            return json.loads(fragment)
        except json.JSONDecodeError:
            pass

    log.warning("Could not parse JSON from Claude response (%d chars)", len(text))
    return None


# --- Formatting helpers ---

def _format_web_articles(articles: list[WebArticle]) -> str:
    """Format web articles for the prompt."""
    parts = []
    for a in articles:
        content = a.content[:2000]
        parts.append(f"## {a.title} (via {a.source})\nURL: {a.url}\n\n{content}")
    return "\n\n---\n\n".join(parts)


def _format_emails(emails: list[EmailMessage]) -> str:
    """Format emails for the prompt."""
    parts = []
    for e in emails:
        body = e.body_text[:2000]  # truncate long emails
        parts.append(f"## {e.subject}\nFrom: {e.sender}\nDate: {e.date}\n\n{body}")
    return "\n\n---\n\n".join(parts)


def _format_candidates(scored: list[ScoredCandidate]) -> str:
    """Format scored candidates for the prompt."""
    parts = []
    for sc in scored:
        c = sc.enriched.candidate
        t = sc.enriched.technicals

        lines = [f"## {c.symbol} (Score: {sc.score}/5)"]
        if c.price:
            lines.append(f"Price: ${c.price:.2f}")
        if c.sector:
            lines.append(f"Sector: {c.sector}")
        if c.change_pct is not None:
            lines.append(f"Today: {c.change_pct:+.1f}%")
        if c.change_5d_pct is not None:
            lines.append(f"5d Change: {c.change_5d_pct:+.1f}%")
        if c.change_1m_pct is not None:
            lines.append(f"1M Change: {c.change_1m_pct:+.1f}%")
        if c.change_52w_pct is not None:
            lines.append(f"52W Change: {c.change_52w_pct:+.1f}%")
        if c.iv_percentile is not None:
            lines.append(f"IV Percentile: {c.iv_percentile:.0f}%")
        if c.put_call_vol_5d is not None:
            lines.append(f"5D Put/Call: {c.put_call_vol_5d:.2f}")
        if c.latest_earnings:
            lines.append(f"Next Earnings: {c.latest_earnings}")

        if t:
            if t.sma_20 is not None:
                lines.append(f"20 SMA: {t.sma_20:.2f}")
            if t.sma_50 is not None:
                lines.append(f"50 SMA: {t.sma_50:.2f} (slope: {t.sma_50_slope})")
            if t.rs_slope_10d is not None:
                lines.append(f"RS slope 10d: {t.rs_slope_10d:+.2f}%")
            if t.bb_width is not None and t.bb_width_avg_20 is not None:
                squeeze = "YES" if t.bb_width < t.bb_width_avg_20 else "no"
                lines.append(f"BB squeeze: {squeeze} (width: {t.bb_width:.4f}, avg: {t.bb_width_avg_20:.4f})")
            if t.rvol is not None:
                lines.append(f"RVOL: {t.rvol:.1f}x")
            if t.volume_contracting is not None:
                lines.append(f"VDU: {'YES' if t.volume_contracting else 'no'}")
            if t.atr_14 is not None:
                lines.append(f"ATR(14): {t.atr_14:.2f}")
            if t.return_12m is not None:
                lines.append(f"12M return: {t.return_12m:+.1f}%")

        lines.append("\n5-Box Results:")
        for box in sc.boxes:
            lines.append(f"  Box {box.box} ({box.name}): {box.status} — {box.reason}")

        parts.append("\n".join(lines))
    return "\n\n---\n\n".join(parts)


def _format_market_context(summary: MarketSummary) -> str:
    """Format market summary as context for other prompts."""
    if not summary.regime:
        return "No market summary available."

    lines = [
        f"Regime: {summary.regime}",
        f"Reasoning: {summary.regime_reasoning}",
    ]
    if summary.themes:
        lines.append(f"Themes: {', '.join(summary.themes)}")
    if summary.risks:
        lines.append(f"Risks: {', '.join(summary.risks)}")
    return "\n".join(lines)


def _format_trades(trades: list[dict[str, Any]]) -> str:
    """Format trade export data for the compliance prompt."""
    parts = []
    for t in trades:
        lines = [
            f"## Trade #{t.get('id', '?')}: {t.get('symbol', '?')}",
            f"Date: {t.get('date', '?')}",
            f"Strategy: {t.get('strategy', '?')}",
            f"Type of Trade: {t.get('typeOfTrade', '?')}",
            f"Directional: {t.get('directional', '?')}",
            f"Budget: {t.get('budget', '?')}",
            f"P/L: {t.get('pnl', '?')}",
            f"Status: {t.get('status', '?')}",
        ]
        if t.get("notes"):
            lines.append(f"Thesis: {t['notes'][:500]}")
        if t.get("intendedManagement"):
            lines.append(f"Intended Management: {t['intendedManagement'][:300]}")
        if t.get("actualManagement"):
            lines.append(f"Actual Management: {t['actualManagement'][:300]}")
        if t.get("managementRating") is not None:
            lines.append(f"Self-Rating: {t['managementRating']}")
        if t.get("learnings"):
            lines.append(f"Learnings: {t['learnings'][:300]}")
        parts.append("\n".join(lines))
    return "\n\n---\n\n".join(parts)
