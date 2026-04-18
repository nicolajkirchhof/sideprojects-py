"""
finance.apps.analyst._tradelog
=================================
Tradelog REST API client for pushing analysis results.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import date
from typing import Any

import requests

from finance.apps.analyst._config import TradelogConfig
from finance.apps.analyst._models import (
    MarketSummary,
    ScoredCandidate,
    TradeAnalysisResult,
    TradeRecommendation,
)

log = logging.getLogger(__name__)


def push_daily_prep(
    *,
    config: TradelogConfig,
    report_date: date,
    market_summary: MarketSummary | None,
    scored: list[ScoredCandidate],
    recommendations: list[TradeRecommendation],
    email_count: int,
) -> bool:
    """Push daily prep report to the Tradelog API."""
    watchlist_data = _build_watchlist_json(scored, recommendations)

    payload = {
        "date": report_date.isoformat(),
        "marketSummary": json.dumps(asdict(market_summary)) if market_summary and market_summary.regime else None,
        "watchlist": json.dumps(watchlist_data) if watchlist_data else None,
        "emailCount": email_count,
        "candidateCount": len(scored),
    }

    url = f"{config.api_url}/api/daily-prep"
    headers = _headers(config)

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        log.info("Pushed daily prep report for %s", report_date)
        return True
    except requests.RequestException:
        log.exception("Failed to push daily prep report")
        return False


def push_trade_analysis(
    *,
    config: TradelogConfig,
    analysis: TradeAnalysisResult,
    model: str,
) -> bool:
    """Push a single trade analysis to the Tradelog API."""
    payload = {
        "analysisDate": date.today().isoformat(),
        "score": analysis.score,
        "analysis": analysis.analysis,
        "model": model,
    }

    url = f"{config.api_url}/api/trades/{analysis.trade_id}/analysis"
    headers = _headers(config)

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        log.info("Pushed analysis for trade #%d (score: %d)", analysis.trade_id, analysis.score)
        return True
    except requests.RequestException:
        log.exception("Failed to push analysis for trade #%d", analysis.trade_id)
        return False


def fetch_trades_for_review(
    config: TradelogConfig,
    status: str = "Closed",
    since: date | None = None,
) -> list[dict[str, Any]]:
    """Fetch trades from the Tradelog API for compliance review."""
    url = f"{config.api_url}/api/trades/export"
    params: dict[str, str] = {"status": status}
    if since:
        params["since"] = since.isoformat()
    headers = _headers(config)

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        log.exception("Failed to fetch trades for review")
        return []


def _headers(config: TradelogConfig) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "X-Account-Id": str(config.account_id),
    }


def _build_watchlist_json(
    scored: list[ScoredCandidate],
    recommendations: list[TradeRecommendation],
) -> list[dict[str, Any]]:
    """Merge scored candidates with Claude recommendations into a single watchlist."""
    rec_map = {r.symbol: r for r in recommendations}

    result = []
    for sc in scored:
        c = sc.enriched.candidate
        rec = rec_map.get(c.symbol)

        entry: dict[str, Any] = {
            "symbol": c.symbol,
            "score": sc.score,
            "price": c.price,
            "change_5d_pct": c.change_5d_pct,
            "change_1m_pct": c.change_1m_pct,
            "boxes": [asdict(b) for b in sc.boxes],
        }

        if rec:
            entry.update({
                "setup_type": rec.setup_type,
                "profit_mechanism": rec.profit_mechanism,
                "confidence": rec.confidence,
                "thesis": rec.thesis,
                "reasoning": rec.reasoning,
                "recommended_structure": rec.recommended_structure,
                "entry": rec.entry,
                "stop": rec.stop,
                "target": rec.target,
                "catalyst_assessment": rec.catalyst_assessment,
            })

        result.append(entry)
    return result
