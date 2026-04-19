"""
finance.apps.analyst._config
==============================
Configuration loader for the analyst pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


@dataclass
class ScannerConfig:
    column_mapping: dict[str, str] = field(default_factory=dict)
    percent_columns: list[str] = field(default_factory=list)


@dataclass
class GmailConfig:
    label: str = "market"
    classification: list[dict[str, str]] = field(default_factory=list)


@dataclass
class ClaudeConfig:
    model_scanner: str = "claude-sonnet-4-6"
    model_review: str = "claude-opus-4-6"
    max_candidates: int = 10
    max_trade_reviews: int = 5


@dataclass
class TradelogConfig:
    api_url: str = "http://localhost:5286"
    account_id: int = 1


@dataclass
class AnalystConfig:
    scanner: ScannerConfig = field(default_factory=ScannerConfig)
    gmail: GmailConfig = field(default_factory=GmailConfig)
    web_sources: list[dict] = field(default_factory=list)
    claude: ClaudeConfig = field(default_factory=ClaudeConfig)
    tradelog: TradelogConfig = field(default_factory=TradelogConfig)


def load_config(override_path: Path | None = None) -> AnalystConfig:
    """Load config from default YAML, optionally merged with an override file."""
    raw = _load_yaml(_DEFAULT_CONFIG)
    if override_path and override_path.exists():
        overrides = _load_yaml(override_path)
        raw = _deep_merge(raw, overrides)
    return _parse(raw)


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base dict."""
    result = base.copy()
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _parse(raw: dict[str, Any]) -> AnalystConfig:
    scanner_raw = raw.get("scanner", {})
    gmail_raw = raw.get("gmail", {})
    claude_raw = raw.get("claude", {})
    tradelog_raw = raw.get("tradelog", {})

    return AnalystConfig(
        web_sources=raw.get("web_sources", []),
        scanner=ScannerConfig(
            column_mapping=scanner_raw.get("column_mapping", {}),
            percent_columns=scanner_raw.get("percent_columns", []),
        ),
        gmail=GmailConfig(
            label=gmail_raw.get("label", "market"),
            classification=gmail_raw.get("classification", []),
        ),
        claude=ClaudeConfig(
            model_scanner=claude_raw.get("model_scanner", "claude-sonnet-4-6"),
            model_review=claude_raw.get("model_review", "claude-opus-4-6"),
            max_candidates=claude_raw.get("max_candidates", 10),
            max_trade_reviews=claude_raw.get("max_trade_reviews", 5),
        ),
        tradelog=TradelogConfig(
            api_url=tradelog_raw.get("api_url", "http://localhost:5286"),
            account_id=tradelog_raw.get("account_id", 1),
        ),
    )
