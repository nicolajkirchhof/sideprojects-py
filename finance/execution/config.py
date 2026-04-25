from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_VALID_MODES = {"paper", "live"}


@dataclass(frozen=True)
class IbkrConfig:
    host: str
    paper_port: int
    live_port: int
    client_id: int


@dataclass(frozen=True)
class RiskConfig:
    max_daily_loss_eur: float
    max_daily_loss_usd: float


@dataclass(frozen=True)
class TelegramConfig:
    bot_token: str
    chat_id: str


@dataclass(frozen=True)
class EngineConfig:
    mode: str
    ibkr: IbkrConfig
    risk: RiskConfig
    telegram: TelegramConfig

    @property
    def active_port(self) -> int:
        return self.ibkr.paper_port if self.mode == "paper" else self.ibkr.live_port


def load_config(path: Path | str) -> EngineConfig:
    """Load and validate engine config from a YAML file.

    Raises:
        ValueError: if mode is not 'paper' or 'live', or required sections are missing.
        KeyError: if required fields are absent within a section.
    """
    raw = yaml.safe_load(Path(path).read_text())

    mode = raw.get("mode")
    if mode not in _VALID_MODES:
        raise ValueError(f"Invalid mode {mode!r}. Must be one of {_VALID_MODES}.")

    if "ibkr" not in raw:
        raise ValueError("Missing required config section: 'ibkr'")
    if "risk" not in raw:
        raise ValueError("Missing required config section: 'risk'")
    if "telegram" not in raw:
        raise ValueError("Missing required config section: 'telegram'")

    ibkr_raw = raw["ibkr"]
    ibkr = IbkrConfig(
        host=ibkr_raw["host"],
        paper_port=ibkr_raw["paper_port"],
        live_port=ibkr_raw["live_port"],
        client_id=ibkr_raw["client_id"],
    )

    risk_raw = raw["risk"]
    risk = RiskConfig(
        max_daily_loss_eur=risk_raw["max_daily_loss_eur"],
        max_daily_loss_usd=risk_raw["max_daily_loss_usd"],
    )

    telegram_raw = raw["telegram"]
    telegram = TelegramConfig(
        bot_token=telegram_raw["bot_token"],
        chat_id=str(telegram_raw["chat_id"]),
    )

    return EngineConfig(mode=mode, ibkr=ibkr, risk=risk, telegram=telegram)
