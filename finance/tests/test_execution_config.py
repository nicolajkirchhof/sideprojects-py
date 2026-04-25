from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from finance.execution.config import load_config


class TestLoadConfig:
    def test_loads_paper_mode(self, tmp_path: Path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(textwrap.dedent("""\
            mode: paper
            ibkr:
              paper_port: 4002
              live_port: 4001
              client_id: 10
              host: 127.0.0.1
            risk:
              max_daily_loss_eur: 100
              max_daily_loss_usd: 100
            telegram:
              bot_token: tok
              chat_id: "123"
        """))
        config = load_config(cfg_file)
        assert config.mode == "paper"
        assert config.ibkr.paper_port == 4002
        assert config.ibkr.client_id == 10

    def test_loads_live_mode(self, tmp_path: Path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(textwrap.dedent("""\
            mode: live
            ibkr:
              paper_port: 4002
              live_port: 4001
              client_id: 10
              host: 127.0.0.1
            risk:
              max_daily_loss_eur: 100
              max_daily_loss_usd: 100
            telegram:
              bot_token: tok
              chat_id: "123"
        """))
        config = load_config(cfg_file)
        assert config.mode == "live"

    def test_raises_on_invalid_mode(self, tmp_path: Path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(textwrap.dedent("""\
            mode: simulation
            ibkr:
              paper_port: 4002
              live_port: 4001
              client_id: 10
              host: 127.0.0.1
            risk:
              max_daily_loss_eur: 100
              max_daily_loss_usd: 100
            telegram:
              bot_token: tok
              chat_id: "123"
        """))
        with pytest.raises(ValueError, match="mode"):
            load_config(cfg_file)

    def test_raises_on_missing_ibkr_section(self, tmp_path: Path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(textwrap.dedent("""\
            mode: paper
            risk:
              max_daily_loss_eur: 100
              max_daily_loss_usd: 100
            telegram:
              bot_token: tok
              chat_id: "123"
        """))
        with pytest.raises((ValueError, KeyError, TypeError)):
            load_config(cfg_file)

    def test_raises_on_missing_risk_section(self, tmp_path: Path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(textwrap.dedent("""\
            mode: paper
            ibkr:
              paper_port: 4002
              live_port: 4001
              client_id: 10
              host: 127.0.0.1
            telegram:
              bot_token: tok
              chat_id: "123"
        """))
        with pytest.raises((ValueError, KeyError, TypeError)):
            load_config(cfg_file)

    def test_risk_config_has_both_loss_limits(self, tmp_path: Path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(textwrap.dedent("""\
            mode: paper
            ibkr:
              paper_port: 4002
              live_port: 4001
              client_id: 10
              host: 127.0.0.1
            risk:
              max_daily_loss_eur: 150
              max_daily_loss_usd: 120
            telegram:
              bot_token: tok
              chat_id: "123"
        """))
        config = load_config(cfg_file)
        assert config.risk.max_daily_loss_eur == 150
        assert config.risk.max_daily_loss_usd == 120

    def test_active_port_is_paper_port_in_paper_mode(self, tmp_path: Path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(textwrap.dedent("""\
            mode: paper
            ibkr:
              paper_port: 4002
              live_port: 4001
              client_id: 10
              host: 127.0.0.1
            risk:
              max_daily_loss_eur: 100
              max_daily_loss_usd: 100
            telegram:
              bot_token: tok
              chat_id: "123"
        """))
        config = load_config(cfg_file)
        assert config.active_port == 4002

    def test_active_port_is_live_port_in_live_mode(self, tmp_path: Path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(textwrap.dedent("""\
            mode: live
            ibkr:
              paper_port: 4002
              live_port: 4001
              client_id: 10
              host: 127.0.0.1
            risk:
              max_daily_loss_eur: 100
              max_daily_loss_usd: 100
            telegram:
              bot_token: tok
              chat_id: "123"
        """))
        config = load_config(cfg_file)
        assert config.active_port == 4001
