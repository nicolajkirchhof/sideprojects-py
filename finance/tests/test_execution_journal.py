from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from finance.execution.journal import Journal

_TODAY = datetime.now(tz=timezone.utc).date().isoformat()


class TestJournalWrite:
    def test_creates_log_file_on_first_write(self, tmp_path: Path):
        journal = Journal(log_dir=tmp_path)
        journal.write({"event": "signal", "strategy_id": "srs_fdxs", "symbol": "FDXS", "direction": "long"})
        files = list(tmp_path.glob("*.jsonl"))
        assert len(files) == 1

    def test_log_file_named_by_utc_today(self, tmp_path: Path):
        journal = Journal(log_dir=tmp_path)
        journal.write({"event": "signal", "strategy_id": "srs_fdxs", "symbol": "FDXS", "direction": "long"})
        assert (tmp_path / f"{_TODAY}.jsonl").exists()

    def test_entry_is_valid_json(self, tmp_path: Path):
        journal = Journal(log_dir=tmp_path)
        journal.write({"event": "placed", "strategy_id": "srs_fdxs", "symbol": "FDXS", "direction": "long"})
        line = (tmp_path / f"{_TODAY}.jsonl").read_text().strip()
        parsed = json.loads(line)
        assert parsed["event"] == "placed"

    def test_ts_field_added_automatically(self, tmp_path: Path):
        journal = Journal(log_dir=tmp_path)
        journal.write({"event": "fill", "strategy_id": "srs_fdxs", "symbol": "FDXS", "direction": "long"})
        line = (tmp_path / f"{_TODAY}.jsonl").read_text().strip()
        parsed = json.loads(line)
        assert "ts" in parsed

    def test_ts_field_is_iso_with_timezone(self, tmp_path: Path):
        journal = Journal(log_dir=tmp_path)
        journal.write({"event": "cancel", "strategy_id": "srs_fdxs", "symbol": "FDXS", "direction": "long"})
        line = (tmp_path / f"{_TODAY}.jsonl").read_text().strip()
        ts = json.loads(line)["ts"]
        dt = datetime.fromisoformat(ts)
        assert dt.tzinfo is not None

    def test_multiple_writes_append_separate_lines(self, tmp_path: Path):
        journal = Journal(log_dir=tmp_path)
        journal.write({"event": "signal", "strategy_id": "srs_fdxs", "symbol": "FDXS", "direction": "long"})
        journal.write({"event": "placed", "strategy_id": "srs_fdxs", "symbol": "FDXS", "direction": "long"})
        lines = (tmp_path / f"{_TODAY}.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2

    def test_each_line_is_independent_json(self, tmp_path: Path):
        journal = Journal(log_dir=tmp_path)
        journal.write({"event": "signal", "strategy_id": "s1", "symbol": "MNQ", "direction": "long"})
        journal.write({"event": "fill", "strategy_id": "s2", "symbol": "FDXS", "direction": "short"})
        lines = (tmp_path / f"{_TODAY}.jsonl").read_text().strip().splitlines()
        first = json.loads(lines[0])
        second = json.loads(lines[1])
        assert first["strategy_id"] == "s1"
        assert second["strategy_id"] == "s2"

    def test_write_does_not_mutate_caller_dict(self, tmp_path: Path):
        journal = Journal(log_dir=tmp_path)
        entry = {"event": "skip", "strategy_id": "srs_mnq", "symbol": "MNQ", "direction": "short"}
        journal.write(entry)
        assert "ts" not in entry

    def test_all_entry_types_are_accepted(self, tmp_path: Path):
        journal = Journal(log_dir=tmp_path)
        entry_types = [
            "signal", "placed", "fill", "cancel", "skip", "flip",
            "eod_flatten", "partial_fill_ignored", "missed_signal",
            "strategy_halted", "pre_trade_gate_blocked",
        ]
        for event in entry_types:
            journal.write({"event": event, "strategy_id": "x", "symbol": "MNQ", "direction": "long"})

        lines = (tmp_path / f"{_TODAY}.jsonl").read_text().strip().splitlines()
        written = [json.loads(line)["event"] for line in lines]
        assert written == entry_types
