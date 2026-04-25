from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class Journal:
    """Append-only JSONL event log, one file per trading day.

    Thread-safety: this class performs plain synchronous appends. It is safe
    because the execution engine runs entirely within a single asyncio event
    loop — cooperative scheduling ensures only one coroutine writes at a time.
    """

    def __init__(self, log_dir: Path | str) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def write(self, entry: dict) -> None:
        """Append one JSON line to today's log file.

        The caller's dict is not modified. A ``ts`` field (UTC ISO-8601) is
        injected into the written record automatically.
        """
        now = datetime.now(tz=timezone.utc)
        record = {"ts": now.isoformat(), **entry}
        log_file = self._log_dir / f"{now.date().isoformat()}.jsonl"
        with log_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
