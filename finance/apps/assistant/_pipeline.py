"""
finance.apps.assistant._pipeline
==================================
Background pipeline thread and JSON cache I/O for the Trading Assistant.

Pipeline stages (PipelineThread.run):
  1. Discover scanner CSVs for today's date
  2. Parse each CSV → deduplicated list[Candidate]
  3. IBKR enrichment (uses local Parquet data; gateway check runs before stage 3)
  4. Assign tags + direction + score each candidate
  5. Write JSON cache → _data/assistant/YYYY-MM-DD.json

Error policy: any unhandled exception emits the ``error`` signal with full
traceback and stops the thread immediately. The caller is responsible for
showing the error dialog and re-enabling the UI.

Cache format:
    {
        "date": "YYYY-MM-DD",
        "created_at": "YYYY-MM-DDTHH:MM:SS",
        "rows": [ { ...result row... }, ... ]
    }
"""
from __future__ import annotations

import dataclasses
import json
import logging
import socket
import traceback
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pyqtgraph.Qt import QtCore

if TYPE_CHECKING:
    from finance.apps.analyst._models import EnrichedCandidate
    from finance.apps.assistant._models import CandidateScore

log = logging.getLogger(__name__)

_DEFAULT_CSV_DIR = Path("finance/_data/barchart/screener")
_DEFAULT_CACHE_DIR = Path("finance/_data/assistant")

_IBKR_HOST = "127.0.0.1"
_IBKR_PORT = 4002          # live gateway; paper uses 4001
_IBKR_TIMEOUT = 3.0        # seconds


# ---------------------------------------------------------------------------
# IBKR gateway reachability check
# ---------------------------------------------------------------------------

def check_ibkr_gateway(
    host: str = _IBKR_HOST,
    port: int = _IBKR_PORT,
    timeout: float = _IBKR_TIMEOUT,
) -> None:
    """
    Verify that the IBKR Gateway is reachable via TCP.

    Raises
    ------
    RuntimeError
        If the connection cannot be established within *timeout* seconds.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
    except OSError as exc:
        raise RuntimeError(
            f"IBKR Gateway not reachable at {host}:{port}. "
            "Start Gateway / TWS and retry."
        ) from exc


# ---------------------------------------------------------------------------
# JSON cache I/O
# ---------------------------------------------------------------------------

def cache_path(trade_date: date, *, base_dir: Path | None = None) -> Path:
    """Return the cache file path for *trade_date*."""
    base = Path(base_dir) if base_dir is not None else _DEFAULT_CACHE_DIR
    return base / f"{trade_date.isoformat()}.json"


def write_cache(
    rows: list[dict],
    trade_date: date,
    *,
    base_dir: Path | None = None,
) -> Path:
    """
    Write *rows* to the daily JSON cache file.

    Returns the path that was written.
    """
    path = cache_path(trade_date, base_dir=base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "date": trade_date.isoformat(),
        "created_at": datetime.now().replace(microsecond=0).isoformat(),
        "rows": rows,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_cache(
    trade_date: date,
    *,
    base_dir: Path | None = None,
) -> list[dict] | None:
    """
    Load today's JSON cache if it exists.

    Returns
    -------
    list[dict] | None
        The ``rows`` list, or ``None`` if no cache file exists for this date.
    """
    path = cache_path(trade_date, base_dir=base_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload.get("rows", [])
    except Exception:
        log.warning("Failed to read cache %s — treating as missing", path, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Result row builder
# ---------------------------------------------------------------------------

def build_result_row(ec: EnrichedCandidate, sc: CandidateScore) -> dict:
    """
    Combine an EnrichedCandidate and a CandidateScore into a flat+nested dict
    suitable for JSON serialisation and watchlist table display.

    The row carries:
    - Identity: symbol, direction, price, sector, latest_earnings
    - Key scanner momentum fields: change_5d_pct, change_1m_pct, rvol_20d, iv_percentile
    - Scoring: score_total, score_tag_bonus, tags, dimensions (nested)
    """
    c = ec.candidate
    return {
        # --- Identity ---
        "symbol": c.symbol,
        "direction": sc.direction,
        # --- Price & momentum ---
        "price": c.price,
        "change_pct": c.change_pct,
        "change_5d_pct": c.change_5d_pct,
        "change_1m_pct": c.change_1m_pct,
        "change_3m_pct": c.change_3m_pct,
        "rvol_20d": c.rvol_20d,
        "atr_pct_20d": c.atr_pct_20d,
        "volume": c.volume,
        # --- Fundamentals / metadata ---
        "sector": c.sector,
        "market_cap_k": c.market_cap_k,
        "latest_earnings": c.latest_earnings,
        # --- Catalyst context ---
        "iv_percentile": c.iv_percentile,
        "put_call_vol_5d": c.put_call_vol_5d,
        "earnings_surprise_pct": c.earnings_surprise_pct,
        "short_float": c.short_float,
        # --- Scoring ---
        "score_total": sc.total,
        "score_tag_bonus": sc.tag_bonus,
        "tags": list(sc.tags),
        "dimensions": [dataclasses.asdict(d) for d in sc.dimensions],
    }


# ---------------------------------------------------------------------------
# Pipeline thread
# ---------------------------------------------------------------------------

class PipelineThread(QtCore.QThread):
    """
    Background thread that runs the full Trading Assistant pipeline.

    Signals
    -------
    stage_changed(str):
        Emitted at each pipeline stage with a human-readable description.
    candidate_count_changed(int):
        Emitted after scoring with the final candidate count.
    finished_ok(object):
        Emitted on success. Payload is ``list[dict]`` — the scored result rows.
    error(str):
        Emitted on any unhandled exception. Payload is a formatted string
        containing the exception type, message, and full traceback.

    Parameters
    ----------
    csv_dir:
        Directory to scan for Barchart screener CSVs.
        Defaults to ``finance/_data/barchart/screener``.
    trade_date:
        Session date. Defaults to today.
    csv_paths:
        If provided, skips CSV discovery and uses these paths directly
        (manual Load CSV mode).
    skip_ibkr_check:
        If True, skips the IBKR gateway reachability check. Useful in
        tests where no gateway is running.
    """

    stage_changed = QtCore.Signal(str)
    candidate_count_changed = QtCore.Signal(int)
    finished_ok = QtCore.Signal(object)
    error = QtCore.Signal(str)

    def __init__(
        self,
        *,
        csv_dir: Path | None = None,
        trade_date: date | None = None,
        csv_paths: list[Path] | None = None,
        skip_ibkr_check: bool = False,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._csv_dir = Path(csv_dir) if csv_dir else _DEFAULT_CSV_DIR
        self._trade_date = trade_date or date.today()
        self._csv_paths = [Path(p) for p in csv_paths] if csv_paths else None
        self._skip_ibkr_check = skip_ibkr_check

    def run(self) -> None:
        """Execute the pipeline stages. Emits finished_ok or error."""
        try:
            rows = self._run_pipeline()
            self.finished_ok.emit(rows)
        except Exception as exc:
            msg = (
                f"{type(exc).__name__}: {exc}\n\n"
                f"{traceback.format_exc()}"
            )
            log.error("Pipeline failed: %s", msg)
            self.error.emit(msg)

    def _run_pipeline(self) -> list[dict]:
        from finance.apps.analyst._config import load_config
        from finance.apps.analyst._enrichment import enrich
        from finance.apps.analyst._scanner import parse_csv, parse_multiple
        from finance.apps.assistant._models import ScoringConfig
        from finance.apps.assistant._scoring import score_candidate
        from finance.apps.assistant._tags import assign_direction, assign_tags
        from finance.apps.assistant._runner import discover_csvs

        config = load_config()

        # Stage 1: Resolve CSV paths
        if self._csv_paths is not None:
            self.stage_changed.emit("Parsing provided CSV files…")
            csv_map: dict[str, Path] = {}
            all_paths = list(self._csv_paths)
            scanner_sets: dict[str, set[str]] = {}
        else:
            self.stage_changed.emit("Discovering scanner files…")
            csv_map = discover_csvs(self._csv_dir, self._trade_date)
            if not csv_map:
                raise RuntimeError(
                    f"No scanner CSVs found in {self._csv_dir} "
                    f"for {self._trade_date.isoformat()}. "
                    "Download today's Barchart exports first."
                )
            all_paths = list(csv_map.values())

            # Build per-scanner membership sets
            scanner_sets = {}
            for key, path in csv_map.items():
                try:
                    per_scanner = parse_csv(path, config.scanner)
                    scanner_sets[key] = {c.symbol.upper() for c in per_scanner}
                except Exception:
                    log.warning("Failed to parse %s for scanner_sets", path, exc_info=True)

        # Stage 2: Parse + deduplicate
        self.stage_changed.emit(f"Parsing {len(all_paths)} scanner file(s)…")
        candidates = parse_multiple(all_paths, config.scanner)
        log.info("Parsed %d unique candidates", len(candidates))

        if not candidates:
            self.stage_changed.emit("No candidates found.")
            self.candidate_count_changed.emit(0)
            return []

        # Stage 3: IBKR check + enrich
        if not self._skip_ibkr_check:
            self.stage_changed.emit("Checking IBKR Gateway…")
            check_ibkr_gateway()

        self.stage_changed.emit(f"Enriching {len(candidates)} candidates…")
        enriched = enrich(candidates)

        # Stage 4: Tag + direction + score
        self.stage_changed.emit("Scoring candidates…")
        scoring_config = ScoringConfig()
        rows: list[dict] = []
        for ec in enriched:
            tags = assign_tags(ec.candidate, scanner_sets)
            direction = assign_direction(tags)
            sc = score_candidate(ec, direction, tags, scoring_config)
            rows.append(build_result_row(ec, sc))

        rows.sort(key=lambda r: r["score_total"], reverse=True)
        self.candidate_count_changed.emit(len(rows))

        # Stage 5: Write cache
        self.stage_changed.emit("Saving results…")
        path = write_cache(rows, self._trade_date)
        log.info("Cache written → %s (%d rows)", path, len(rows))

        self.stage_changed.emit(f"Done — {len(rows)} candidates scored.")
        return rows
