"""
finance.apps.assistant._pipeline
==================================
Background pipeline thread and JSON cache I/O for the Trading Assistant.

Pipeline stages (PipelineThread.run):
  1. Fetch screener CSVs from Gmail (label: TradeAnalyst) → staging dir
     Manual mode (csv_paths provided): skip Gmail, use supplied paths directly
  2. Parse each CSV → deduplicated list[Candidate]
  3. IBKR enrichment (uses local Parquet data; gateway check runs before stage 3)
  4. Assign tags + direction + score each candidate
  5. Fetch economic calendar (ForexFactory, 5-day window)
  6. Write JSON cache → _data/assistant/YYYY-MM-DD.json

Error policy: any unhandled exception emits the ``error`` signal with full
traceback and stops the thread immediately. The caller is responsible for
showing the error dialog and re-enabling the UI.

Cache format:
    {
        "date": "YYYY-MM-DD",
        "created_at": "YYYY-MM-DDTHH:MM:SS",
        "rows": [ { ...result row... }, ... ],
        "events": [ { ...economic event dict... }, ... ]   # optional
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

from finance.apps.assistant._claude import analyze_candidate, summarize_market

if TYPE_CHECKING:
    from finance.apps.analyst._models import EnrichedCandidate
    from finance.apps.assistant._models import CandidateScore
    from finance.apps.assistant._gmail import EmailBody

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
    events: list[dict] | None = None,
    market_summary: dict | None = None,
) -> Path:
    """
    Write *rows* (and optional *events* / *market_summary*) to the daily JSON cache file.

    Returns the path that was written.
    """
    path = cache_path(trade_date, base_dir=base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "date": trade_date.isoformat(),
        "created_at": datetime.now().replace(microsecond=0).isoformat(),
        "rows": rows,
        "events": events or [],
        "market_summary": market_summary,
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


def read_events_from_cache(
    trade_date: date,
    *,
    base_dir: Path | None = None,
) -> list[dict] | None:
    """
    Load economic events from the daily JSON cache if they exist.

    Returns
    -------
    list[dict] | None
        The ``events`` list, or ``None`` if no cache file exists or the
        cache was written before events support was added.
    """
    path = cache_path(trade_date, base_dir=base_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        events = payload.get("events")
        return events if events is not None else None
    except Exception:
        log.warning("Failed to read events cache %s — treating as missing", path, exc_info=True)
        return None


def update_cache_market_summary(
    trade_date: date,
    summary: dict,
    *,
    base_dir: Path | None = None,
) -> None:
    """
    Patch the ``market_summary`` field in an existing cache file.

    No-op if no cache file exists for *trade_date*.
    """
    path = cache_path(trade_date, base_dir=base_dir)
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["market_summary"] = summary
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        log.warning("Failed to patch market_summary in cache %s", path, exc_info=True)


def read_market_summary_from_cache(
    trade_date: date,
    *,
    base_dir: Path | None = None,
) -> dict | None:
    """
    Load the Claude market summary dict from the daily JSON cache if present.

    Returns
    -------
    dict | None
        The ``market_summary`` dict, or ``None`` if no cache exists, the
        field is absent (old cache format), or the file cannot be read.
    """
    path = cache_path(trade_date, base_dir=base_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload.get("market_summary") or None
    except Exception:
        log.warning("Failed to read market_summary from cache %s", path, exc_info=True)
        return None


def update_cache_candidate_analysis(
    trade_date: date,
    symbol: str,
    analysis: dict,
    *,
    base_dir: Path | None = None,
) -> None:
    """
    Patch ``candidate_analyses[symbol]`` in an existing cache file.

    No-op if no cache file exists for *trade_date*.
    """
    path = cache_path(trade_date, base_dir=base_dir)
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        analyses = payload.setdefault("candidate_analyses", {})
        analyses[symbol] = analysis
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        log.warning("Failed to patch candidate_analyses in cache %s", path, exc_info=True)


def read_candidate_analysis_from_cache(
    trade_date: date,
    symbol: str,
    *,
    base_dir: Path | None = None,
) -> dict | None:
    """
    Load the Claude analysis dict for *symbol* from the daily JSON cache.

    Returns
    -------
    dict | None
        The analysis dict, or ``None`` if absent.
    """
    path = cache_path(trade_date, base_dir=base_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        analyses = payload.get("candidate_analyses") or {}
        return analyses.get(symbol) or None
    except Exception:
        log.warning("Failed to read candidate_analyses from cache %s", path, exc_info=True)
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
    calendar_updated = QtCore.Signal(list)
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
            # Fetch screener CSVs from Gmail, then map by scanner key.
            self.stage_changed.emit("Fetching screener CSVs from Gmail…")
            from finance.apps.assistant._gmail import fetch_screener_csvs
            downloaded = fetch_screener_csvs(
                config.gmail,
                staging_dir=self._csv_dir,
                trade_date=self._trade_date,
            )
            if not downloaded:
                raise RuntimeError(
                    f"No screener emails found in Gmail label '{config.gmail.label}' "
                    f"for {self._trade_date.isoformat()}. "
                    "Check that today's Barchart exports were sent and the label is correct."
                )

            self.stage_changed.emit(f"Mapping {len(downloaded)} CSV file(s) to scanner keys…")
            csv_map = discover_csvs(self._csv_dir, self._trade_date)
            all_paths = list(csv_map.values()) if csv_map else downloaded

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

        # Stage 5: Fetch economic calendar
        self.stage_changed.emit("Fetching economic calendar…")
        event_dicts: list[dict] = []
        try:
            from finance.apps.assistant._calendar import events_to_dicts, fetch_upcoming_events
            events = fetch_upcoming_events(days_ahead=5, impact_filter="Medium")
            event_dicts = events_to_dicts(events)
            self.calendar_updated.emit(event_dicts)
            log.info("Fetched %d calendar events", len(event_dicts))
        except Exception:
            log.warning("Calendar fetch failed — continuing without events", exc_info=True)

        # Stage 6: Write cache
        self.stage_changed.emit("Saving results…")
        path = write_cache(rows, self._trade_date, events=event_dicts)
        log.info("Cache written → %s (%d rows)", path, len(rows))

        self.stage_changed.emit(f"Done — {len(rows)} candidates scored.")
        return rows


# ---------------------------------------------------------------------------
# Claude summary thread
# ---------------------------------------------------------------------------


class ClaudeSummaryThread(QtCore.QThread):
    """
    Background thread that calls ``summarize_market()`` and emits the result.

    Signals
    -------
    summary_ready(object):
        Emitted on success. Payload is ``dict`` — dataclasses.asdict of the
        MarketSummary returned by summarize_market().
    error(str):
        Emitted on any unhandled exception. Payload is a formatted string
        containing the exception type, message, and full traceback.

    Parameters
    ----------
    emails:
        EmailBody instances to summarise.
    model:
        Claude model ID (e.g. ``"claude-sonnet-4-6"``).
    """

    summary_ready = QtCore.Signal(object)
    error = QtCore.Signal(str)

    def __init__(
        self,
        *,
        emails: list[EmailBody],
        model: str,
        trade_date: date | None = None,
        spy_status: object | None = None,
        qqq_status: object | None = None,
        vix_status: object | None = None,
        regime_status: str = "",
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._emails = emails
        self._model = model
        self._trade_date = trade_date or date.today()
        self._spy_status = spy_status
        self._qqq_status = qqq_status
        self._vix_status = vix_status
        self._regime_status = regime_status

    def run(self) -> None:
        """Call summarize_market (with regime + web context) and emit summary_ready or error."""
        try:
            from finance.apps.assistant._claude import fetch_market_context, format_regime_context
            regime_block = format_regime_context(
                self._spy_status, self._qqq_status, self._vix_status, self._regime_status
            )
            web_block = fetch_market_context(self._trade_date)
            market_context = "\n\n".join(filter(None, [regime_block, web_block]))
            result = summarize_market(self._emails, self._model, market_context=market_context)
            self.summary_ready.emit(dataclasses.asdict(result))
        except Exception as exc:
            msg = (
                f"{type(exc).__name__}: {exc}\n\n"
                f"{traceback.format_exc()}"
            )
            log.error("Claude summary failed: %s", msg)
            self.error.emit(msg)


# ---------------------------------------------------------------------------
# Candidate analysis thread
# ---------------------------------------------------------------------------


class CandidateAnalysisThread(QtCore.QThread):
    """
    Background thread that calls ``analyze_candidate()`` for a single row.

    Signals
    -------
    analysis_ready(object):
        Emitted on success. Payload is ``dict`` — dataclasses.asdict of the
        CandidateAnalysis returned by analyze_candidate().
    error(str):
        Emitted on any unhandled exception.

    Parameters
    ----------
    row:
        Result row dict to analyse.
    model:
        Claude model ID.
    """

    analysis_ready = QtCore.Signal(object)
    error = QtCore.Signal(str)

    def __init__(
        self,
        *,
        row: dict,
        model: str,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._row = row
        self._model = model

    def run(self) -> None:
        """Call analyze_candidate and emit analysis_ready or error."""
        try:
            result = analyze_candidate(self._row, self._model)
            self.analysis_ready.emit(dataclasses.asdict(result))
        except Exception as exc:
            msg = (
                f"{type(exc).__name__}: {exc}\n\n"
                f"{traceback.format_exc()}"
            )
            log.error("Candidate analysis failed: %s", msg)
            self.error.emit(msg)


# ---------------------------------------------------------------------------
# Top-N auto analysis thread
# ---------------------------------------------------------------------------


class TopNAnalysisThread(QtCore.QThread):
    """
    Background thread that sequentially analyses the top-N candidates.

    Signals
    -------
    row_analysis_ready(str, object):
        Emitted after each successful analysis. Payload is (symbol, analysis_dict).
    error(str):
        Emitted if an unhandled exception stops the thread. Per-row API
        failures are logged but do not stop the loop.

    Parameters
    ----------
    rows:
        Full list of result rows (sorted by score descending).
    model:
        Claude model ID.
    top_n:
        Number of top rows to analyse.
    """

    row_analysis_ready = QtCore.Signal(str, object)
    error = QtCore.Signal(str)

    def __init__(
        self,
        *,
        rows: list[dict],
        model: str,
        top_n: int,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._rows = rows
        self._model = model
        self._top_n = top_n

    def run(self) -> None:
        """Analyse rows[:top_n] sequentially, emitting per-row results."""
        try:
            for row in self._rows[: self._top_n]:
                symbol = row.get("symbol", "")
                try:
                    result = analyze_candidate(row, self._model)
                    self.row_analysis_ready.emit(symbol, dataclasses.asdict(result))
                except Exception:
                    log.warning("Auto-analysis failed for %s", symbol, exc_info=True)
        except Exception as exc:
            msg = (
                f"{type(exc).__name__}: {exc}\n\n"
                f"{traceback.format_exc()}"
            )
            log.error("TopNAnalysisThread failed: %s", msg)
            self.error.emit(msg)
