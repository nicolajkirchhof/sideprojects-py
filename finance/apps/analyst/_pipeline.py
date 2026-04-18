"""
finance.apps.analyst._pipeline
=================================
Pipeline orchestrator — runs stages in order.

Phase 1 (current): Scanner → Enrich → Score → Print report
Phase 2 (future):  + Gmail fetch → Claude analysis → Tradelog push
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

from finance.apps.analyst._config import AnalystConfig, load_config
from finance.apps.analyst._enrichment import enrich
from finance.apps.analyst._models import ScoredCandidate
from finance.apps.analyst._scanner import parse_multiple
from finance.apps.analyst._scoring import score

log = logging.getLogger(__name__)


def run(*, dry_run: bool = False, csv_paths: list[str] | None = None, **_kwargs) -> None:  # noqa: ANN003
    """Run the analyst pipeline.

    Args:
        dry_run: If True, skip Claude API calls and Tradelog push.
        csv_paths: Explicit CSV file paths. If None, uses config csv_directory.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("=== Trade Analyst Pipeline ===")
    config = load_config()

    # Stage 1: Parse scanner CSVs
    paths = _resolve_csv_paths(csv_paths, config)
    if not paths:
        log.warning("No CSV files found. Provide --csv paths or set scanner.csv_directory in config.")
        return

    log.info("Stage 1/3: Parsing %d CSV file(s)...", len(paths))
    candidates = parse_multiple(paths, config.scanner)
    log.info("  → %d unique candidates", len(candidates))

    if not candidates:
        log.info("No candidates to process. Done.")
        return

    # Stage 2: Enrich with IBKR data
    log.info("Stage 2/3: Enriching with IBKR market data...")
    enriched = enrich(candidates)

    # Stage 3: 5-box scoring
    log.info("Stage 3/3: Scoring against 5-box checklist...")
    scored = score(enriched)

    # Output report
    _print_report(scored)
    log.info("Pipeline complete. %d candidates scored.", len(scored))


def _resolve_csv_paths(
    explicit_paths: list[str] | None, config: AnalystConfig,
) -> list[Path]:
    """Resolve CSV file paths from explicit args or config directory.

    When no explicit paths are given, scans the configured directory
    for screener CSVs from the last trading day (skips weekends).
    Pattern: screener-{name}_{MM-DD-YYYY}.csv
    """
    if explicit_paths:
        return [Path(p) for p in explicit_paths if Path(p).exists()]

    csv_dir = Path(config.scanner.csv_directory).expanduser()
    if not csv_dir.is_dir():
        log.warning("Scanner CSV directory does not exist: %s", csv_dir)
        return []

    target_date = _last_trading_day()
    date_suffix = target_date.strftime("%m-%d-%Y")
    prefix = config.scanner.filename_prefix

    pattern = f"{prefix}*_{date_suffix}.csv"
    matches = sorted(csv_dir.glob(pattern))

    if matches:
        log.info("Found %d screener CSV(s) for %s", len(matches), target_date)
    else:
        log.warning("No screener CSVs matching '%s' in %s", pattern, csv_dir)

    return matches


def _last_trading_day(reference: date | None = None) -> date:
    """Return the most recent trading day (Mon-Fri), skipping weekends."""
    d = reference or date.today()
    # If today is a weekend, go back to Friday
    # Otherwise use today (the user runs this after market close or next morning)
    while d.weekday() >= 5:  # 5=Saturday, 6=Sunday
        d -= timedelta(days=1)
    return d


def _print_report(scored: list[ScoredCandidate]) -> None:
    """Print a formatted report of scored candidates to stdout."""
    print("\n" + "=" * 80)
    print(f"  WATCHLIST — {len(scored)} candidates scored")
    print("=" * 80)

    for sc in scored:
        c = sc.enriched.candidate
        data_flag = "" if sc.enriched.data_available else " [NO DATA]"
        print(f"\n  {c.symbol:<8} Score: {sc.score}/5{data_flag}")
        if c.price:
            print(f"           Price: ${c.price:.2f}", end="")
        if c.change_5d_pct is not None:
            print(f"  |  5d: {c.change_5d_pct:+.1f}%", end="")
        if c.change_1m_pct is not None:
            print(f"  |  1M: {c.change_1m_pct:+.1f}%", end="")
        print()

        for box in sc.boxes:
            status_icon = {"PASS": "✓", "FAIL": "✗", "MANUAL": "?"}[box.status]
            print(f"    [{status_icon}] Box {box.box} {box.name}: {box.reason}")

    print("\n" + "=" * 80)
