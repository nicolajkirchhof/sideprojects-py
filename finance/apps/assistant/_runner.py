"""
finance.apps.assistant._runner
================================
Headless archive pipeline — no Qt, no Claude, no Tradelog.

Stages:
  1. Discover today's scanner CSVs from the screener data directory
  2. Parse each file → build per-scanner membership sets for tag assignment
  3. parse_multiple → deduplicated list[Candidate]
  4. enrich → list[EnrichedCandidate]
  5. Assign tags + direction + score each candidate
  6. candidates_to_df → candidates DataFrame
  7. load_all_market → market DataFrame
  8. Concatenate + write_archive → YYYY-MM-DD.parquet

Usage:
    python -m finance.apps assistant --archive
    or call run_archive() directly.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scanner key → filename fragment mapping
# ---------------------------------------------------------------------------

#: Maps the fragment that appears in the screener filename to the scanner key
#: used in _tags.py membership sets.
SCREENER_KEY_MAP: dict[str, str] = {
    "long-universe":          "long-universe",
    "pead-scanner":           "pead-scanner",
    "ep-gap-scanner":         "ep-gap-scanner",
    "rw-breakdown-candidates": "rw-breakdown-candidates",
    "short-squeeze":          "short-squeeze",
    "high-put-ratio":         "high-put-ratio",
    "high-call-ratio":        "high-call-ratio",
}

_DEFAULT_CSV_DIR = Path("finance/_data/barchart/screener")


# ---------------------------------------------------------------------------
# CSV discovery
# ---------------------------------------------------------------------------

def discover_csvs(csv_dir: Path, trade_date: date) -> dict[str, Path]:
    """
    Glob csv_dir for screener files matching today's date pattern.

    Files follow the naming convention:
        stocks-screener-{scanner-key}-MM-DD-YYYY.csv

    Parameters
    ----------
    csv_dir:
        Directory containing Barchart screener CSV exports.
    trade_date:
        Session date — used to filter files by date suffix.

    Returns
    -------
    dict[str, Path]
        Mapping of scanner key → file path for files found today.
    """
    date_suffix = trade_date.strftime("%m-%d-%Y")
    found: dict[str, Path] = {}
    for path in csv_dir.glob(f"stocks-screener-*-{date_suffix}.csv"):
        stem = path.stem  # e.g. stocks-screener-long-universe-04-22-2026
        for fragment, key in SCREENER_KEY_MAP.items():
            if fragment in stem:
                found[key] = path
                break
    return found


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_archive(
    *,
    csv_dir: Path | None = None,
    trade_date: date | None = None,
    dry_run: bool = False,
) -> Path:
    """
    Run the headless archive pipeline and write today's Parquet file.

    Parameters
    ----------
    csv_dir:
        Directory containing Barchart screener CSVs.
        Defaults to finance/_data/barchart/screener/.
    trade_date:
        Session date. Defaults to today.
    dry_run:
        If True, skips writing the archive file but returns the path
        that would have been written.

    Returns
    -------
    Path
        The archive path (written or would-have-been-written).

    Raises
    ------
    RuntimeError
        If no scanner CSVs are found for the given date.
    """
    import pandas as pd

    from finance.apps.analyst._config import load_config
    from finance.apps.analyst._enrichment import enrich
    from finance.apps.analyst._scanner import parse_csv, parse_multiple
    from finance.apps.assistant._archive import candidates_to_df, write_archive, archive_path
    from finance.apps.assistant._market import load_all_market
    from finance.apps.assistant._models import ScoringConfig
    from finance.apps.assistant._scoring import score_candidate
    from finance.apps.assistant._tags import assign_direction, assign_tags

    csv_dir = Path(csv_dir) if csv_dir else _DEFAULT_CSV_DIR
    trade_date = trade_date or date.today()

    log.info("=== Trading Assistant Archive Pipeline ===")
    log.info("Session date: %s", trade_date.isoformat())
    log.info("CSV directory: %s", csv_dir)

    # Stage 1: Discover CSVs
    csv_map = discover_csvs(csv_dir, trade_date)
    if not csv_map:
        raise RuntimeError(
            f"No scanner CSVs found in {csv_dir} for date {trade_date.isoformat()}. "
            "Ensure today's Barchart exports are present."
        )
    log.info("Stage 1: Found %d scanner file(s): %s", len(csv_map), list(csv_map.keys()))

    # Stage 2: Build per-scanner membership sets for tag assignment
    config = load_config()
    scanner_sets: dict[str, set[str]] = {}
    for key, path in csv_map.items():
        try:
            per_scanner = parse_csv(path, config.scanner)
            scanner_sets[key] = {c.symbol.upper() for c in per_scanner}
        except Exception:
            log.warning("Failed to parse %s for scanner_sets — skipping", path, exc_info=True)

    # Stage 3: Parse all CSVs → deduplicated candidates
    all_paths = list(csv_map.values())
    candidates = parse_multiple(all_paths, config.scanner)
    log.info("Stage 3: %d unique candidates after deduplication", len(candidates))

    if not candidates:
        log.warning("No candidates to process.")

    # Stage 4: IBKR enrichment
    log.info("Stage 4: Enriching with IBKR data...")
    enriched = enrich(candidates)

    # Stage 5: Tag + direction + score
    log.info("Stage 5: Assigning tags, direction, and scores...")
    scoring_config = ScoringConfig()
    scores = []
    for ec in enriched:
        tags = assign_tags(ec.candidate, scanner_sets)
        direction = assign_direction(tags)
        sc = score_candidate(ec, direction, tags, scoring_config)
        scores.append(sc)

    # Stage 6: Candidates DataFrame
    candidates_df = candidates_to_df(enriched, scores, trade_date=trade_date)
    log.info("Stage 6: %d candidate rows built", len(candidates_df))

    # Stage 7: Market data
    log.info("Stage 7: Loading market context instruments...")
    market_df = load_all_market()
    market_df["date"] = trade_date.isoformat()
    log.info("  → %d market rows loaded", len(market_df))

    # Stage 8: Combine and write
    combined = pd.concat([candidates_df, market_df], ignore_index=True)
    path = archive_path(trade_date)

    if dry_run:
        log.info("Stage 8: DRY RUN — would write %d rows to %s", len(combined), path)
        return path

    written = write_archive(combined, trade_date)
    log.info(
        "Stage 8: Archive written → %s (%d candidates, %d market rows)",
        written, len(candidates_df), len(market_df),
    )
    return written
