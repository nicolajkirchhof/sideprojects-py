"""
finance.apps.analyst._scanner
===============================
Barchart scanner CSV parser with configurable column mapping.

Reads one or more CSV files, maps columns to internal field names,
deduplicates by symbol, and returns a list of Candidate objects.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from finance.apps.analyst._config import ScannerConfig
from finance.apps.analyst._models import Candidate

log = logging.getLogger(__name__)


def parse_csv(path: Path, config: ScannerConfig) -> list[Candidate]:
    """Parse a single Barchart scanner CSV into Candidate objects."""
    try:
        df = pd.read_csv(path)
    except Exception:
        log.warning("Failed to read CSV: %s", path)
        return []

    if df.empty:
        log.info("Empty CSV: %s", path)
        return []

    # Build reverse mapping: internal_name -> csv_column
    reverse_map: dict[str, str] = {}
    for csv_col, internal_name in config.column_mapping.items():
        if csv_col in df.columns:
            reverse_map[internal_name] = csv_col
        else:
            log.debug("Column '%s' not found in %s", csv_col, path.name)

    if "symbol" not in reverse_map:
        log.warning("No symbol column mapped in %s", path.name)
        return []

    # Determine which Candidate fields are str vs numeric
    import dataclasses
    str_fields = {f.name for f in dataclasses.fields(Candidate)
                  if f.type in ("str | None", "str")} - {"symbol"}
    skip_fields = {"symbol"}

    candidates: list[Candidate] = []
    for _, row in df.iterrows():
        symbol = str(row[reverse_map["symbol"]]).strip().upper()
        if not symbol or symbol == "NAN":
            continue

        kwargs: dict = {"symbol": symbol}
        for field_name in reverse_map:
            if field_name in skip_fields:
                continue
            if field_name in str_fields:
                kwargs[field_name] = _parse_str(row, reverse_map, field_name)
            elif field_name in config.percent_columns:
                kwargs[field_name] = _parse_percent(row, reverse_map, field_name, config)
            else:
                kwargs[field_name] = _parse_float(row, reverse_map, field_name)

        candidates.append(Candidate(**kwargs))

    log.info("Parsed %d candidates from %s", len(candidates), path.name)
    return candidates


def parse_multiple(paths: list[Path], config: ScannerConfig) -> list[Candidate]:
    """Parse multiple CSVs and deduplicate by symbol (keep first occurrence)."""
    all_candidates: list[Candidate] = []
    for path in paths:
        all_candidates.extend(parse_csv(path, config))
    return deduplicate(all_candidates)


def deduplicate(candidates: list[Candidate]) -> list[Candidate]:
    """Deduplicate candidates by symbol, keeping the first occurrence."""
    seen: set[str] = set()
    result: list[Candidate] = []
    for c in candidates:
        if c.symbol not in seen:
            seen.add(c.symbol)
            result.append(c)
    return result


def _parse_float(
    row: pd.Series, mapping: dict[str, str], field: str,
) -> float | None:
    if field not in mapping:
        return None
    val = row[mapping[field]]
    if isinstance(val, str):
        val = val.replace(",", "").strip()
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _parse_percent(
    row: pd.Series,
    mapping: dict[str, str],
    field: str,
    config: ScannerConfig,
) -> float | None:
    """Parse a value that may have a '%' suffix."""
    if field not in mapping:
        return None
    val = row[mapping[field]]
    if isinstance(val, str) and field in config.percent_columns:
        val = val.replace("%", "").replace(",", "").strip()
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _parse_str(
    row: pd.Series, mapping: dict[str, str], field: str,
) -> str | None:
    if field not in mapping:
        return None
    val = row[mapping[field]]
    if pd.isna(val):
        return None
    return str(val).strip()
