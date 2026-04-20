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
from finance.apps.analyst._models import Candidate, OptionsContract, UoaSignal

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


# --- Options screener parsing ---

def parse_options_csv(path: Path, column_mapping: dict[str, str]) -> list[OptionsContract]:
    """Parse a Barchart options screener CSV into OptionsContract objects."""
    try:
        df = pd.read_csv(path)
    except Exception:
        log.warning("Failed to read options CSV: %s", path)
        return []

    if df.empty:
        return []

    reverse_map: dict[str, str] = {}
    for csv_col, internal_name in column_mapping.items():
        if csv_col in df.columns:
            reverse_map[internal_name] = csv_col

    if "symbol" not in reverse_map:
        log.warning("No symbol column mapped in options CSV %s", path.name)
        return []

    contracts: list[OptionsContract] = []
    for _, row in df.iterrows():
        raw_symbol = str(row[reverse_map["symbol"]]).strip()
        # Options symbol format varies — extract underlying (letters before digits/spaces)
        underlying = _extract_underlying(raw_symbol)
        if not underlying:
            continue

        contracts.append(OptionsContract(
            symbol=underlying,
            underlying_price=_parse_float(row, reverse_map, "underlying_price"),
            iv_percentile=_parse_float(row, reverse_map, "iv_percentile"),
            implied_vol=_parse_float(row, reverse_map, "implied_vol"),
            iv_chg_1d=_parse_float(row, reverse_map, "iv_chg_1d"),
            iv_chg_5d=_parse_float(row, reverse_map, "iv_chg_5d"),
            option_type=_parse_str(row, reverse_map, "option_type") or "",
            strike=_parse_float(row, reverse_map, "strike"),
            expiration=_parse_str(row, reverse_map, "expiration"),
            delta=_parse_float(row, reverse_map, "delta"),
            moneyness=_parse_str(row, reverse_map, "moneyness"),
            vol_oi_ratio=_parse_float(row, reverse_map, "vol_oi_ratio"),
            volume=_parse_float(row, reverse_map, "volume"),
            vol_pct_chg=_parse_float(row, reverse_map, "vol_pct_chg"),
            open_interest=_parse_float(row, reverse_map, "open_interest"),
            oi_pct_chg=_parse_float(row, reverse_map, "oi_pct_chg"),
            theta=_parse_float(row, reverse_map, "theta"),
            expires_before_earnings=_parse_str(row, reverse_map, "expires_before_earnings"),
        ))

    log.info("Parsed %d option contracts from %s", len(contracts), path.name)
    return contracts


def aggregate_uoa(contracts: list[OptionsContract]) -> dict[str, UoaSignal]:
    """Aggregate option contracts into per-underlying UOA signals."""
    by_symbol: dict[str, list[OptionsContract]] = {}
    for c in contracts:
        by_symbol.setdefault(c.symbol, []).append(c)

    signals: dict[str, UoaSignal] = {}
    for symbol, group in by_symbol.items():
        calls = [c for c in group if c.option_type.lower() == "call"]
        puts = [c for c in group if c.option_type.lower() == "put"]
        max_vol_oi = max((c.vol_oi_ratio or 0) for c in group)

        call_deltas = [c.delta for c in calls if c.delta is not None]
        avg_delta_calls = sum(call_deltas) / len(call_deltas) if call_deltas else None

        iv_pctl = next((c.iv_percentile for c in group if c.iv_percentile is not None), None)

        signals[symbol] = UoaSignal(
            symbol=symbol,
            call_count=len(calls),
            put_count=len(puts),
            max_vol_oi=max_vol_oi,
            avg_delta_calls=avg_delta_calls,
            iv_percentile=iv_pctl,
            contracts=group,
        )

    return signals


def _extract_underlying(symbol_str: str) -> str:
    """Extract the underlying ticker from a Barchart options symbol.

    Barchart options symbols look like 'AAPL|20260515|200.00C' or just 'AAPL'.
    The underlying is the part before the first pipe, digit, or space.
    """
    import re
    match = re.match(r"([A-Z]+)", symbol_str.upper())
    return match.group(1) if match else ""
