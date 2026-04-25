"""
build_noon_butterfly_parquet.py
================================
Builds per-symbol noon butterfly Parquet files used by BT-6-S3.

Two-step process:
  Step A — Consolidate existing PKL results files (*_noon_to_close_results_0_1.pkl)
            into a single Parquet file per symbol.
  Step B — Generate new rows for dates not covered by the PKLs.
            IV proxy: HV20 (20-day realised vol annualised) derived from Parquet daily bars.
            Gap rows are flagged with iv_source='hv_proxy' so they can be distinguished
            from PKL rows (iv_source='pkl') where real IV was available.

Output: finance/_data/intraday/noon_butterfly/{SYMBOL}.parquet

Run from repo root:
    python finance/intraday_pm/data/build_noon_butterfly_parquet.py
"""
from __future__ import annotations

import sys
import pickle
from datetime import datetime, timedelta, timezone, date
from math import sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
import blackscholes as bs

from finance.utils.intraday import get_bars
from finance.utils.options import iron_butterfly_profit_loss, risk_free_rate

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PKL_ROOT = "N:/My Drive/Trading/Strategies/noon_to_close"
OUT_DIR = Path("finance/_data/intraday/noon_butterfly")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Mapping: PKL symbol name → Parquet symbol + session noon UTC time
SYMBOL_MAP = {
    "DAX":    {"parquet": "IBDE40",   "noon_utc_hour": 11, "region": "EU", "strike_multiple": 25},
    "ESTX50": {"parquet": "IBES35",   "noon_utc_hour": 11, "region": "EU", "strike_multiple": 5},
    "SPX":    {"parquet": "IBUS500",  "noon_utc_hour": 17, "region": "US", "strike_multiple": 5},
}

SCHEMA = "cfd"
UTC = timezone.utc
HV_WINDOW = 20          # 20-day realised vol for IV proxy
DELTA_CUTOFF = 0.2      # wing selection threshold
T_HALF_DAY = 0.5 / 365  # time to close ≈ half a trading day


# ---------------------------------------------------------------------------
# Step A: consolidate PKLs
# ---------------------------------------------------------------------------
def consolidate_pkl(pkl_symbol: str) -> pd.DataFrame:
    path = f"{PKL_ROOT}/{pkl_symbol}_noon_to_close_results_0_1.pkl"
    try:
        df = pd.read_pickle(path)
    except FileNotFoundError:
        print(f"  PKL not found: {path}")
        return pd.DataFrame()

    df = df.copy()
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")
    df["trade_date"] = pd.to_datetime(df["date_str"]).dt.date
    df["symbol"] = pkl_symbol
    df["iv_source"] = "pkl"
    keep = ["trade_date", "symbol", "pnl", "underlying", "close",
            "wing_call", "wing_put", "atm_call", "atm_put",
            "noon_iv", "close_iv", "iv_source"]
    available = [c for c in keep if c in df.columns]
    return df[available].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step B: generate gap rows using HV proxy
# ---------------------------------------------------------------------------
def _hv20_series(daily_close: pd.Series) -> pd.Series:
    """20-day annualised realised vol from daily log returns."""
    log_ret = np.log(daily_close / daily_close.shift(1))
    hv = log_ret.rolling(HV_WINDOW).std() * sqrt(252)
    hv.index = hv.index.date
    return hv


def _price_butterfly(underlying: float, iv: float, rfr: float, multiple: int, trade_date) -> dict | None:
    """
    Price an iron butterfly at noon using Black-Scholes.
    Returns dict with wing/atm strikes or None if pricing fails.
    """
    iv_day = iv / sqrt(252)
    ul_low = underlying - underlying * 2 * iv_day
    ul_high = underlying + underlying * 2 * iv_day
    low_boundary = int(np.floor(ul_low / multiple) * multiple)
    high_boundary = int(np.ceil(ul_high / multiple) * multiple)
    strikes = list(range(low_boundary, high_boundary, multiple))
    if len(strikes) < 4:
        return None

    opts = []
    for K in strikes:
        try:
            call = bs.BlackScholesCall(underlying, K, T_HALF_DAY, rfr, iv)
            put = bs.BlackScholesPut(underlying, K, T_HALF_DAY, rfr, iv)
            opts.append({"right": "C", "delta": call.delta(), "price": call.price(), "strike": K,
                         "theta": call.theta(), "gamma": call.gamma(), "vega": call.vega(), "pos": 0})
            opts.append({"right": "P", "delta": put.delta(), "price": put.price(), "strike": K,
                         "theta": put.theta(), "gamma": put.gamma(), "vega": put.vega(), "pos": 0})
        except Exception:
            continue

    if not opts:
        return None

    df_opts = pd.DataFrame(opts)
    calls = df_opts[df_opts["right"] == "C"]
    puts = df_opts[df_opts["right"] == "P"]

    wc_mask = calls["delta"] > -DELTA_CUTOFF
    wp_mask = puts["delta"] < DELTA_CUTOFF
    ac_mask = calls["strike"] < underlying
    ap_mask = puts["strike"] > underlying

    if not (wc_mask.any() and wp_mask.any() and ac_mask.any() and ap_mask.any()):
        return None

    wing_call = calls[wc_mask].iloc[-1]
    atm_call = calls[ac_mask].iloc[-1]
    wing_put = puts[wp_mask].iloc[0]
    atm_put = puts[ap_mask].iloc[0]
    return {
        "wing_call": wing_call,
        "atm_call": atm_call,
        "atm_put": atm_put,
        "wing_put": wing_put,
    }


def generate_gap_rows(pkl_symbol: str, cfg: dict, start_date: date, end_date: date) -> pd.DataFrame:
    parquet_sym = cfg["parquet"]
    noon_utc_hour = cfg["noon_utc_hour"]
    region = cfg["region"]
    strike_multiple = cfg["strike_multiple"]

    # Load daily bars for HV computation (need lookback)
    load_start = datetime.combine(start_date - timedelta(days=30), datetime.min.time(), tzinfo=UTC)
    load_end = datetime.combine(end_date + timedelta(days=1), datetime.min.time(), tzinfo=UTC)

    daily_raw = get_bars(parquet_sym, SCHEMA, load_start, load_end, period="1D")
    if daily_raw.empty:
        return pd.DataFrame()
    daily_raw = daily_raw.rename(columns={"c": "close"})
    hv_series = _hv20_series(daily_raw["close"])

    # Load 5-min bars to get noon and close prices
    df5 = get_bars(parquet_sym, SCHEMA, load_start, load_end, period="5min")
    if df5.empty:
        return pd.DataFrame()
    df5 = df5.rename(columns={"c": "close"})

    rows: list[dict] = []
    trading_dates = sorted(set(
        d for d in df5.index.date
        if start_date <= d < end_date
    ))

    for trade_date in trading_dates:
        # Noon price: last bar at or before noon UTC
        noon_mask = (df5.index.date == trade_date) & (df5.index.hour <= noon_utc_hour)
        if not noon_mask.any():
            continue
        noon_bar = df5[noon_mask].iloc[-1]
        noon_price = float(noon_bar["close"])

        # Session close price: last bar of the day
        day_mask = df5.index.date == trade_date
        day_bars = df5[day_mask]
        if day_bars.empty:
            continue
        close_price = float(day_bars.iloc[-1]["close"])

        hv_val = hv_series.get(trade_date, np.nan)
        if pd.isna(hv_val) or hv_val <= 0:
            continue

        try:
            rfr = risk_free_rate(
                pd.Timestamp(trade_date), region
            )
        except Exception:
            rfr = 0.02  # fallback

        butterfly = _price_butterfly(noon_price, hv_val, rfr, strike_multiple, trade_date)
        if butterfly is None:
            continue

        try:
            pnl = iron_butterfly_profit_loss(
                close_price,
                butterfly["wing_call"],
                butterfly["atm_call"],
                butterfly["atm_put"],
                butterfly["wing_put"],
            )
        except Exception:
            continue

        rows.append({
            "trade_date": trade_date,
            "symbol": pkl_symbol,
            "pnl": float(pnl),
            "underlying": noon_price,
            "close": close_price,
            "wing_call": butterfly["wing_call"]["strike"],
            "wing_put": butterfly["wing_put"]["strike"],
            "atm_call": butterfly["atm_call"]["strike"],
            "atm_put": butterfly["atm_put"]["strike"],
            "noon_iv": hv_val,
            "close_iv": np.nan,
            "iv_source": "hv_proxy",
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_symbol(pkl_symbol: str, cfg: dict) -> None:
    out_path = OUT_DIR / f"{pkl_symbol}.parquet"

    # Step A
    print(f"  [A] Consolidating PKL...")
    df_pkl = consolidate_pkl(pkl_symbol)
    if df_pkl.empty:
        print(f"  [A] No PKL data")
        return
    last_pkl = max(df_pkl["trade_date"])
    print(f"  [A] {len(df_pkl)} rows, last={last_pkl}")

    # Step B
    gap_start = last_pkl + timedelta(days=1)
    gap_end = date.today()

    if gap_start < gap_end:
        print(f"  [B] Generating gap {gap_start} -> {gap_end} with HV proxy...")
        df_new = generate_gap_rows(pkl_symbol, cfg, gap_start, gap_end)
        if not df_new.empty:
            print(f"  [B] {len(df_new)} new rows ({df_new['trade_date'].nunique()} days)")
            df_all = pd.concat([df_pkl, df_new], ignore_index=True)
        else:
            print(f"  [B] No new rows")
            df_all = df_pkl
    else:
        df_all = df_pkl

    df_all = df_all.sort_values("trade_date").reset_index(drop=True)
    df_all.to_parquet(out_path, index=False)
    print(f"  Written: {out_path} ({len(df_all)} rows)")


def main() -> None:
    for pkl_symbol, cfg in SYMBOL_MAP.items():
        print(f"\n{pkl_symbol}")
        build_symbol(pkl_symbol, cfg)


if __name__ == "__main__":
    main()
