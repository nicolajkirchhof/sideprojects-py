"""
Core PM Backtest Engine

Validates short premium strategies using daily price + IV + HV data.
No options chain data needed — uses delta-approximation from Black-Scholes.

Supports: short puts, strangles, iron condors (approximate).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy.stats import norm


# ── Constants ────────────────────────────────────────────

DEFAULT_DTE = 45
TRADING_DAYS_PER_YEAR = 252
DELTA_ENTRY = 0.25
DELTA_STOP = 0.50
RISK_FREE_RATE = 0.045  # approximate, adjust as needed


# ── Data Structures ──────────────────────────────────────

@dataclass
class TradeResult:
    """Result of a single simulated trade."""
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    underlying: str
    structure: str  # 'short_put', 'strangle', 'iron_condor'
    entry_price: float
    exit_price: float
    iv_at_entry: float
    hv_at_entry: float
    ivp_at_entry: float
    forward_rv: float
    vrp: float  # IV - forward RV
    put_strike: float
    call_strike: float | None
    credit_received: float
    pnl: float
    win: bool
    stopped_out: bool
    above_200sma: bool
    dte: int
    regime: str  # 'low_iv', 'mid_iv', 'high_iv'


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""
    dte: int = DEFAULT_DTE
    delta_entry: float = DELTA_ENTRY
    delta_stop: float = DELTA_STOP
    entry_interval_days: int = 14  # new trade every N days
    ivp_filter: float | None = None  # None = no filter, e.g. 50.0
    sma_filter: bool = False  # require above 200d SMA
    structure: str = 'short_put'  # 'short_put', 'strangle', 'iron_condor'
    call_delta: float = 0.25  # for strangles
    ic_wing_width: float = 0.05  # iron condor wing width as % of underlying
    stop_at_delta: float = 0.50  # close when estimated delta reaches this


# ── Black-Scholes Helpers ────────────────────────────────

def bs_put_strike(spot: float, iv: float, dte: int, delta: float = 0.25) -> float:
    """Approximate the strike price for an OTM put at a given |delta|.
    A 25-delta put is below the current spot price."""
    t = dte / TRADING_DAYS_PER_YEAR
    if iv <= 0 or t <= 0:
        return spot * 0.90
    # For a put with |delta| = 0.25, we need N(d1) = 1 - delta = 0.75
    # so d1 = norm.ppf(0.75), and strike = spot * exp(-d1 * sigma * sqrt(t) + 0.5 * sigma^2 * t)
    # But we want OTM, so strike < spot. Use the direct formula:
    z = norm.ppf(delta)  # negative for delta < 0.5
    strike = spot * np.exp(z * iv * np.sqrt(t) - 0.5 * iv**2 * t)
    return strike


def bs_call_strike(spot: float, iv: float, dte: int, delta: float = 0.25) -> float:
    """Approximate the strike price for an OTM call at a given delta.
    A 25-delta call is above the current spot price."""
    t = dte / TRADING_DAYS_PER_YEAR
    if iv <= 0 or t <= 0:
        return spot * 1.10
    z = norm.ppf(1 - delta)  # positive for delta < 0.5
    strike = spot * np.exp(z * iv * np.sqrt(t) + 0.5 * iv**2 * t)
    return strike


def bs_put_price(spot: float, strike: float, iv: float, dte: int) -> float:
    """Black-Scholes put price approximation."""
    t = dte / TRADING_DAYS_PER_YEAR
    if iv <= 0 or t <= 0:
        return max(strike - spot, 0)
    d1 = (np.log(spot / strike) + (RISK_FREE_RATE + 0.5 * iv**2) * t) / (iv * np.sqrt(t))
    d2 = d1 - iv * np.sqrt(t)
    price = strike * np.exp(-RISK_FREE_RATE * t) * norm.cdf(-d2) - spot * norm.cdf(-d1)
    return max(price, 0)


def bs_call_price(spot: float, strike: float, iv: float, dte: int) -> float:
    """Black-Scholes call price approximation."""
    t = dte / TRADING_DAYS_PER_YEAR
    if iv <= 0 or t <= 0:
        return max(spot - strike, 0)
    d1 = (np.log(spot / strike) + (RISK_FREE_RATE + 0.5 * iv**2) * t) / (iv * np.sqrt(t))
    d2 = d1 - iv * np.sqrt(t)
    price = spot * norm.cdf(d1) - strike * np.exp(-RISK_FREE_RATE * t) * norm.cdf(d2)
    return max(price, 0)


def bs_put_delta(spot: float, strike: float, iv: float, dte: int) -> float:
    """Black-Scholes put delta (negative value)."""
    t = dte / TRADING_DAYS_PER_YEAR
    if iv <= 0 or t <= 0:
        return -1.0 if spot < strike else 0.0
    d1 = (np.log(spot / strike) + (RISK_FREE_RATE + 0.5 * iv**2) * t) / (iv * np.sqrt(t))
    return norm.cdf(d1) - 1  # negative


# ── Forward Realized Vol ─────────────────────────────────

def forward_realized_vol(prices: pd.Series, start_idx: int, window: int) -> float:
    """Calculate annualized realized vol over the forward window from start_idx."""
    end_idx = start_idx + window
    if end_idx >= len(prices):
        return np.nan
    window_prices = prices.iloc[start_idx:end_idx + 1]
    if len(window_prices) < 2:
        return np.nan
    log_returns = np.log(window_prices / window_prices.shift(1)).dropna()
    if len(log_returns) < 5:
        return np.nan
    return log_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


# ── VRP Analysis ─────────────────────────────────────────

def compute_vrp(df: pd.DataFrame, windows: list[int] = None) -> pd.DataFrame:
    """
    Compute variance risk premium for each day.
    VRP = IV at entry - forward realized vol over the window.
    Positive VRP = IV overestimates RV = edge for premium sellers.
    """
    if windows is None:
        windows = [30, 45, 60]

    prices = df['c']
    results = df[['c', 'ivc', 'hvc', 'iv_pct']].copy()

    for w in windows:
        fwd_rv = pd.Series(index=df.index, dtype=float, name=f'fwd_rv_{w}d')
        for i in range(len(df)):
            fwd_rv.iloc[i] = forward_realized_vol(prices, i, w)
        results[f'fwd_rv_{w}d'] = fwd_rv
        results[f'vrp_{w}d'] = results['ivc'] - fwd_rv

    return results.dropna(subset=['ivc'])


# ── Range Test ───────────────────────────────────────────

def compute_range_test(df: pd.DataFrame, dte: int = 45) -> pd.DataFrame:
    """
    For each day, check if price stayed within ±1 SD (IV-implied) over the next DTE days.
    This approximates a 25Δ strangle win rate.
    """
    prices = df['c'].values
    iv = df['ivc'].values
    n = len(df)
    results = []

    for i in range(n - dte):
        if np.isnan(iv[i]) or iv[i] <= 0:
            continue

        spot = prices[i]
        sd_move = spot * iv[i] * np.sqrt(dte / TRADING_DAYS_PER_YEAR)
        upper = spot + sd_move
        lower = spot - sd_move

        window_prices = prices[i + 1:i + dte + 1]
        max_price = np.max(window_prices)
        min_price = np.min(window_prices)

        stayed_in_range = (min_price >= lower) and (max_price <= upper)
        put_breached = min_price < lower
        call_breached = max_price > upper

        results.append({
            'date': df.index[i],
            'spot': spot,
            'iv': iv[i],
            'ivp': df['iv_pct'].iloc[i] if 'iv_pct' in df.columns else np.nan,
            'upper': upper,
            'lower': lower,
            'max_in_window': max_price,
            'min_in_window': min_price,
            'stayed_in_range': stayed_in_range,
            'put_breached': put_breached,
            'call_breached': call_breached,
            'above_200sma': prices[i] > df['ma200'].iloc[i] if 'ma200' in df.columns else True,
            'final_price': prices[i + dte],
        })

    return pd.DataFrame(results)


# ── Trade Simulation ─────────────────────────────────────

def simulate_trades(df: pd.DataFrame, config: BacktestConfig) -> list[TradeResult]:
    """
    Simulate short premium trades on the provided daily data.
    Enters a new trade every config.entry_interval_days.
    """
    prices = df['c'].values
    ivs = df['ivc'].values
    hvs = df['hvc'].values
    ivps = df['iv_pct'].values if 'iv_pct' in df.columns else np.full(len(df), 50.0)
    ma200s = df['ma200'].values if 'ma200' in df.columns else np.full(len(df), 0.0)
    dates = df.index
    n = len(df)

    trades: list[TradeResult] = []
    next_entry_idx = 0

    for i in range(n):
        if i < next_entry_idx:
            continue
        if i + config.dte >= n:
            break
        if np.isnan(ivs[i]) or ivs[i] <= 0:
            next_entry_idx = i + 1
            continue

        # Apply filters
        if config.ivp_filter is not None and ivps[i] < config.ivp_filter:
            next_entry_idx = i + 1
            continue
        if config.sma_filter and prices[i] < ma200s[i]:
            next_entry_idx = i + 1
            continue

        spot = prices[i]
        iv = ivs[i]
        hv = hvs[i] if not np.isnan(hvs[i]) else iv

        # Calculate strikes and credits
        put_strike = bs_put_strike(spot, iv, config.dte, config.delta_entry)
        put_credit = bs_put_price(spot, put_strike, iv, config.dte)

        call_strike = None
        call_credit = 0.0
        if config.structure in ('strangle', 'iron_condor'):
            call_strike = bs_call_strike(spot, iv, config.dte, config.call_delta)
            call_credit = bs_call_price(spot, call_strike, iv, config.dte)

        total_credit = put_credit + call_credit

        # Simulate forward — check each day for stop or expiry
        fwd_rv = forward_realized_vol(df['c'], i, config.dte)
        stopped_out = False
        exit_idx = i + config.dte  # default: hold to expiry

        for j in range(i + 1, min(i + config.dte + 1, n)):
            remaining_dte = config.dte - (j - i)
            if remaining_dte <= 0:
                break
            current_price = prices[j]
            current_iv = ivs[j] if not np.isnan(ivs[j]) else iv

            # Check put side delta stop
            put_delta = bs_put_delta(current_price, put_strike, current_iv, remaining_dte)
            if abs(put_delta) >= config.stop_at_delta:
                exit_idx = j
                stopped_out = True
                break

            # For strangles: also check if call side would trigger
            # (approximated as price > call_strike by enough to imply 50Δ)
            if call_strike is not None and current_price > call_strike:
                # Call is ITM — approximate stop
                call_moneyness = (current_price - call_strike) / current_price
                if call_moneyness > 0.02:  # > 2% ITM = roughly 50Δ+
                    exit_idx = j
                    stopped_out = True
                    break

        # Calculate P&L at exit
        exit_price = prices[exit_idx]
        remaining_dte_at_exit = max(config.dte - (exit_idx - i), 0)

        if remaining_dte_at_exit == 0:
            # At expiry — intrinsic value only
            put_exit_value = max(put_strike - exit_price, 0)
            call_exit_value = max(exit_price - call_strike, 0) if call_strike else 0
        else:
            # Early exit — use BS to estimate remaining value
            exit_iv = ivs[exit_idx] if not np.isnan(ivs[exit_idx]) else iv
            put_exit_value = bs_put_price(exit_price, put_strike, exit_iv, remaining_dte_at_exit)
            call_exit_value = bs_call_price(exit_price, call_strike, exit_iv, remaining_dte_at_exit) if call_strike else 0

        pnl = total_credit - (put_exit_value + call_exit_value)

        # Iron condor: cap the loss at wing width
        if config.structure == 'iron_condor':
            wing_width = spot * config.ic_wing_width
            max_loss = wing_width - total_credit
            pnl = max(pnl, -max_loss)

        regime = 'low_iv' if ivps[i] < 30 else ('high_iv' if ivps[i] > 70 else 'mid_iv')

        trades.append(TradeResult(
            entry_date=dates[i],
            exit_date=dates[exit_idx],
            underlying=df.attrs.get('symbol', 'unknown'),
            structure=config.structure,
            entry_price=spot,
            exit_price=exit_price,
            iv_at_entry=iv,
            hv_at_entry=hv,
            ivp_at_entry=ivps[i],
            forward_rv=fwd_rv if not np.isnan(fwd_rv) else 0,
            vrp=iv - fwd_rv if not np.isnan(fwd_rv) else 0,
            put_strike=put_strike,
            call_strike=call_strike,
            credit_received=total_credit,
            pnl=pnl,
            win=pnl > 0,
            stopped_out=stopped_out,
            above_200sma=prices[i] > ma200s[i],
            dte=config.dte,
            regime=regime,
        ))

        next_entry_idx = i + config.entry_interval_days

    return trades


def trades_to_dataframe(trades: list[TradeResult]) -> pd.DataFrame:
    """Convert list of TradeResult to a DataFrame for analysis."""
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([t.__dict__ for t in trades])
