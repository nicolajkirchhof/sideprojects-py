## Easiest to Research (data + tools already exist)

### 1. PM-09 — Mean Reversion to Trend
**Research gap:** Quantify oscillation periodicity; validate Bollinger lower band touch entries

**Why it's easy:** The momentum_earnings parquet dataset already has everything needed:
- `ma10_dist`, `ma20_dist`, `ma50_dist`, `ma200_dist` — distance from every MA
- `ma5_slope` through `ma50_slope` — trend direction
- `cpct1-24`, `w_cpct1-8` — forward return trajectories (1-24 days, 1-8 weeks)
- `hv9`, `hv14`, `hv20` — volatility context
- Hurst exponent in `indicators.py` for mean-reversion detection

**Approach:** Filter parquet for stocks in uptrend (`ma50_slope > 0`, `ma200_dist > 0`) that pull back to lower band (`ma20_dist < -2σ`), then measure forward returns. Essentially a SQL-like query on existing data.

## What exists today

Three event types feed the parquet data:
1. **Earnings** — announcement date
2. **ATRP breakout** — `abs(pct) > 1.5 * atrp20`
3. **Green line breakout** — ATH after consolidation

Each event row already has the exact columns you'd need for mean reversion analysis:
- **Uptrend context:** `ma50_slope`, `ma200_dist`, `1M_chg` through `12M_chg`
- **Pullback depth:** `ma20_dist0`, `ma50_dist0` at event time
- **Recovery measurement:** `cpct1..cpct24` (forward returns), `ma20_dist1..ma20_dist24`
- **Probability plot:** already shows % holding above MA5/10/20/50 and entry price over time

## What's needed

**One change in two places:**

1. **Data pipeline** (`create_data_momentum_earnings_analysis.py`) — add an `evt_ma_pullback` event, triggered when price touches/crosses below a key MA (e.g., `ma20_dist < -1%`) while the broader trend is intact (`ma50_slope > 0`, `ma200_dist > 0`). Same 51-day window, same indicator columns — just a new trigger condition.

2. **Dashboard** (`_dashboard.py`) — add `evt_ma_pullback` as a checkbox next to the existing Earnings / ATRP / GreenLine checkboxes. The OR-logic filter system already handles multiple event types.

No new columns, no new plots, no new filter infrastructure. The existing momentum/MA-distance filters already let you slice by pullback depth, trend strength, and market cap. The trajectory + distribution + probability plots already answer "what happens after this event?"

## Fit assessment

It's a very clean fit — the hardest part is defining the pullback trigger precisely (which MA, what threshold, what uptrend confirmation). The dashboard and data pipeline are already built for exactly this pattern of analysis.

---

### 2. PM-08 / DPM-04 — Overnight Reversal
**Research gap:** Backtest overnight returns on RS stocks after >2% intraday drops

**Why it's feasible:** Daily OHLCV from IBKR/Dolt has open/high/low/close — overnight return is simply `next_open / close - 1`. The swing_trading_data class provides ATR percentile (to identify >2% drops) and SPY-relative metrics (for RS filtering). No intraday data needed.

**Approach:** Load daily data via `SwingTradingData`, compute close-to-next-open returns, filter for days with intraday drop > 2 ATR, compare overnight returns vs baseline.

---

## Moderate Effort (need minor extensions)

### 3. DPM-06 — 0DTE Variance Risk Premium
**Research gap:** Backtest 0DTE IC on SPX by GEX regime, VIX level, day-of-week

**Why moderate:** `core_pm/backtest.py` already does Black-Scholes premium selling backtests with IV/HV, VRP calculation, and IVP filtering. VIX data is fetched from IBKR. Would need to adapt `BacktestConfig` for 0-1 DTE (currently defaults to 45) and add day-of-week segmentation. GEX data is not available — but VIX regime + day-of-week analysis is doable now.

### 4. DPM-02 — Dealer Gamma Regime
**Research gap:** Backtest momentum returns on negative vs positive GEX days

**Why moderate:** GEX data itself isn't available, but a proxy could be built from the options chain generation in `utils/options.py` + open interest data if available from Dolt. Otherwise, VIX regime (which correlates with gamma exposure) can serve as a first-pass proxy using existing data.

