# Tradelog — Domain Logic Specification

This document describes the intended business logic for the Tradelog application. It is the single source of truth for requirements engineering and implementation. It is independent of any specific technology (database, framework, UI).

---

## 1. Domain Overview

The Tradelog tracks the full lifecycle of a discretionary swing trading operation:

1. **Portfolio allocation** — budget limits per strategy bucket
2. **Trade entry rationale** — why a trade was taken, under what thesis
3. **Position tracking** — individual contracts/shares with live Greeks
4. **Instrument summaries** — aggregated P/L and risk per symbol
5. **Capital monitoring** — account-level snapshots over time
6. **Journaling** — weekly market prep, daily news notes, learnings
7. **Performance analytics** — strategy-level P/L, win rates, equity curves

---

## 2. Enums & Lookups

### Budgets
- `Core` — structural / drift exposure (indexes, metals)
- `Speculative` — momentum, breakout, earnings plays

### Strategies
- Positive Drift
- Range Bound Commodities
- PEADS (Post-Earnings Announcement Drift)
- Momentum
- IV Mean Reversion
- Sector Strength
- Sector Weakness
- Breakout
- Green Line Breakout
- Slingshot
- Pre Earnings

### Trade Structures (TypeOfTrade)
- Short Strangle, Short Put Spread, Short Call Spread
- Long Call, Long Put, Long Call Vertical, Long Put Vertical
- Synthetic Long, Covered Strangle, Butterfly
- Ratio Diagonal Spread, Long Strangle
- Short Put, Short Call
- Long Stock, Short Stock

### Directional Bias
- Bullish, Neutral, Bearish (and combinations like "Bullish, Neutral")

### Timeframes
- `1d` — single day (typical swing)
- `1w` — weekly
- `Delta Band` — managed by delta target, not time

### Position Right
- `C` — Call
- `P` — Put

### Close Reasons (bitmask)
- TakeLoss, TakeProfit, Roll, AssumptionInvalidated, TimeLimit, Other

### Security Types
- Stock, Future, Forex

### Sentiment (bitmask)
- Bullish, Neutral, Bearish

### Profit Mechanisms (bitmask)
- Momentum, Time, Volatility, Drift, Other

### Management Ratings
- As Planned, Minor Adjustments, Rouge (deviated from plan)

---

## 3. Entities

### 3.1 TradeEntry (TradeLog)

The rationale and thesis behind a trade. Entered manually at trade initiation. One entry per trade idea (a trade idea may spawn multiple positions across options and stock).

| Field | Type | Source | Description |
|-------|------|--------|-------------|
| symbol | string | manual | Ticker symbol |
| date | date | manual | Entry date |
| typeOfTrade | enum | manual | Trade structure (see Enums) |
| notes | text | manual | Assumptions and thesis |
| directional | enum | manual | Directional bias |
| timeframe | enum | manual | Intended holding period |
| budget | enum | manual | Core or Speculative |
| strategy | enum | manual | Which strategy this trade belongs to |
| newsCatalyst | bool | manual | Was there a news catalyst? |
| recentEarnings | bool | manual | Recent earnings event? |
| sectorPerformanceSupport | bool | manual | Sector tailwind? |
| ath | bool | manual | Near all-time high? |
| rvol | float? | manual | Relative volume multiple |
| institutionalSupport | string? | manual | Evidence of institutional activity |
| gapPct | float? | manual | Overnight gap percentage |
| xAtrMove | float? | manual | ATR multiple of the triggering move |
| taFaNotes | text? | manual | Technical/fundamental analysis notes |
| intendedManagement | text? | manual | Planned exit/adjustment rules |
| actualManagement | text? | manual | What actually happened (filled during trade) |
| managementRating | enum? | manual | Self-assessment of execution |
| learnings | text? | manual | Post-trade reflection |

### 3.2 OptionPosition

Individual option contract. One row per contract leg. Positions are grouped by symbol for the InstrumentSummary view.

| Field | Type | Source | Description |
|-------|------|--------|-------------|
| symbol | string | manual/IBKR | Underlying ticker |
| contractId | string | IBKR | IBKR contract identifier |
| opened | date | manual | Date position was opened |
| expiry | date | manual | Option expiration date |
| closed | date? | manual | Date position was closed (null if open) |
| pos | int | manual | Position size (positive=long, negative=short) |
| right | enum | manual | C (Call) or P (Put) |
| strike | decimal | manual | Strike price |
| cost | decimal | manual | Entry price per contract |
| closePrice | decimal? | manual | Exit price per contract (null if open) |
| commission | decimal | manual | Total commission paid |
| multiplier | int | manual | Contract multiplier (100 for equity, 1000 for CL, 10000 for NG) |

**Computed fields — from latest OptionPositionsLog snapshot (live data):**

| Field | Formula | Condition |
|-------|---------|-----------|
| lastPrice | Latest price from OptionPositionsLog for this contractId | Open positions only |
| lastValue | `lastPrice * multiplier` | |
| timeValue | Latest timeValue from OptionPositionsLog | Open only |
| delta | `latestDelta * pos` | Open only (position-adjusted) |
| theta | `latestTheta * pos` | Open only (position-adjusted) |
| gamma | Latest gamma from OptionPositionsLog | Open only |
| vega | Latest vega from OptionPositionsLog | Open only |
| iv | `latestIV * 100` (convert decimal to percent) | Open only |
| margin | Latest margin from OptionPositionsLog (max observed) | |

**Computed fields — local calculations:**

| Field | Formula | Condition |
|-------|---------|-----------|
| unrealizedPnL | `pos * (lastPrice - cost)` | Open only; 0 if closed |
| unrealizedPnLValue | `unrealizedPnL * multiplier` | |
| unrealizedPnLPct | `round(unrealizedPnL / cost, 2) * 100` | Open only |
| realizedPnL | `(closePrice - cost) * multiplier * pos - commission` | Closed only; 0 if open |
| realizedPnLPct | `round(realizedPnL / (cost * multiplier), 1) * 100` | |
| durationPct | `(closedOrToday - opened) / (expiry - opened) * 100` | % of total option life elapsed |
| pnlPerDurationPct | `durationPct / unrealizedPnLPct` | Open only; measures time efficiency |
| ROIC | `realizedPnL * 100 / margin` | Return on invested capital (margin-based) |

### 3.3 OptionInstrumentSummary (computed view)

Aggregated view of all option positions for a given symbol. Not stored — computed on demand by grouping OptionPositions by symbol.

| Field | Aggregation | Description |
|-------|-------------|-------------|
| symbol | Group key | |
| opened | `MIN(opened) WHERE closed IS NULL` | Earliest open date among open positions |
| closed | `MAX(closed)` only if ALL positions are closed | Latest close date; null if any position is open |
| DIT | `closedOrToday - opened` | Days in trade |
| DTE | `MIN(expiry - today) WHERE closed IS NULL` | Days to nearest expiry among open positions |
| status | `Open` if any position unclosed, else `Closed` | |
| budget | From TradeEntry: latest entry for symbol, ordered by date desc | |
| currentSetup | From TradeEntry: `typeOfTrade` of latest entry for symbol | |
| strikes | Comma-joined strikes of open positions | |
| intendedManagement | From TradeEntry: `intendedManagement` of latest entry for symbol | |
| pnl | `SUM(unrealizedPnLValue) + SUM(realizedPnL)` | Total P/L |
| unrealizedPnL | `SUM(unrealizedPnLValue)` | |
| unrealizedPnLPct | `AVG(unrealizedPnLPct)` | |
| realizedPnL | `SUM(realizedPnL)` | |
| realizedPnLPct | `AVG(realizedPnLPct)` | |
| timeValue | `SUM(timeValue)` | |
| delta | `SUM(delta)` | Position-adjusted net delta |
| theta | `SUM(theta)` | |
| gamma | `SUM(gamma)` | |
| vega | `SUM(vega)` | |
| avgIV | `AVG(iv)` | |
| margin | `SUM(margin)` | |
| durationPct | `AVG(durationPct)` | |
| pnlPerDurationPct | `AVG(pnlPerDurationPct)` | |
| ROIC | `unrealizedPnL / totalMargin * 100` | |
| commissions | `SUM(commission)` | |

### 3.4 Trade (execution log — stocks and futures)

Individual buy/sell executions for stocks and futures. Multiple trades per symbol form a position lifecycle. Futures and stocks use the same model; the only difference is the multiplier (1 for stocks, contract-specific for futures like M6E=12500, MES=5).

| Field | Type | Source | Description |
|-------|------|--------|-------------|
| symbol | string | manual | Ticker (e.g., AAPL, M6E, MES) |
| date | datetime | manual | Execution timestamp |
| posChange | int | manual | Units bought (+) or sold (-) |
| price | decimal | manual | Execution price |
| commission | decimal? | manual | Commission |
| multiplier | int | manual | Contract multiplier (default: 1 for stocks; e.g., 12500 for M6E) |

**Computed fields (running position tracking):**

| Field | Formula | Description |
|-------|---------|-------------|
| lastPos | Previous trade's totalPos for same symbol | Position before this trade |
| totalPos | `SUMIFS(posChange, symbol, date <= thisDate)` | Running position after this trade |
| avgPrice | If new position: `price`. If adding (same direction): weighted average `(lastPos * prevAvg + change * price) / (lastPos + change)`. If reducing (opposite direction): keep previous avg. | Running average entry price |
| PnL | `posChange * (avgPrice - price) * multiplier` only when `sign(posChange) != sign(lastPos)` (i.e., reducing/closing). 0 otherwise. | Realized P/L on this execution |

### 3.5 TradeInstrumentSummary (computed view)

Aggregated view per symbol for stocks and futures. Same pattern as OptionInstrumentSummary.

| Field | Aggregation |
|-------|-------------|
| symbol | Group key |
| opened | Earliest trade date where position was opened (totalPos went from 0 to non-zero) |
| status | `Open` if current totalPos != 0, else `Closed` |
| budget | From TradeEntry (latest for symbol) |
| positionType | From TradeEntry: `typeOfTrade` (latest for symbol) |
| intendedManagement | From TradeEntry: `intendedManagement` (latest for symbol) |
| lastPrice | Live price from market data (IBKR or equivalent) |
| multiplier | From the symbol's most recent trade |
| pnl | `unrealizedPnL + realizedPnL` |
| unrealizedPnL | `totalPos * (lastPrice - avgPrice) * multiplier` (only if open) |
| unrealizedPnLPct | `unrealizedPnL * 100 / (avgPrice * totalPos * multiplier)` |
| realizedPnL | `SUM(Trade.PnL) WHERE symbol` |
| commissions | `SUM(Trade.commission) WHERE symbol` |

### 3.6 OptionPositionsLog (Greeks time-series) (Greeks time-series)

Periodic snapshots of live Greeks for open option positions. Populated automatically by IBKR sync (currently manual copy/paste from `finance/ibkr/portfolio/options_portfolio_tracking.py`, to be automated).

| Field | Type | Source | Description |
|-------|------|--------|-------------|
| dateTime | datetime | IBKR sync | Snapshot timestamp |
| contractId | string | IBKR | Contract identifier (foreign key to OptionPosition) |
| underlying | decimal | IBKR | Underlying price at snapshot time |
| iv | decimal | IBKR | Implied volatility (decimal, e.g., 0.35 = 35%) |
| price | decimal | IBKR | Option price |
| timeValue | decimal | IBKR | Time value component |
| delta | decimal | IBKR | Delta |
| theta | decimal | IBKR | Theta |
| gamma | decimal | IBKR | Gamma |
| vega | decimal | IBKR | Vega |
| margin | decimal | IBKR | Margin requirement |

**Usage**: OptionPosition computed fields query this table for the latest snapshot by contractId, ordered by dateTime descending.

### 3.7 Capital (account snapshot)

Periodic account-level snapshot. Entered manually (or future: IBKR sync).

| Field | Type | Source | Description |
|-------|------|--------|-------------|
| date | date | manual | Snapshot date |
| netLiquidity | decimal | manual | Net liquidation value |
| maintenance | decimal | manual | Maintenance margin |
| excessLiquidity | decimal | manual | Excess liquidity |
| BPR | decimal | manual | Buying power reduction |

**Computed fields (aggregated from positions at snapshot time):**

| Field | Formula |
|-------|---------|
| maintenancePct | `maintenance * 100 / netLiquidity` |
| totalPnL | `SUM(OptionInstrumentSummary.pnl) + SUM(StockTrades.PnL)` |
| unrealizedPnL | `SUM(OptionInstrumentSummary.unrealizedPnL)` |
| unrealizedPnLPct | `AVG(OptionInstrumentSummary.unrealizedPnLPct, StockInstrumentSummary.unrealizedPnLPct) / 2` |
| realizedPnL | `SUM(OptionInstrumentSummary.realizedPnL) + SUM(StockInstrumentSummary.realizedPnL)` |
| realizedPnLPct | `AVG(OptionInstrumentSummary.realizedPnLPct)` |
| netDelta | `SUM(OptionInstrumentSummary.delta)` |
| netTheta | `SUM(OptionInstrumentSummary.theta)` |
| netVega | `SUM(OptionInstrumentSummary.vega)` |
| netGamma | `SUM(OptionInstrumentSummary.gamma)` |
| avgIV | `AVG(OptionInstrumentSummary.avgIV)` |
| totalMargin | `SUM(OptionInstrumentSummary.margin)` |
| avgDurationPct | `AVG(OptionInstrumentSummary.durationPct)` |
| avgPnlPerDurationPct | `AVG(OptionInstrumentSummary.pnlPerDurationPct)` |
| totalCommissions | `SUM(OptionInstrumentSummary.commissions) * 2` (round-trip estimate) |

### 3.8 Portfolio (budget allocation)

Tracks allocation limits per budget category.

| Field | Type | Description |
|-------|------|-------------|
| budget | enum | Core or Speculative |
| strategy | string | Associated strategy name |
| minAllocation | decimal | Minimum allocation fraction (e.g., 0.10) |
| maxAllocation | decimal | Maximum allocation fraction (e.g., 0.30) |
| currentAllocation | decimal | Current allocation (computed from open positions) |
| pnl | decimal | P/L for this budget bucket |
| pnlPct | decimal | P/L percentage |

### 3.9 WeeklyPrep (journal)

Weekly market preparation notes. Entered manually each week.

| Field | Type | Description |
|-------|------|-------------|
| date | date | Week start date |
| indexBias | string | Overall market bias (Bullish/Neutral/Bearish) |
| breadth | string | Market breadth assessment |
| notableSectors | text | Sector rotation observations |
| volatilityNotes | text | VIX / IV environment |
| openPositionsRequiringManagement | text | Which positions need attention |
| currentPortfolioRisk | string | Risk level assessment (Insufficient/Sufficient/Excessive) |
| portfolioNotes | text | Portfolio-level observations |
| scanningFor | string | What opportunities to look for (Portfolio Need/Opportunity) |
| indexSectorPreference | text | Preferred sectors/indexes this week |
| watchlist | text | Ticker watchlist with directional bias |
| learnings | text | Lessons from the past week |
| focusForImprovement | text | Self-improvement focus |
| externalComments | text | Notes from mentors/peers |

---

## 4. Analytics (computed dashboards)

### 4.1 Trade Analysis — Per-Strategy Performance

Group all closed trades by strategy (from TradeEntry). For each strategy:

| Metric | Formula |
|--------|---------|
| tradeCount | Count of closed trade entries |
| totalPnL | Sum of P/L across all positions linked to trades of this strategy |
| avgWin | `SUM(pnl WHERE pnl > 0) / COUNT(pnl WHERE pnl > 0)` |
| avgLoss | `SUM(pnl WHERE pnl < 0) / COUNT(pnl WHERE pnl < 0)` |
| winRate | `COUNT(pnl > 0) / tradeCount * 100` |
| expectancy | `winRate * avgWin + (1 - winRate) * avgLoss` |
| runningPnL | Cumulative sum ordered by close date |
| maxDrawdown | Largest peak-to-trough decline in runningPnL |

### 4.2 Trade Analysis — Per-Budget Performance

Same metrics as per-strategy, but grouped by budget (Core vs. Speculative).

### 4.3 Trade Analysis — Overall Performance

| Metric | Formula |
|--------|---------|
| dailyPnL | `totalPnL / tradingDays` (calendar days with at least one closed trade) |
| annualizedROI | `365 * dailyPnL * 100 / accountSize` |
| totalCommissions | Sum of all commissions across all position types |
| netPnL | `totalPnL - totalCommissions` |

### 4.4 Greeks Time-Series Visualization

For any open (or recently closed) position, plot the evolution of Greeks over time using OptionPositionsLog data:
- Delta, Theta, Gamma, Vega over time per contract
- Aggregated portfolio Greeks over time (sum across all open positions at each snapshot)
- IV evolution per contract
- Price vs. TimeValue decomposition

---

## 5. Data Flow & Automation

### Current (manual)
```
IBKR TWS → options_portfolio_tracking.py → console output → copy/paste into Google Sheets
Manual entry → Google Sheets → formulas compute aggregations
```

### Target (automated)
```
IBKR TWS → API sync endpoint → OptionPositionsLog table
Manual entry → Tradelog webapp → backend computes aggregations
OptionPositionsLog → backend serves computed fields on demand
```

### IBKR Sync Job
- Periodically (e.g., end of day) connect to IBKR via `ib-async`
- For each open option position: fetch current price, Greeks, margin
- Insert a row into OptionPositionsLog with timestamp
- Trigger recomputation of OptionPosition computed fields (or compute on read)

---

## 6. Cross-Entity Relationships

```
TradeEntry (1) ←—— (N) OptionPosition     [linked by symbol]
TradeEntry (1) ←—— (N) Trade              [linked by symbol]

OptionPosition (1) ←—— (N) OptionPositionsLog  [linked by contractId]

OptionPosition (N) ——→ (1) OptionInstrumentSummary  [grouped by symbol, computed]
Trade          (N) ——→ (1) TradeInstrumentSummary   [grouped by symbol, computed]

OptionInstrumentSummary (N) ——→ Capital             [aggregated into snapshot]
TradeInstrumentSummary  (N) ——→ Capital             [aggregated into snapshot]
```

Note: TradeEntry to positions is linked by **symbol**, not a foreign key. A single TradeEntry for "SLV" on 2026-01-12 relates to all SLV option positions opened around that date. The exact linkage is "latest TradeEntry for this symbol by date" — the webapp should support explicit linking in the future.
