# Tradelog — Product Backlog

Based on `SPEC.md`. Last updated: 2026-03-19.

---

## Epic Overview

| Epic | Stories | Description | Core dependency |
|---|---|---|---|
| **E1** | 6 | Data Model Overhaul | Foundation — everything depends on this |
| **E2** | 3 | Trade Entry (TradeLog) | E1 |
| **E3** | 4 | Option Positions & Greeks | E1 |
| **E4** | 2 | Stock & Future Trades | E1 |
| **E5** | 3 | Instrument Summaries | E2, E3, E4 |
| **E6** | 3 | Capital Monitoring | E5 |
| **E7** | 3 | IBKR Greeks Sync | E3 |
| **E8** | 2 | Journaling (WeeklyPrep) | E1 |
| **E9** | 4 | Performance Analytics | E5, E6 |
| **E10** | 2 | Utilities | Various |

**Total: 10 epics, 25 stories**

---

## Open Architectural Decisions

1. **E6-1**: Should Capital aggregations be snapshotted (stored at capture time) or computed live from current positions?
2. **E6-3 / E9-4 / E10-3**: Which charting library for Angular? (Chart.js / ng2-charts is the lightest option)
3. **E7-2**: Standalone sync script or integrated sync job?

---

## E1: Data Model Overhaul

### E1-1: Redesign TradeEntry model

**As a** trader,
**I want to** record my full trade thesis at entry time,
**So that** I can review my reasoning and track which strategies I'm using.

**Acceptance criteria:**
- [ ] TradeEntry entity replaces the existing `Log` model
- [ ] All fields from SPEC 3.1 are present: symbol, date, typeOfTrade, notes, directional, timeframe, budget, strategy, newsCatalyst, recentEarnings, sectorPerformanceSupport, ath, rvol, institutionalSupport, gapPct, xAtrMove, taFaNotes, intendedManagement, actualManagement, managementRating, learnings
- [ ] Enums defined for: Budget, Strategy, TypeOfTrade, DirectionalBias, Timeframe, ManagementRating
- [ ] Migration drops/renames old `Log` table and creates new `TradeEntry` table
- [ ] Existing Log data is not silently lost — migration should be reviewed before applying

**Affected layers:** Data model
**Dependencies:** none
**Notes:** The existing `Log` model has `Notes`, `ProfitMechanism`, `Sentiment`. The new TradeEntry expands this significantly. The old `ProfitMechanism` bitmask is replaced by `strategy` enum. Decision for `/architect`: keep `ProfitMechanism` as a separate field or drop it?

---

### E1-2: Redesign OptionPosition model

**As a** trader,
**I want to** track individual option contracts with all fields needed for P/L and Greeks computation,
**So that** the system can compute unrealized/realized P/L and aggregate by symbol.

**Acceptance criteria:**
- [ ] OptionPosition entity replaces the existing `Position` model
- [ ] Stored fields per SPEC 3.2: symbol, contractId, opened, expiry, closed, pos, right, strike, cost, closePrice, commission, multiplier
- [ ] `right` uses enum (Call/Put), not the old `Type` enum which mixed Call/Put/Underlying
- [ ] `pos` is signed integer (positive = long, negative = short)
- [ ] Old `Position` data migration reviewed before applying

**Affected layers:** Data model
**Dependencies:** none
**Notes:** The existing `Position` model has `Type` (Call/Put/Underlying), `Size`, `Strike`, `Cost`, `Close`, `Comission`, `CloseReasons`. Stock positions are moving to Trade (E1-3), so `Underlying` type is removed. `CloseReasons` bitmask can remain as an optional field.

---

### E1-3: Add Trade model

**As a** trader,
**I want to** record individual buy/sell executions for stocks and futures in a single table,
**So that** all non-option position tracking uses the same lifecycle logic.

**Acceptance criteria:**
- [ ] `Trade` entity created with fields: symbol, date, posChange, price, commission, multiplier
- [ ] `multiplier` defaults to 1 (stocks) but is editable (e.g., 12500 for M6E, 5 for MES)
- [ ] Table created via migration
- [ ] No computed fields stored — those are calculated at query time (E5)

**Affected layers:** Data model
**Dependencies:** none

---

### E1-5: Add OptionPositionsLog model

**As a** trader,
**I want to** store periodic Greeks snapshots for each option contract,
**So that** open positions show live Greeks and I can visualize their evolution over time.

**Acceptance criteria:**
- [ ] OptionPositionsLog entity created per SPEC 3.7: dateTime, contractId, underlying, iv, price, timeValue, delta, theta, gamma, vega, margin
- [ ] Replaces existing `Tracking` model (which exists in DB but has no API endpoint)
- [ ] Index on (contractId, dateTime DESC) for efficient "latest snapshot" queries

**Affected layers:** Data model
**Dependencies:** none
**Notes:** The existing `Tracking` model in the backend has similar fields but is not exposed via API. This replaces it with proper indexing.

---

### E1-6: Add WeeklyPrep model

**As a** trader,
**I want to** record my weekly market prep notes in a structured format,
**So that** I maintain a consistent pre-week review process.

**Acceptance criteria:**
- [ ] WeeklyPrep entity created per SPEC 3.10: all 14 fields (date, indexBias, breadth, notableSectors, volatilityNotes, openPositionsRequiringManagement, currentPortfolioRisk, portfolioNotes, scanningFor, indexSectorPreference, watchlist, learnings, focusForImprovement, externalComments)
- [ ] Table created via migration

**Affected layers:** Data model
**Dependencies:** none

---

### E1-7: Refresh enum definitions across all models

**As a** developer,
**I want to** have a single consistent set of enums shared between backend and frontend,
**So that** dropdowns, filters, and validations use the same values everywhere.

**Acceptance criteria:**
- [ ] All enums defined per SPEC Section 2: Budget, Strategy, TypeOfTrade, DirectionalBias, Timeframe, ManagementRating, PositionRight, CloseReasons, SecType, Sentiment, ProfitMechanism
- [ ] Backend enums serialize as camelCase strings (existing pattern via `JsonStringEnumConverter`)
- [ ] Frontend TypeScript enums/types mirror backend exactly

**Affected layers:** Data model | Frontend (types only)
**Dependencies:** E1-1 through E1-6

---

## E2: Trade Entry (TradeLog)

### E2-1: TradeEntry CRUD API

**As a** trader,
**I want to** create, read, update, and delete trade entries via the API,
**So that** the frontend can manage trade rationale records.

**Acceptance criteria:**
- [ ] `GET /api/trade-entries` — returns all entries, ordered by date descending
- [ ] `GET /api/trade-entries/{id}` — single entry
- [ ] `POST /api/trade-entries` — create with all SPEC 3.1 fields
- [ ] `PUT /api/trade-entries/{id}` — update (used to add actualManagement, learnings post-trade)
- [ ] `DELETE /api/trade-entries/{id}` — delete
- [ ] Optional query params: `?symbol=`, `?budget=`, `?strategy=` for filtering

**Affected layers:** API
**Dependencies:** E1-1, E1-7

---

### E2-2: TradeEntry list page

**As a** trader,
**I want to** see a table of all my trade entries with key columns,
**So that** I can quickly find and review past trade rationale.

**Acceptance criteria:**
- [ ] Table shows: symbol, date, typeOfTrade, directional, budget, strategy, managementRating
- [ ] Rows sorted by date descending (most recent first)
- [ ] Clicking a row opens the detail/edit form in the right sidebar (existing split-pane pattern)
- [ ] Filter controls for budget and strategy dropdowns

**Affected layers:** Frontend
**Dependencies:** E2-1

---

### E2-3: TradeEntry create/edit form

**As a** trader,
**I want to** fill in my trade thesis using a structured form,
**So that** I capture all relevant entry criteria consistently.

**Acceptance criteria:**
- [ ] Form fields match SPEC 3.1 with appropriate controls:
  - symbol: text input
  - date: date picker
  - typeOfTrade, directional, timeframe, budget, strategy, managementRating: dropdowns (from enums)
  - newsCatalyst, recentEarnings, sectorPerformanceSupport, ath: checkboxes
  - rvol, gapPct, xAtrMove: numeric inputs (optional)
  - notes, taFaNotes, intendedManagement, actualManagement, learnings: Quill rich-text editor (reuse existing pattern)
  - institutionalSupport: text input
- [ ] Create saves via POST, edit saves via PUT
- [ ] Form validates: symbol and date are required

**Affected layers:** Frontend
**Dependencies:** E2-1, E2-2

---

## E3: Option Positions & Greeks

### E3-1: OptionPosition CRUD API

**As a** trader,
**I want to** manage individual option contract positions via the API,
**So that** I can track every leg of my trades.

**Acceptance criteria:**
- [ ] `GET /api/option-positions` — all positions; supports `?symbol=` and `?status=open|closed` filters
- [ ] `GET /api/option-positions/{id}` — single position with computed fields included in response
- [ ] `POST /api/option-positions` — create with stored fields from SPEC 3.2
- [ ] `PUT /api/option-positions/{id}` — update (primarily to set closePrice and closed date)
- [ ] `DELETE /api/option-positions/{id}` — delete
- [ ] Response DTO includes computed fields: unrealizedPnL, unrealizedPnLPct, realizedPnL, realizedPnLPct, durationPct, ROIC
- [ ] Computed fields sourced from latest OptionPositionsLog entry for this contractId (lastPrice, delta, theta, gamma, vega, iv, timeValue, margin)
- [ ] If no OptionPositionsLog entry exists, Greeks fields return null/0

**Affected layers:** API
**Dependencies:** E1-2, E1-5, E1-7

---

### E3-2: OptionPositionsLog API (bulk insert)

**As a** trader (or automated sync job),
**I want to** submit Greeks snapshots for multiple contracts at once,
**So that** position data stays current without manual entry per contract.

**Acceptance criteria:**
- [ ] `POST /api/option-positions-log/bulk` — accepts an array of snapshot records (SPEC 3.7 fields)
- [ ] `GET /api/option-positions-log?contractId={id}` — returns time-series for a specific contract, ordered by dateTime ascending
- [ ] `GET /api/option-positions-log/latest` — returns the most recent snapshot per open contractId (for dashboard use)
- [ ] Inserting snapshots does not require positions to exist yet (contractId is a loose reference)

**Affected layers:** API
**Dependencies:** E1-5

---

### E3-3: Option positions list page

**As a** trader,
**I want to** see all my option positions in a table with live Greeks and P/L,
**So that** I can monitor my book at a glance.

**Acceptance criteria:**
- [ ] Table columns: symbol, contractId, right, strike, expiry, pos, cost, lastPrice, unrealizedPnL, unrealizedPnLPct, realizedPnL, delta, theta, durationPct, ROIC
- [ ] Toggle filter: Open / Closed / All
- [ ] Rows sorted by symbol then expiry
- [ ] Closed positions show realized P/L; open positions show unrealized P/L + Greeks
- [ ] Color coding: green for positive P/L, red for negative

**Affected layers:** Frontend
**Dependencies:** E3-1

---

### E3-4: Option position create/edit form

**As a** trader,
**I want to** add or close option positions through a form,
**So that** I can record new legs and mark positions as closed with exit prices.

**Acceptance criteria:**
- [ ] Create mode: fields for symbol, contractId, opened, expiry, pos, right (C/P), strike, cost, commission, multiplier
- [ ] Edit mode: additionally shows closePrice and closed date fields (to close a position)
- [ ] Multiplier defaults to 100 but is editable (for futures options: CL=1000, NG=10000)
- [ ] Validation: symbol, contractId, opened, expiry, pos, right, strike, cost are required

**Affected layers:** Frontend
**Dependencies:** E3-1, E3-3

---

## E4: Stock & Future Trades

### E4-1: Trade CRUD API with computed fields

**As a** trader,
**I want to** record buy/sell executions for stocks and futures and see running position, avg price, and P/L,
**So that** I can track my equity and futures portfolio lifecycle.

**Acceptance criteria:**
- [ ] `GET /api/trades?symbol=` — returns trades for a symbol, ordered by date ascending
- [ ] `GET /api/trades` — returns all trades
- [ ] `POST /api/trades` — create with: symbol, date, posChange, price, commission, multiplier (default 1)
- [ ] `PUT /api/trades/{id}` — update
- [ ] `DELETE /api/trades/{id}` — delete (recomputes running fields for subsequent trades)
- [ ] Response includes computed fields: lastPos, totalPos, avgPrice, PnL
- [ ] avgPrice computation: new position = price; adding (same sign) = weighted average; reducing (opposite sign) = keep previous avg
- [ ] PnL computation: `posChange * (avgPrice - price) * multiplier` only when reducing; 0 otherwise
- [ ] Multiplier defaults to 1 for stocks; user sets it for futures (M6E=12500, MES=5, etc.)

**Affected layers:** API
**Dependencies:** E1-3

---

### E4-3: Trades list page and form

**As a** trader,
**I want to** see my stock and futures executions in a table with running position and P/L,
**So that** I can track my non-options book.

**Acceptance criteria:**
- [ ] Table columns: symbol, date, posChange, price, multiplier, totalPos, avgPrice, PnL, commission
- [ ] `multiplier` column hidden by default (most rows are 1), shown if any row has multiplier != 1
- [ ] Filter by symbol
- [ ] Create form: symbol, date, posChange (signed), price, commission, multiplier (defaults to 1)
- [ ] Running fields (totalPos, avgPrice, PnL) are read-only / computed by API

**Affected layers:** Frontend
**Dependencies:** E4-1

---

## E5: Instrument Summaries

### E5-1: OptionInstrumentSummary API endpoint

**As a** trader,
**I want to** see aggregated P/L, Greeks, and risk per underlying symbol,
**So that** I can assess my exposure per name without mentally summing individual legs.

**Acceptance criteria:**
- [ ] `GET /api/instrument-summaries/options` — returns one row per symbol with all fields from SPEC 3.3
- [ ] `GET /api/instrument-summaries/options?status=open` — only symbols with at least one open position
- [ ] Aggregation logic per SPEC: SUM for pnl/Greeks/margin/commissions, AVG for percentages/iv/duration
- [ ] `budget`, `currentSetup`, `intendedManagement` resolved from latest TradeEntry for the symbol
- [ ] `strikes` is comma-joined strikes of open positions, ordered ascending
- [ ] `opened`/`closed`/`DIT`/`DTE`/`status` derived per SPEC 3.3 rules
- [ ] This is a computed view — no dedicated table, computed at query time

**Affected layers:** API
**Dependencies:** E3-1, E3-2, E2-1

---

### E5-2: TradeInstrumentSummary API endpoint

**As a** trader,
**I want to** see per-symbol position summaries for stocks and futures,
**So that** I have a consolidated view of each non-option name.

**Acceptance criteria:**
- [ ] `GET /api/instrument-summaries/trades` — returns one row per symbol
- [ ] `status` derived: Open if current totalPos != 0
- [ ] `realizedPnL` = SUM of Trade.PnL for the symbol (includes multiplier)
- [ ] `unrealizedPnL = totalPos * (lastPrice - avgPrice) * multiplier` (lastPrice initially null)
- [ ] `budget`, `positionType`, `intendedManagement` from latest TradeEntry
- [ ] Multiplier sourced from the instrument's most recent trade

**Affected layers:** API
**Dependencies:** E4-1, E2-1

---

### E5-3: Instrument summaries dashboard page

**As a** trader,
**I want to** see all my open positions summarized by underlying on a single page,
**So that** I can monitor my full book from one screen.

**Acceptance criteria:**
- [ ] Two tables: Option Instruments (top) and Trade Instruments (bottom)
- [ ] Option table columns: symbol, status, currentSetup, strikes, DIT, DTE, pnl, unrealizedPnLPct, delta, theta, vega, margin, ROIC
- [ ] Trade table columns: symbol, status, positionType, pnl, unrealizedPnLPct, realizedPnL
- [ ] Default filter: Open only, with toggle to show Closed
- [ ] Row click navigates to the underlying's positions (filtered option positions or trades)
- [ ] Portfolio totals row at the bottom: sum of delta, theta, vega, gamma, margin, total P/L

**Affected layers:** Frontend
**Dependencies:** E5-1, E5-2

---

## E6: Capital Monitoring

### E6-1: Capital CRUD API with computed fields

**As a** trader,
**I want to** record account snapshots and see portfolio-level aggregations alongside them,
**So that** I can track my account health and risk utilization over time.

**Acceptance criteria:**
- [ ] `GET /api/capital` — returns all snapshots ordered by date descending
- [ ] `GET /api/capital/{id}` — single snapshot
- [ ] `POST /api/capital` — create with stored fields: date, netLiquidity, maintenance, excessLiquidity, BPR
- [ ] `PUT /api/capital/{id}` — update
- [ ] `DELETE /api/capital/{id}` — delete
- [ ] Response includes computed `maintenancePct = maintenance * 100 / netLiquidity`
- [ ] Response includes live portfolio aggregations (computed from current instrument summaries at query time): netDelta, netTheta, netVega, netGamma, totalMargin, avgIV, totalPnL, unrealizedPnL, realizedPnL, totalCommissions

**Affected layers:** API
**Dependencies:** E5-1, E5-2
**Notes:** Decision for `/architect`: should the aggregated values be snapshotted (stored at capture time) or always computed live?

---

### E6-2: Capital list page

**As a** trader,
**I want to** see my account snapshots in a table with key risk metrics,
**So that** I can monitor margin utilization and account growth.

**Acceptance criteria:**
- [ ] Table columns: date, netLiquidity, maintenance, maintenancePct, excessLiquidity, BPR, totalPnL, netDelta, netTheta, totalMargin
- [ ] Sorted by date descending
- [ ] Create/edit form in sidebar with stored fields only (date, netLiquidity, maintenance, excessLiquidity, BPR)
- [ ] Computed fields are read-only in the form

**Affected layers:** Frontend
**Dependencies:** E6-1

---

### E6-3: Capital timeline chart

**As a** trader,
**I want to** see my net liquidity, margin utilization, and P/L plotted over time,
**So that** I can spot trends in account health and risk exposure.

**Acceptance criteria:**
- [ ] Line chart: netLiquidity over time (primary axis)
- [ ] Line chart: maintenancePct over time (secondary axis, with warning threshold at 50%)
- [ ] Bar chart overlay: totalPnL per snapshot
- [ ] Displayed above or as a tab alongside the capital table

**Affected layers:** Frontend
**Dependencies:** E6-2
**Notes:** Decision for `/architect`: charting library choice.

---

## E7: IBKR Greeks Sync

### E7-1: IBKR sync endpoint (receive snapshots)

**As a** trader running the sync script,
**I want to** push Greeks snapshots from the IBKR Python script to the backend,
**So that** I no longer need to copy/paste data into spreadsheets.

**Acceptance criteria:**
- [ ] `POST /api/option-positions-log/sync` — accepts the same payload format as the bulk insert (E3-2) but is explicitly the "sync" entrypoint
- [ ] Accepts an array of snapshots with fields: dateTime, contractId, underlying, iv, price, timeValue, delta, theta, gamma, vega, margin
- [ ] Returns count of records inserted
- [ ] Idempotent: if a snapshot with the same (contractId, dateTime) already exists, skip it (don't duplicate)

**Affected layers:** API
**Dependencies:** E3-2

---

### E7-2: Adapt Python sync script to POST to backend

**As a** trader,
**I want** `options_portfolio_tracking.py` to push data to the Tradelog API instead of printing to console,
**So that** Greeks flow into the webapp automatically.

**Acceptance criteria:**
- [ ] After fetching Greeks from IBKR, script POSTs to `POST /api/option-positions-log/sync`
- [ ] Script reads the backend URL from config or environment variable (default: `http://localhost:5186`)
- [ ] On HTTP error, prints the error and continues (don't crash the IBKR connection)
- [ ] Existing console output preserved (print AND post)
- [ ] Script can be run as a cron/scheduled task

**Affected layers:** Python script (`finance/ibkr/portfolio/options_portfolio_tracking.py`)
**Dependencies:** E7-1
**Notes:** Decision for `/architect`: standalone script or integrated sync job?

---

### E7-3: Sync status indicator in frontend

**As a** trader,
**I want to** see when the last Greeks sync happened,
**So that** I know if my position data is stale.

**Acceptance criteria:**
- [ ] `GET /api/option-positions-log/last-sync` — returns the most recent dateTime across all snapshots
- [ ] Frontend toolbar shows "Last sync: X minutes/hours ago" or "Never synced"
- [ ] Visual warning (amber) if last sync > 24 hours ago

**Affected layers:** API | Frontend
**Dependencies:** E7-1

---

## E8: Journaling

### E8-1: WeeklyPrep CRUD API

**As a** trader,
**I want to** manage my weekly market preparation notes via the API,
**So that** the frontend can provide a structured weekly review workflow.

**Acceptance criteria:**
- [ ] Standard CRUD at `/api/weekly-prep`
- [ ] All 14 fields from SPEC 3.10
- [ ] `GET` returns entries ordered by date descending
- [ ] Optional filter: `?year=` to limit to a specific year

**Affected layers:** API
**Dependencies:** E1-6

---

### E8-2: WeeklyPrep page

**As a** trader,
**I want to** fill in my weekly preparation using a structured form,
**So that** I maintain a consistent pre-week routine.

**Acceptance criteria:**
- [ ] List view: table with date, indexBias, breadth, currentPortfolioRisk, scanningFor
- [ ] Detail/edit form with all 14 fields:
  - indexBias, breadth, currentPortfolioRisk, scanningFor: dropdowns or short text
  - notableSectors, volatilityNotes, openPositionsRequiringManagement, portfolioNotes, watchlist, learnings, focusForImprovement, externalComments: multiline text / Quill editor
  - indexSectorPreference: text
- [ ] "New Week" button pre-fills date to the upcoming Monday
- [ ] Split-pane layout consistent with other pages

**Affected layers:** Frontend
**Dependencies:** E8-1

---

## E9: Performance Analytics

### E9-1: Per-strategy performance API

**As a** trader,
**I want to** see win rate, avg win/loss, expectancy, and equity curve per strategy,
**So that** I can evaluate which strategies are working and which need adjustment.

**Acceptance criteria:**
- [ ] `GET /api/analytics/strategies` — returns per-strategy metrics for closed trades
- [ ] For each strategy: tradeCount, totalPnL, avgWin, avgLoss, winRate, expectancy (per SPEC 4.1 formulas)
- [ ] `GET /api/analytics/strategies/{strategy}/equity-curve` — returns chronological running P/L series (date, cumulativePnL)
- [ ] Trades are linked to strategies via TradeEntry; positions linked to TradeEntry by symbol
- [ ] A trade's P/L = sum of its linked OptionPosition realizedPnL + Trade (stock/futures) realizedPnL

**Affected layers:** API
**Dependencies:** E5-1, E5-2, E2-1

---

### E9-2: Per-budget performance API

**As a** trader,
**I want to** compare Core vs. Speculative bucket performance,
**So that** I can assess whether my budget allocation is balanced.

**Acceptance criteria:**
- [ ] `GET /api/analytics/budgets` — returns same metrics as E9-1 but grouped by budget (Core/Speculative)
- [ ] `GET /api/analytics/budgets/{budget}/equity-curve` — running P/L series per budget

**Affected layers:** API
**Dependencies:** E9-1

---

### E9-3: Overall performance API

**As a** trader,
**I want to** see aggregate portfolio performance metrics,
**So that** I can track my overall trading edge.

**Acceptance criteria:**
- [ ] `GET /api/analytics/overall` — returns:
  - totalPnL, netPnL (after commissions), totalCommissions
  - dailyPnL (`totalPnL / tradingDays`)
  - annualizedROI (`365 * dailyPnL * 100 / accountSize`)
  - accountSize sourced from most recent Capital snapshot's netLiquidity
  - tradingDays = distinct dates with at least one closed trade
- [ ] `GET /api/analytics/overall/equity-curve` — full portfolio running P/L series

**Affected layers:** API
**Dependencies:** E9-1, E6-1

---

### E9-4: Analytics dashboard page

**As a** trader,
**I want to** see all performance metrics and equity curves on a single dashboard page,
**So that** I can review my trading performance in one place.

**Acceptance criteria:**
- [ ] Top section: overall metrics cards (totalPnL, netPnL, annualizedROI, winRate, avgWin, avgLoss)
- [ ] Middle section: equity curve chart (overall, with toggle to overlay per-strategy or per-budget curves)
- [ ] Bottom section: strategy breakdown table (one row per strategy with tradeCount, totalPnL, winRate, expectancy, avgWin, avgLoss)
- [ ] Budget comparison: side-by-side Core vs. Speculative summary
- [ ] Date range filter (default: all time)

**Affected layers:** Frontend
**Dependencies:** E9-1, E9-2, E9-3
**Notes:** Same charting library as E6-3.

---

## E10: Utilities

### E10-2: Portfolio allocation tracking

**As a** trader,
**I want to** define budget allocation limits and see current utilization,
**So that** I stay within my Core/Speculative risk budget.

**Acceptance criteria:**
- [ ] `GET /api/portfolio` — returns budget rows with min/max allocation and current allocation
- [ ] `PUT /api/portfolio/{budget}` — update min/max allocation limits
- [ ] `currentAllocation` computed from total margin of open positions in each budget, divided by latest netLiquidity from Capital
- [ ] Frontend: simple table showing budget, strategy, minAllocation, maxAllocation, currentAllocation, P/L
- [ ] Visual indicator when currentAllocation exceeds maxAllocation (red) or is below minAllocation (amber)

**Affected layers:** Data model | API | Frontend
**Dependencies:** E5-1, E5-2, E6-1

---

### E10-3: Greeks time-series charts

**As a** trader,
**I want to** see how delta, theta, IV, and P/L evolved over time for a specific position or symbol,
**So that** I can evaluate how my positions behaved and learn from the Greeks dynamics.

**Acceptance criteria:**
- [ ] Accessible from the option positions page: "Greeks History" button per contractId
- [ ] Charts: delta, theta, gamma, vega, IV, price, timeValue over time for that contract
- [ ] Also accessible from instrument summary page: aggregated Greeks over time for all contracts of a symbol
- [ ] Aggregated view sums delta/theta/gamma/vega and averages IV across all contracts of the symbol at each snapshot timestamp
- [ ] Date range defaults to the position's opened-to-closed (or opened-to-now)

**Affected layers:** Frontend
**Dependencies:** E3-2, E5-3
**Notes:** Same charting library as E6-3 and E9-4.
