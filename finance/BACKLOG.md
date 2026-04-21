# Finance Apps — Backlog

## Apps — Conditions Dashboard

App: `conditions` | Module: `finance.apps.conditions/` | Platform: PyQtGraph desktop

### Epic 1 — Swing Trading Regime Panel (Level A)

At a glance, know whether the market is GO or NO-GO for breakout entries.

#### E1-S1: SPY/QQQ trend status indicators

Status: Done

**As a** swing trader,
**I want to** see whether SPY and QQQ are above/below their 50d and 200d SMAs with slope direction,
**So that** I can immediately determine if the Minervini Trend Template is satisfied at the index level.

**Acceptance criteria:**
- [x] Displays SPY and QQQ each with: last price, 50d SMA, 200d SMA
- [x] Visual indicator (green/red) for price > SMA and SMA slope direction (up/flat/down)
- [x] Overall GO/NO-GO flag derived from: SPY above both SMAs with 200d rising
- [x] Data sourced from IBKR cache (parquet), refreshed on-demand via button

**Affected layers:** Data loading | UI
**Dependencies:** None

#### E1-S2: VIX level and direction indicator

Status: Done

**As a** swing trader,
**I want to** see the current VIX level, its direction (rising/falling/spiking), and whether it's above the 20/30 thresholds,
**So that** I know if volatility conditions support breakout entries.

**Acceptance criteria:**
- [x] Shows VIX last value with colour coding: green (<20), amber (20–30), red (>30)
- [x] Direction arrow: falling (favourable), rising, spiking (>20% 5d increase)
- [x] Contributes to GO/NO-GO composite: VIX >30 or spiking → NO-GO

**Affected layers:** Data loading | UI
**Dependencies:** None
**Notes:** VIX data available via IBKR or `_data/VIX_History.csv` as fallback.

#### E1-S3: Breadth indicator (Advance/Decline)

Status: Pending

**As a** swing trader,
**I want to** see whether advancing stocks outnumber declining stocks,
**So that** I can confirm broad market participation (Bruzzese sector confirmation).

**Acceptance criteria:**
- [ ] Displays A/D ratio or net advances
- [ ] Red flag when declining > advancing by 2:1+
- [ ] Contributes to GO/NO-GO composite

**Affected layers:** Data loading | UI
**Dependencies:** None
**Notes:** May require a new data source — flag for `/architect` if IBKR doesn't provide breadth data directly. Could approximate from sector ETF performance as fallback.

#### E1-S4: Composite GO / NO-GO status with stop-out counter

Status: Pending

**As a** swing trader,
**I want to** see a single composite GO/NO-GO status combining all regime signals, plus a manual stop-out counter,
**So that** I have a definitive answer before scanning for setups.

**Acceptance criteria:**
- [ ] Composite status: **GO** (all green), **CAUTION** (mixed), **NO-GO** (any red)
- [ ] Each sub-signal (SPY trend, VIX, breadth) shown with its individual status
- [ ] Manual counter for consecutive stopped-out trades; >=3 triggers NO-GO override
- [ ] Counter persists within session, resets on app restart
- [ ] Macro event proximity warning: "FOMC in 2 days" (manual input or hardcoded calendar)

**Affected layers:** UI | State
**Dependencies:** E1-S1, E1-S2, E1-S3

---

### Epic 2 — DRIFT Regime & Underlying Eligibility (Level B)

Show the current DRIFT BP tier, and for each underlying whether it passes the pre-trade checklist + which structure to use.

#### E2-S1: DRIFT regime tier panel (SPY drawdown x VIX)

Status: Pending

**As an** investor running the DRIFT portfolio,
**I want to** see which regime tier I'm in (Normal → Bear) with the corresponding BP allocation,
**So that** I know how much capital to deploy in the next trade cycle.

**Acceptance criteria:**
- [ ] Calculates SPY drawdown from 52-week high
- [ ] Cross-references VIX level to determine tier (Normal/Elevated/Correction/Deep/Bear)
- [ ] Displays: current tier name, SPY drawdown %, VIX, recommended BP% (30–80%)
- [ ] Visual scaling bar showing where current conditions sit across the 5 tiers
- [ ] VX contango/backwardation status (if available from IBKR, otherwise omit with note)

**Affected layers:** Data loading | UI
**Dependencies:** None (reuses SPY/VIX data from E1 but independently loadable)

#### E2-S2: Underlying registry with block classification

Status: Pending

**As an** investor,
**I want to** see all DRIFT underlyings organised by block (Directional vs Neutral) with their assigned structures,
**So that** I have the full tradeable universe in one view.

**Acceptance criteria:**
- [ ] Directional block: XSP, IWM, TQQQ, ESTX50, EEM, FXI, EWZ
- [ ] Neutral block: UNG, USO, GLD, WEAT, PDBC, DBA, TLT, BNO, SLV
- [ ] Each underlying shows: name, block, default structure(s), tier (Core/Selective/Optional)
- [ ] Registry is config-driven (dict/dataclass), not hardcoded in UI

**Affected layers:** Data model | UI
**Dependencies:** None

#### E2-S3: Per-underlying pre-trade checklist

Status: Pending

**As an** investor,
**I want to** see which underlyings currently pass the DRIFT pre-trade checklist,
**So that** I know which trades are eligible right now.

**Acceptance criteria:**
- [ ] For each underlying, evaluate and display:
  - IVP >= 50 (or IVR >= 30) — from IBKR IV data
  - Price vs 200d SMA — above/below
  - IV vs HV — premium exists (IV > HV)
- [ ] Each check shown as pass/fail icon per underlying
- [ ] Overall eligibility: all checks pass → "Eligible", any fail → show which failed
- [ ] Underlyings sorted: eligible first, then by block

**Affected layers:** Data loading | UI
**Dependencies:** E2-S2
**Notes:** "No FOMC/CPI within 7 days" and "BP <= 50%" are portfolio-level — handled in E2-S5.

#### E2-S4: Structure recommendation per underlying

Status: Pending

**As an** investor,
**I want to** see the recommended option structure for each eligible underlying based on current conditions,
**So that** I know whether to sell a put, open a strangle, or use an iron condor.

**Acceptance criteria:**
- [ ] Logic from playbook:
  - Directional block + above 200 SMA → **Short put + kicker** (or XYZ for XSP)
  - Directional block + below 200 SMA → **Spreads only** (PDS / iron condor)
  - Neutral block (commodity) → **Iron condor** or **Strangle** (per underlying default)
  - IVP < 50 → **Debit spreads only** or "Wait"
- [ ] Display: underlying | structure name | rationale (one-liner)
- [ ] No strike/expiry selection — just the structure type

**Affected layers:** Business logic | UI
**Dependencies:** E2-S2, E2-S3

#### E2-S5: Portfolio-level guardrails

Status: Pending

**As an** investor,
**I want to** see portfolio-level warnings (BP utilisation, macro calendar proximity),
**So that** I don't open trades that violate non-negotiable rules.

**Acceptance criteria:**
- [ ] Manual BP input field (current BP % used) — no brokerage integration
- [ ] Warning when BP > 50% ("BP cap exceeded — no new trades")
- [ ] Macro event input: next FOMC/CPI date (manual), warns if within 7 days of standard 45-60 DTE expiry
- [ ] Per-trade max loss reminder: "2% max loss per trade" always visible

**Affected layers:** UI | State
**Dependencies:** None

---

### Epic 3 — App Shell & Launcher Integration

Wire the dashboard into the existing launcher and provide the shared app frame.

#### E3-S1: App skeleton and launcher registration

Status: Done

**As a** user,
**I want to** launch the conditions dashboard from the existing launcher grid,
**So that** it's accessible alongside the other dashboards.

**Acceptance criteria:**
- [x] New module at `finance/apps/conditions/` with `__init__.py` exposing `launch()`, `APP_NAME`, `APP_DESCRIPTION`, `APP_ICON_ID`
- [x] Registered in `finance/apps/__init__.py` APPS dict as `"conditions"`
- [x] Launches from launcher grid and via CLI: `python -m finance.apps conditions`
- [x] Dark-themed window using `apply_dark_palette()` from `_qt_bootstrap.py`
- [x] Two-panel layout: left panel = Swing Regime (E1), right panel = DRIFT (E2)

**Affected layers:** App structure | UI
**Dependencies:** None (can be built as empty shell first)

#### E3-S2: On-demand data refresh

Status: Pending

**As a** user,
**I want to** click a "Refresh" button to reload all market data from IBKR,
**So that** I see current conditions without restarting the app.

**Acceptance criteria:**
- [ ] Single "Refresh All" button in toolbar
- [ ] Reloads SPY, QQQ, VIX, and all DRIFT underlying data from IBKR
- [ ] Shows loading spinner / progress indicator during refresh
- [ ] Timestamp of last refresh displayed in status bar
- [ ] Graceful fallback to cached parquet if IBKR connection fails

**Affected layers:** Data loading | UI
**Dependencies:** E3-S1

---

### Implementation Order

```
E3-S1  App shell + launcher registration          (scaffold)
  |
E1-S1  SPY/QQQ trend status                       (first visible content)
E1-S2  VIX indicator                               (parallel with S1)
  |
E1-S4  Composite GO/NO-GO                          (ties E1 together)
  |
E2-S1  DRIFT regime tier                           (first DRIFT content)
E2-S2  Underlying registry                         (parallel with S1)
  |
E2-S3  Pre-trade checklist                         (needs registry + data)
E2-S4  Structure recommendation                    (needs checklist)
  |
E2-S5  Portfolio guardrails                        (independent, slot in anytime)
E1-S3  Breadth indicator                           (may need data source decision)
  |
E3-S2  Refresh button                              (polish, after content works)
```

### Open Questions

- **E1-S3 (Breadth):** Confirm whether IBKR provides advance/decline data or approximate via sector ETF dispersion. Flag for `/architect`.

---

## Apps — Trade Analyst Pipeline

Daily pre-market pipeline that ingests Barchart scanner CSVs and market emails from Gmail,
scores candidates against the 5-box checklist, calls Claude API for trade reasoning and
historical compliance review, and pushes results to the Tradelog web app.

**Module:** `finance/apps/analyst/`
**Platform:** Python CLI → Tradelog (.NET/Angular) display

### Open issues

- Screener CSV filename schema is `screener-*04-17-2026.csv` — investigate whether attachments can be downloaded directly from Gmail instead of manual export.

### Architecture Decisions (pending `/architect`)

1. **Gmail credential storage** — OAuth2 token location, refresh strategy
2. **Claude API prompt design** — System prompt structure, playbook context size, token budget per call
3. **Python → SQL Server bridge** — Direct `pyodbc` read vs calling the .NET API
4. **Daily Prep data model** — JSON columns vs normalized tables for analysis results
5. **Cost management** — Claude API call budget per daily run, caching strategy for historical reviews

### Implementation Order

```
Phase 1 — Python pipeline (immediately useful, no LLM):
  E1-S1 → E1-S2 → E1-S3   Scanner ingestion + 5-box scoring
  E2-S1 → E2-S2 → E2-S3   Gmail fetch + classification
  (E1 and E2 can run in parallel)

Phase 2 — Claude integration:
  E3-S1 → E3-S2            Market summary + trade reasoning
  E3-S3                     Compliance review + optimal reconstruction (merged)

Phase 3 — Tradelog display:
  E4-S1 → E4-S2 → E4-S3    API + frontend + historical view

Phase 4 — Orchestration & launcher:
  E5-S1 → E5-S2 → E5-S3    CLI + config + launcher button
```

---

### Epic 1 — Scanner Ingestion & 5-Box Screening

Parse Barchart scanner CSVs, apply the 5-box checklist programmatically,
and produce a scored watchlist.

#### E1-S1: Scanner CSV parser

Status: Done

**As a** swing trader,
**I want to** load Barchart scanner CSVs with adaptable column mappings,
**So that** I can change my scanner configuration without breaking the pipeline.

**Acceptance criteria:**

- [x] Reads one or more CSV files from a configurable directory or from email attachments
- [x] Column mapping is configurable (YAML/JSON config mapping Barchart column names → internal field names)
- [x] Extracts at minimum: symbol, price, volume, 5d change %, 1M change %, 52W high distance, sector
- [x] Deduplicates symbols appearing across multiple scans
- [x] Outputs a normalized DataFrame/list of scanner candidates
- [x] Handles missing columns gracefully (warns, doesn't crash)

**Affected layers:** Python data pipeline
**Dependencies:** None

---

#### E1-S2: Market data enrichment

Status: Done

**As a** swing trader,
**I want** scanner candidates enriched with technical data from my IBKR cache,
**So that** I can evaluate them against my 5-box checklist.

**Acceptance criteria:**

- [x] For each candidate symbol, loads daily price history from IBKR Parquet cache (if available)
- [x] Computes: 20/50/200 SMA, SMA slopes, Bollinger Band width, ATR(14), 52W high/low, 12-month return
- [x] Computes RS line vs SPY (relative performance ratio)
- [x] Computes volume metrics: current volume vs 50d avg (RVOL), volume contraction detection (VDU)
- [x] Marks symbols with no IBKR data as "data missing" (included but flagged for manual review)

**Affected layers:** Python data pipeline
**Dependencies:** E1-S1
**Notes:** Reuses `finance/apps/conditions/_data.load_daily` and `classify_slope`.

---

#### E1-S3: Automated 5-box scoring

Status: Done

**As a** swing trader,
**I want** each candidate automatically scored against the 5-box checklist,
**So that** I can focus on stocks that pass all filters.

**Acceptance criteria:**

- [x] Box 1 — Trend Template: Price > 20 SMA > 50 SMA, 50 SMA rising, within 25% of 52W high, positive 12-month return
- [x] Box 2 — RS/RW: RS line vs SPY trending up (10d slope positive), outperforming SPY over 1M
- [x] Box 3 — Base Quality: BB squeeze, volume contracting, ATR within 0–6× from base, SMA stack all rising
- [x] Box 4 — Catalyst: flagged as "requires manual review" (evaluated by Claude in E3-S2)
- [x] Box 5 — Risk: computable stop distance, flags if stop > 7% from price, computes position size at 0.5% risk
- [x] Each box scored as PASS / FAIL / MANUAL with reasoning text
- [x] Overall score: count of passed boxes (0–5), Box 4 always MANUAL
- [x] Output sorted by score descending, then by RS strength

**Affected layers:** Python data pipeline
**Dependencies:** E1-S2

---

### Epic 2 — Gmail Email Ingestion

Fetch daily market emails from a labeled Gmail folder and extract content + CSV attachments.

#### E2-S1: Gmail API authentication setup

Status: Done

**Acceptance criteria:**

- [x] OAuth2 flow for Gmail API (offline refresh token, stored securely)
- [x] Credentials stored outside the repository (`finance/apps/analyst/_credentials/`, private)
- [x] First run prompts browser-based auth; subsequent runs use refresh token
- [x] Scopes limited to read-only + attachment download

**Dependencies:** None
**Notes:** Setup docs at `docs/SETUP-GMAIL-API.md`.

---

#### E2-S2: Fetch labeled emails and extract attachments

Status: Done

**Acceptance criteria:**

- [x] Fetches emails from a configurable Gmail label
- [x] Downloads CSV attachments to a local staging directory
- [x] Extracts email body text for market commentary emails
- [x] Idempotent — uses `_state.json` date-based query, doesn't reprocess handled emails

**Dependencies:** E2-S1

---

#### E2-S3: Email content classification

Status: Done

**Acceptance criteria:**

- [x] Classifies by sender/subject pattern matching (configurable rules in YAML)
- [x] Categories: `scanner` (CSV attachment), `market_commentary` (newsletter body), `research` (other)
- [x] Scanner emails → CSV attachments fed to E1-S1 parser
- [x] Market commentary → text content fed to Claude summarization (E3-S1)

**Dependencies:** E2-S2

---

### Epic 3 — Claude-Powered Analysis

#### E3-S1: Market context summarization

Status: Done

**Acceptance criteria:**

- [x] Sends email text content to Claude API with a structured prompt
- [x] Output: market regime assessment, key themes/sectors, notable movers, risk events ahead
- [x] Structured as JSON: `regime`, `themes`, `movers`, `risks`, `action_items`
- [x] System prompt contains condensed playbook rules (~4K tokens, cached)

**Dependencies:** E2-S2
**Notes:** Prompt templates in `_prompts/`. Uses Sonnet.

---

#### E3-S2: Trade candidate reasoning

Status: Done

**Acceptance criteria:**

- [x] Sends top N candidates with 5-box scores + technical data to Claude
- [x] Claude identifies profit mechanism (PM-01 Breakout, PM-02 PEAD, PM-03 Pre-Earnings, PM-05 RS Divergence)
- [x] For each candidate: trade thesis, setup type (A/B/C/D), recommended structure, entry/stop/target, confidence
- [x] Flags Box 4 (Catalyst) — Claude evaluates whether news/earnings qualify
- [x] Output structured as JSON per candidate

**Dependencies:** E1-S3, E3-S1

---

#### E3-S3: Historical trade compliance review

Status: Done

**Acceptance criteria:**

- [x] Loads closed trades from Tradelog via REST API
- [x] Enriches with market data at time of trade: SPY regime, VIX level, stock's 5-box status at entry
- [x] Claude evaluates entry, management, and exit consistency against stated strategy
- [x] Per-trade: score (1–5) + markdown analysis including optimal reconstruction
- [x] Pushes analysis to Tradelog via `POST /api/trades/{id}/analysis`
- [x] Aggregate insights included in DailyPrep market summary

**Dependencies:** E1-S2, E4-S1
**Notes:** E3-S4 (optimal trade reconstruction) merged into this story — it's a section within the analysis markdown.

---

### Epic 4 — Tradelog Integration & Display

#### E4-S1: Analysis results API endpoint

Status: Done

**Acceptance criteria:**

- [x] `POST /api/daily-prep` accepts structured JSON report
- [x] `GET /api/daily-prep?date={date}` and `/latest`
- [x] DailyPrep + TradeAnalysis entities, EF migration created

**Dependencies:** None

---

#### E4-S2: Daily Prep frontend page

Status: Done

**Acceptance criteria:**

- [x] "Daily Prep" nav link in sidebar
- [x] Sections: Market Regime Summary, Watchlist, Trade Recommendations, Historical Review
- [x] Watchlist table with click-to-expand Claude reasoning detail
- [x] Date picker for previous days' reports

**Dependencies:** E4-S1

---

#### E4-S3: Trade analysis display and editing

Status: Done

**Acceptance criteria:**

- [x] Trade detail view shows "Analysis" section listing all TradeAnalysis entries
- [x] Each entry: analysis date, score (visual 1–5), model badge, rendered markdown
- [x] Edit button → rich-text editor (Quill), saves via `PUT /api/trades/{id}/analysis/{id}`

**Dependencies:** E4-S1, E3-S3

---

### Epic 5 — Pipeline Orchestration & Launcher

#### E5-S1: Pipeline orchestrator and CLI entry point

Status: Done

**Acceptance criteria:**

- [x] Stages run in order: fetch emails → parse scanners → enrich → score → Claude → push to Tradelog
- [x] Partial failures don't block subsequent stages
- [x] Dry-run mode: `--dry-run` skips Claude API calls and Tradelog push
- [x] CLI: `uv run python -m finance.apps analyst`

**Dependencies:** E1, E2, E3, E4-S1

---

#### E5-S2: Configuration file

Status: Done

**Acceptance criteria:**

- [x] `finance/apps/analyst/config.yaml` with sections: `gmail`, `scanner`, `claude`, `tradelog`
- [x] Defaults provided — pipeline runs with zero config on first use (except Gmail OAuth)

**Dependencies:** E5-S1

---

#### E5-S3: Launcher integration

Status: Done

**Acceptance criteria:**

- [x] `analyst` registered in `finance/apps/__init__.py`
- [x] `launch()` runs pipeline with visible console output
- [x] Custom icon in `_launcher_icons.py`

**Dependencies:** E5-S1
**Open questions (pending `/architect`):** Console visibility on Windows, blocking vs non-blocking launcher, output persistence, error surfacing.

---

## Intraday Execution Engine

Automated live execution of intraday PM strategies via IBKR Gateway.
Runs as a long-lived Python process; fires bracket orders at bar closes,
manages trailing stops natively via IBKR, and journals all actions to JSONL.

**Module:** `finance/execution/`
**Platform:** Python async process (ib_async + APScheduler)
**Dependencies:** IBKR Gateway running locally (paper: port 4002, live: port 4001)

### Architecture decisions (settled in `/architect` session)
- ib_async async API (`asyncio.run` + `APScheduler.AsyncIOScheduler`)
- Entry: OCA stop orders (BUY STOP + SELL STOP in same OCA group)
- Stop: native IBKR trailing stop (`orderType='TRAIL'`, `auxPrice=trail_pts`) placed on fill
- Fill callback cancels unfilled OCA leg, places trailing stop
- ATR trail: `atr * 0.20` (SRS strategies); bar range + 4 pts (ASRS strategy)
- Conflict resolution: same direction → skip; opposite direction → flip (cancel + reverse)
- Journal: append-only JSONL at `finance/execution/logs/YYYY-MM-DD.jsonl`
- Tradelog sync: Flex sync handles fills; JSONL holds automation metadata keyed by ExecutionId
- Paper/live: single config flag (`mode: paper | live`)

### Signal timing
| Strategy | Instrument | Bar closes | Scheduler fires |
|----------|------------|------------|-----------------|
| Hougaard SRS | FDXS | 09:45 Frankfurt | 09:45:30 Frankfurt |
| Hougaard ASRS | MNQ | 09:50 ET | 09:50:30 ET |
| Hougaard SRS | MNQ | 10:00 ET | 10:00:30 ET |
| OCO Opening Bar 30m | MNQ | 10:00 ET | 10:00:30 ET |

### Epic 1 — Core Infrastructure

#### E1-S1: Config module

Status: Pending

**As a** developer running the execution engine,
**I want** a YAML-based config with typed dataclasses,
**So that** switching between paper and live requires changing one line.

**Acceptance criteria:**
- [ ] `EngineConfig`, `IbkrConfig`, `RiskConfig` dataclasses
- [ ] `load_config(path)` validates required fields and raises on invalid mode
- [ ] Default `config.yaml`: paper mode, `client_id=10`, `max_daily_loss_usd=500`, `max_daily_loss_eur=400`
- [ ] `mode: paper` selects `paper_port`; `mode: live` selects `live_port`

**Affected layers:** Config
**Dependencies:** None

---

#### E1-S2: Journal

Status: Pending

**As a** trader reviewing automation performance,
**I want** every engine action logged to a JSONL file,
**So that** I can join it with Flex import data for strategy-level P&L analysis.

**Acceptance criteria:**
- [ ] Writes to `finance/execution/logs/YYYY-MM-DD.jsonl`, one JSON object per line
- [ ] Entry types: `signal`, `placed`, `fill`, `cancel`, `skip`, `flip`, `eod_flatten`
- [ ] Every entry includes: `ts` (ISO with TZ), `strategy_id`, `symbol`, `direction`
- [ ] Fill entries include: `fill_price`, `trail_pts`, `atr`, `ibkr_execution_id`
- [ ] File rotates daily (new file per trading day)
- [ ] Thread-safe append (no partial writes)

**Affected layers:** Persistence
**Dependencies:** None

---

#### E1-S3: Position tracker

Status: Pending

**As a** developer,
**I want** an in-memory record of open positions and pending orders per instrument,
**So that** conflict resolution and EOD flattening work correctly even after reconnects.

**Acceptance criteria:**
- [ ] `InstrumentState` dataclass: `long_order_id`, `short_order_id`, `direction`, `fill_price`, `trail_order_id`
- [ ] `PositionTracker.get(symbol)` → `InstrumentState | None`
- [ ] `set_pending`, `on_fill`, `on_close` transition methods
- [ ] `reconcile(ibkr_positions, ibkr_orders)` rebuilds state from live IBKR data on reconnect
- [ ] All state changes logged to journal

**Affected layers:** State
**Dependencies:** E1-S2

---

#### E1-S4: Broker wrapper

Status: Pending

**As a** developer,
**I want** a single `Broker` class that abstracts all ib_async calls,
**So that** strategies and the engine never interact with ib_async directly.

**Acceptance criteria:**
- [ ] `connect(config)` — async connect; selects port based on `mode`
- [ ] `qualify_contracts(specs)` → `dict[str, Contract]` — qualifies all instruments at startup
- [ ] `fetch_daily_bars(contract, n_days)` → list of BarData (for ATR)
- [ ] `fetch_intraday_bars(contract, bar_size, session_open, session_tz)` → session bars for today
- [ ] `place_oca_entry(contract, spec)` → `(Trade, Trade)` — OCA group, `transmit=True`
- [ ] `place_trailing_stop(contract, direction, trail_pts, qty)` → `Trade`
- [ ] `cancel_order(order_id)`, `flatten_position(contract)`
- [ ] `get_open_positions()`, `get_open_orders()` — for reconcile
- [ ] Raises `BrokerError` (not ib_async internals) on failure

**Affected layers:** Broker
**Dependencies:** E1-S1

---

### Epic 2 — Strategies

#### E2-S1: Strategy base

Status: Pending

**Acceptance criteria:**
- [ ] `SignalBar` dataclass: `open, high, low, close, bar_time`
- [ ] `OrderSpec` dataclass: `symbol, direction, entry_long, entry_short, trail_pts, qty, strategy_id, signal_bar, atr`
- [ ] `Strategy` ABC: `strategy_id`, `symbol`, `session_tz`, `signal_fire_time`, `eod_time`, abstract `compute_signal(bars, atr) → OrderSpec | None`
- [ ] `compute_14d_atr(daily_bars) → float` shared utility

**Affected layers:** Strategy
**Dependencies:** E1-S4

---

#### E2-S2: Hougaard SRS strategy

Status: Pending

**Signal:** Nth 15-min bar (bar[2] = 09:30 Frankfurt for FDXS; bar[1] = 09:45 ET for MNQ)
**Stop:** `atr * ATR_TRAIL_FACTOR` (0.20)

**Acceptance criteria:**
- [ ] `HougaardSrsStrategy(symbol, session_tz, session_open, bar_index, signal_fire_time, eod_time)`
- [ ] `compute_signal` returns `None` if bar index unavailable
- [ ] Entry: `bar.high + ENTRY_OFFSET_PTS` (long), `bar.low - ENTRY_OFFSET_PTS` (short)
- [ ] Instantiated for FDXS (bar_index=2, Frankfurt) and MNQ (bar_index=1, ET)

**Dependencies:** E2-S1

---

#### E2-S3: Hougaard ASRS strategy

Status: Pending

**Signal:** 4th 5-min bar (bar[3] = 09:45 ET for MNQ)
**Stop:** `bar_range + 2 * ENTRY_OFFSET_PTS`

**Acceptance criteria:**
- [ ] `compute_signal` returns `None` if bar range < `MIN_BAR_RANGE_PTS` (no fallback in Phase 1)
- [ ] `trail_pts = signal_bar.high - signal_bar.low + 2 * ENTRY_OFFSET_PTS`

**Dependencies:** E2-S1

---

#### E2-S4: OCO Opening Bar 30m strategy

Status: Pending

**Signal:** 1st 30-min bar (bar[0] = 09:30 ET for MNQ, closes at 10:00)
**Stop:** `atr * ATR_TRAIL_FACTOR`

**Acceptance criteria:**
- [ ] Fires at 10:00:30 ET
- [ ] `compute_signal` returns `None` if bar unavailable

**Dependencies:** E2-S1

---

### Epic 3 — Engine & Entry Point

#### E3-S1: Engine

Status: Pending

**Acceptance criteria:**
- [ ] `Engine(config)` — holds broker, tracker, journal, strategies, daily ATR cache
- [ ] `start()`: connect → qualify contracts → fetch ATR → reconcile → schedule jobs → run loop
- [ ] Pre-market job (08:45 Frankfurt / 09:00 ET): refresh ATR cache via `reqHistoricalDataAsync`
- [ ] Signal job per strategy: `_run_signal(strategy)` → fetch bars → compute_signal → conflict check → place OCA → register fill callbacks
- [ ] Fill callback: cancel other OCA leg → place trailing stop → update tracker → journal
- [ ] EOD job per instrument: `get_open_positions()` + `get_open_orders()` from IBKR → cancel → flatten → journal
- [ ] `_on_reconnect()`: calls `broker.get_open_positions/orders()` → `tracker.reconcile()`
- [ ] Conflict logic: same direction → skip + journal `skip`; opposite → flatten + new bracket + journal `flip`
- [ ] `apscheduler>=3.10` added to `pyproject.toml`

**Affected layers:** Engine
**Dependencies:** E1-S1 → E1-S4, E2-S1 → E2-S4

---

#### E3-S2: Entry point

Status: Pending

**Acceptance criteria:**
- [ ] `python -m finance.execution` starts the engine
- [ ] `--config PATH` overrides default config path
- [ ] `--mode paper|live` overrides config mode
- [ ] Logs startup banner with mode, instruments, and scheduled signal times

**Dependencies:** E3-S1

---

### Epic 4 — Tests

#### E4-S1: Unit tests

Status: Pending

**Acceptance criteria:**
- [ ] `test_execution_config.py` — YAML loading, mode validation, missing fields raise
- [ ] `test_execution_journal.py` — JSONL written correctly, all required fields present, file rotates by day
- [ ] `test_execution_tracker.py` — state transitions: pending → fill → close; reconcile rebuilds state
- [ ] `test_execution_strategies.py` — `compute_signal` with synthetic bars: correct entry/trail prices; `None` on insufficient bars; ASRS `None` on narrow bar

**Dependencies:** E1-S1 → E1-S3, E2-S1 → E2-S4

