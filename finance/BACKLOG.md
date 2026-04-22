# Finance Apps — Backlog

## Apps — Conditions Dashboard *(superseded by Trading Assistant)*

> **Superseded.** The Trading Assistant (below) absorbs the conditions dashboard. Swing
> regime panel (E1) and DRIFT eligibility panel (E2) move into the assistant's left panel.
> The `conditions` app remains functional during transition but receives no new work.
> Pending stories (E1-S3, E1-S4, E2-*) are re-scoped under Trading Assistant Epic 3.

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

## Apps — Trade Analyst Pipeline *(superseded by Trading Assistant)*

> **Superseded.** The Trading Assistant absorbs the analyst pipeline. Shared modules
> (`_gmail.py`, `_scanner.py`, `_enrichment.py`, `_claude.py`, `_calendar.py`, `_config.py`)
> are imported by the assistant; the pass/fail `_scoring.py` is replaced by weighted scoring.
> The CLI entry point (`python -m finance.apps analyst`) remains functional for dry-runs
> but receives no new work. Tradelog push deferred to post-validation.

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

---

## Apps — Trading Assistant

Desktop application for the evening prep workflow. **Supersedes** both the conditions
dashboard (`finance/apps/conditions/`) and the analyst pipeline (`finance/apps/analyst/`).
Single app for market regime, scored watchlist, AI analysis, and ticker export.
Target: ~1 hour evening session → export watchlists → execute during first and last market hour.

**Module:** `finance/apps/assistant/`
**Platform:** PyQtGraph desktop (reuses `_qt_bootstrap.py`, dark theme, panel layout)
**Absorbs:** Conditions dashboard (swing regime panel, DRIFT eligibility) + Analyst pipeline (Gmail, scanner, enrichment, Claude, calendar)
**Replaces:** `analyst._scoring.py` (pass/fail) with weighted scoring engine
**Scoring spec:** `investing_framework/ScoringSystem.md` (weights, inversion rules, hard gates, tag bonuses)

### Key design change: Weighted Scoring

The current analyst pipeline uses **pass/fail** (0–5 boxes). The new system uses **weighted
scoring** (0–100 points) where each dimension contributes a continuous score. Scanner tag
overlap (`pead-long`, `52w-high`, `vol-spike`, etc.) adds bonus points. The result is a
ranked watchlist where nothing is silently dropped — low-scoring candidates are visible but
deprioritized.

**Scoring weights** (evidence-based, from `/researcher` evaluation 2026-04-22):

| Dimension | Weight | Rationale |
|-----------|--------|-----------|
| D1 Trend Template | **25** | Core momentum signal — century momentum (Geczy & Samonov 2017), cross-sectional momentum (Jegadeesh & Titman 1993). Strongest academic backing, longest documented persistence. |
| D2 Relative Strength | **25** | Cross-sectional momentum IS relative strength. Equally well-documented. Together with D1, captures 50% of the signal. |
| D3 Base Quality | **15** | Timing/entry quality — practitioner evidence (Minervini VCP, BB squeeze) stronger than academic. Reduces whipsaw but doesn't generate alpha independently. |
| D4 Catalyst | **20** | PEAD sub-component is Grade A academic (Ball & Brown 1968). Options flow moderate (Pan & Poteshman 2006). Earnings blackout is a hard constraint (score 0 on violation). |
| D5 Risk | **15** | Survival constraint, not alpha. 7% stop is a hard gate (score 0 if exceeded). Within that range, tighter stops don't predict better outcomes — they prevent ruin (Kaminski & Lo 2014). |
| **Total** | **100** | |
| Tag bonus | +2/tag, cap +12 | Multi-signal convergence reward. Correctly sized: +12 on a 70 → 82, enough to break ties without distorting base ranking. |

**Barchart scanner filter relaxation** (decided 2026-04-22):
- Removed from scanner: % 50D MA > 0%, Slope 50D Rising, Slope 200D > 0, Weighted Alpha > 0
- Retained: Price > $5, Vol > 1M, MktCap > $200M, No earnings within 5 days
- Relaxed: ADR > 3% → ADR > 2%
- Universe: ~150 → ~300–400 stocks (Stage 1/3/4 + turnarounds now visible, scored low but present)

### User workflow

```
Evening (~1h):
  1. Launch app → auto-fetches today's scanner emails
  2. Review scored watchlist — sort, filter, read AI reasoning
  3. Check market conditions panel and upcoming events
  4. Select top candidates → export to Barchart watchlist + TWS CSV
  5. Set ORB alerts in TWS from exported list

Execution day:
  Morning (9:30–10:15): ORB entries from exported watchlist
  Dead zone (10:15–15:00): no action
  Last hour (15:15–15:45): last-hour ORB entries
  Intraday RVOL (#8): manual Barchart scan at 9:45 and 15:00
```

---

### Epic 1 — Weighted Scoring Engine

Replace pass/fail 5-box scoring with continuous 0–100 weighted scoring. Each dimension
produces a 0–weight sub-score (25/25/15/20/15). Scanner tags add bonus points (+2 each,
cap +12). Full spec with anchor tables: `investing_framework/ScoringSystem.md`.

#### TA-E1-S1: Scoring model and configuration

Status: Pending

**As a** swing trader,
**I want** candidates scored on a 0–100 scale with configurable weights per dimension,
**So that** I see a ranked watchlist where near-misses aren't silently dropped.

**Acceptance criteria:**
- [ ] `ScoringConfig` dataclass with weight per dimension (default: 25/25/15/20/15)
- [ ] `ComponentScore` dataclass: `name`, `raw_score` (0.0–1.0), `available` (bool), `source` ("ibkr" | "scanner" | "none")
- [ ] `DimensionScore` dataclass: `dimension` (1–5), `name`, `raw_score` (0.0–1.0), `weighted_score` (0–weight), `components: list[ComponentScore]`, `hard_gate_fired` (bool), `partial` (bool — true if any component unavailable)
- [ ] `CandidateScore` dataclass: `direction` ("long" | "short"), `dimensions: list[DimensionScore]`, `tag_bonus: float`, `total: float` (0–100+), `tags: list[str]`
- [ ] Config loaded from `config.yaml` scoring section; weights must sum to 100
- [ ] Dimensions: Trend Template (25), Relative Strength (25), Base Quality (15), Catalyst (20), Risk (15)
- [ ] Hard gate semantics: gates evaluated FIRST; if any fires, `weighted_score = 0` for that dimension. Sub-components still computed for display but do not contribute to total.
- [ ] Missing data policy: unavailable sub-components excluded from average (reweight over available only). If ALL sub-components unavailable, dimension = 0 + flagged "no data".
- [ ] Sub-component weighting within dimensions: equal (intentional — no differential weighting)

**Affected layers:** Data model | Config
**Dependencies:** None

---

#### TA-E1-S2: Dimension 1 — Trend Template scoring

Status: Pending

**As a** swing trader,
**I want** the Trend Template scored as a gradient rather than pass/fail,
**So that** a stock 26% from its 52W high still appears (scored lower) instead of being cut.

**Acceptance criteria:**
- [ ] Sub-components, each scored 0.0–1.0:
  - Price vs 50d SMA: above = 1.0, within 2% below = 0.5, further = linear decay to 0
  - 50d SMA slope: rising = 1.0, flat = 0.5, falling = 0
  - Price vs 200d SMA: above = 1.0, within 5% below = 0.5, further = decay
  - 200d SMA slope: rising = 1.0, flat = 0.5, falling = 0
  - 52W high distance: 0–5% = 1.0, 5–15% = linear 0.8–0.4, 15–25% = 0.4–0.1, >25% = 0
  - 12-month return: >20% = 1.0, 0–20% = linear, <0% = 0
- [ ] Dimension score = weighted average of sub-components × 25 (dimension weight)
- [ ] Short inversion: `1.0 - raw_score` on SMA sub-components; swap 52W high → low distance; flip 12M return curve
- [ ] Falls back to scanner fields (`pct_from_50d_sma`, `slope_50d_sma`) when IBKR data unavailable

**Affected layers:** Scoring logic
**Dependencies:** TA-E1-S1
**Spec:** `investing_framework/ScoringSystem.md` § D1

---

#### TA-E1-S3: Dimension 2 — Relative Strength scoring

Status: Pending

**Acceptance criteria:**
- [ ] Sub-components:
  - RS slope vs SPY (10d): positive and steep = 1.0, flat = 0.3, negative = 0
  - Perf vs Market 5D: >5% = 1.0, 0–5% = linear, <0% = 0
  - Perf vs Market 1M: >0% = scaled 0.5–1.0, <0% = 0
  - Perf vs Market 3M: >0% = 0.5–1.0, <0% = 0 (if available)
- [ ] Short inversion: `1.0 - raw_score` on all sub-components (clean mirror)
- [ ] Falls back to scanner fields when IBKR RS data unavailable

**Affected layers:** Scoring logic
**Dependencies:** TA-E1-S1
**Spec:** `investing_framework/ScoringSystem.md` § D2

---

#### TA-E1-S4: Dimension 3 — Base Quality scoring

Status: Pending

**Acceptance criteria:**
- [ ] Sub-components:
  - BB squeeze: squeeze on = 1.0, recently fired = 0.8, no squeeze = 0.3
  - Volume contraction (VDU): contracting = 1.0, flat = 0.5, expanding = 0.2
  - SMA stack (5>10>20>50): all aligned = 1.0, 3 of 4 = 0.6, fewer = linear decay
  - ADR%: 3–7% = 1.0, 7–10% = 0.6, <3% or >10% = 0.2
- [ ] Short scoring (direction-aware, NOT simple inversion):
  - BB: expanding = 0.8 (distribution), squeeze on = 0.5
  - Volume: expanding/flat = 0.8 (institutional distribution), contracting = 0.4
  - SMA stack: 5<10<20<50 all falling = 1.0 (mirror)
  - ADR%: same as long
- [ ] Falls back to scanner fields (`ttm_squeeze`, `bb_rank`, `adr_pct_20d`)

**Affected layers:** Scoring logic
**Dependencies:** TA-E1-S1
**Spec:** `investing_framework/ScoringSystem.md` § D3

---

#### TA-E1-S5: Dimension 4 — Catalyst scoring

Status: Pending

**Acceptance criteria:**
- [ ] Sub-components (from scanner data, no IBKR needed):
  - Earnings proximity: >20d → 1.0, 10d → 0.8, 5d → 0.5, <5d = HARD GATE (D4=0)
  - Earnings surprise: ≥+10% → 1.0, +5% → 0.7, 0–5% → 0.3, miss → 0.0
  - Surprise history: 4/4 beats → 1.0, 3/4 → 0.8, 2/4 → 0.5, <2 → 0.2
  - Put/Call ratio: <0.3 → 1.0, 0.5 → 0.8, 1.0 → 0.3, >1.5 → 0.0
  - RVOL: ≥3.0 → 1.0, 2.0 → 0.7, 1.5 → 0.5, 1.0 → 0.3, <0.8 → 0.0
  - IV percentile: context only, no score contribution (used for structure selection)
- [ ] Scored as MANUAL-biased: raw score is a starting estimate, Claude upgrades/downgrades in TA-E5
- [ ] Earnings blackout: <5 days long (HARD GATE D4=0), <10 days short (HARD GATE D4=0). Matches TradingPlaybook.md Box 4 and Barchart base filter.
- [ ] Short scoring (direction-aware):
  - Earnings blackout: 10 days (vs 5 for longs, per Layer 1)
  - Surprise: miss ≤−10% → 1.0, −5% → 0.7, beat → 0.0
  - History: first miss → 1.0, consecutive → 0.7 (weaker drift, Layer 3)
  - P/C ratio: >2.0 → 1.0, 1.5 → 0.8, <0.3 → 0.0

**Affected layers:** Scoring logic
**Dependencies:** TA-E1-S1
**Spec:** `investing_framework/ScoringSystem.md` § D4
**Notes:** This dimension was always MANUAL in the old system. The new scorer gives a starting estimate; Claude provides the final assessment.

---

#### TA-E1-S6: Dimension 5 — Risk scoring

Status: Pending

**Acceptance criteria:**
- [ ] Sub-components:
  - Stop distance (ATR% or 20 SMA distance): ≤3% = 1.0, 3–5% = 0.7, 5–7% = 0.4, >7% = 0
  - ADR vs stop: stop < 1× ADR = 1.0, 1–2× = 0.5, >2× = 0.2
  - Market cap: >$2B = 1.0, $500M–$2B = 0.7, $200M–$500M = 0.4
- [ ] 7% hard stop is a hard gate — D5 = 0 if exceeded (not 0.1)
- [ ] Short scoring adds Short Float sub-component:
  - <10% = 1.0, 10-15% = 0.5, 15-20% = 0.2, >20% = hard gate (D5 → 0)
  - Source: scanner field `short_float` (PEAD/EP view col 7, Options/Flow view col 19)

**Affected layers:** Scoring logic
**Dependencies:** TA-E1-S1
**Spec:** `investing_framework/ScoringSystem.md` § D5

---

#### TA-E1-S7: Scanner tag bonus scoring

Status: Pending

**As a** swing trader,
**I want** candidates appearing on multiple scanner tags to score higher,
**So that** multi-signal convergence is rewarded.

**Acceptance criteria:**
- [ ] Tags from Long Universe scanner: `52w-high`, `5d-momentum`, `1m-strength`, `vol-spike`, `trend-seeker`, `ttm-fired`
- [ ] Tags from PEAD scanner: `pead-long`, `pead-short`, `consecutive-miss`
- [ ] Tag bonus: +2 points per tag (configurable), capped at +12
- [ ] Tags displayed as badges next to the candidate in the watchlist
- [ ] Total score = sum of 5 dimensions + tag bonus (can exceed 100)
- [ ] **Direction assignment from tags** (before scoring): `pead-short` or scanner #14 → short; `consecutive-miss` without `pead-long` → short; conflict (both long+short tags) → long; everything else → long

**Affected layers:** Scoring logic
**Dependencies:** TA-E1-S1
**Notes:** Tag assignment logic matches the conditions defined in `BarchartScreeners.md` Long Universe and PEAD Scanner tag tables.

---

### Epic 2 — App Shell & Data Pipeline

Wire the app into the launcher and connect the existing Gmail/scanner/calendar modules.

#### TA-E2-S1: App skeleton and launcher registration

Status: Pending

**As a** user,
**I want to** launch the trading assistant from the launcher or CLI,
**So that** it's accessible alongside the conditions dashboard.

**Acceptance criteria:**
- [ ] Module at `finance/apps/assistant/` with `APP_NAME = "assistant"`, `APP_DESCRIPTION`, `launch()`
- [ ] Registered in `finance/apps/__init__.py`
- [ ] Launches via `python -m finance.apps assistant`
- [ ] Dark-themed window using `apply_dark_palette()`, 1400×900 default size
- [ ] Three-panel layout in QSplitter: left (300px) = Market Context (regime + events + DRIFT collapsible), centre (flexible) = Watchlist Table, right (350px) = Detail Panel
- [ ] Supersedes `conditions` and `analyst` apps — single entry point for all evening prep

**Affected layers:** App structure | UI
**Dependencies:** None

---

#### TA-E2-S2: Data pipeline integration

Status: Pending

**As a** swing trader,
**I want** the app to fetch and process scanner data on launch or on-demand,
**So that** I have fresh scored candidates each evening.

**Acceptance criteria:**
- [ ] "Run Pipeline" toolbar button triggers: Gmail fetch → CSV parse → IBKR enrich → score → Claude calls
- [ ] Reuses existing modules: `_gmail.py`, `_scanner.py`, `_enrichment.py`, `_calendar.py`
- [ ] New scoring module replaces `_scoring.py` (TA-E1)
- [ ] Pipeline runs in a background QThread — UI remains responsive with progress indicator
- [ ] Results stored in-memory; optionally cached to `_data/assistant/YYYY-MM-DD.json`
- [ ] Fallback: "Load CSV" button for manual CSV import when Gmail unavailable
- [ ] Status bar shows: pipeline stage, candidate count, last run timestamp

**Affected layers:** Data pipeline | UI
**Dependencies:** TA-E2-S1, TA-E1-S1 through TA-E1-S7

---

### Epic 3 — Market Context Panel

Left panel showing market conditions, news summary, upcoming events, and DRIFT eligibility.
**Absorbs** conditions dashboard E1 (swing regime) + E2 (DRIFT) + analyst calendar/Claude.

#### TA-E3-S1: Swing regime indicators (absorbs Conditions E1-S1, E1-S2)

Status: Pending

**As a** swing trader,
**I want** the GO/NO-GO regime displayed in the assistant app,
**So that** I have one app for all evening prep.

**Acceptance criteria:**
- [ ] Reuses `conditions/_data.py` logic: SPY/QQQ trend status, VIX zone, composite GO/NO-GO
- [ ] Compact display: single GO/NO-GO banner + trend rows + VIX row
- [ ] Colour-coded: green banner for GO, amber for CAUTION, red for NO-GO
- [ ] SPY/QQQ each show: price, 50d SMA status (green/red + slope arrow), 200d SMA status
- [ ] VIX shows: level, zone badge (green/amber/red), direction arrow, spike warning

**Affected layers:** UI | Data (reuse from conditions)
**Dependencies:** TA-E2-S1
**Absorbs:** Conditions E1-S1 (Done), E1-S2 (Done) — logic reused, UI rebuilt as compact panel

---

#### TA-E3-S2: Composite GO/NO-GO with stop-out counter (absorbs Conditions E1-S4)

Status: Pending

**As a** swing trader,
**I want** a composite GO/NO-GO status with a manual stop-out counter and macro warnings,
**So that** I have a definitive answer before scanning.

**Acceptance criteria:**
- [ ] Composite: GO (all green), CAUTION (mixed), NO-GO (any red trigger)
- [ ] Each sub-signal shown with individual status
- [ ] Manual counter for consecutive stopped-out trades; ≥3 triggers NO-GO override
- [ ] Counter persists within session, resets on app restart
- [ ] Macro event proximity from economic calendar: "FOMC in 2 days" shown inline

**Affected layers:** UI | State
**Dependencies:** TA-E3-S1, TA-E3-S4
**Absorbs:** Conditions E1-S4 (Pending) — now includes macro events from calendar integration

---

#### TA-E3-S3: Economic events calendar

Status: Pending

**As a** swing trader,
**I want** to see high-impact economic events for the next 5 trading days,
**So that** I know when to pause entries or reduce size.

**Acceptance criteria:**
- [ ] Reuses `_calendar.py:fetch_upcoming_events()` with `impact_filter="High"`
- [ ] Table: date, time, event name, country, impact badge (red/amber), forecast/previous
- [ ] Events within 48h highlighted with warning colour
- [ ] NO-GO keywords (FOMC, CPI, NFP) shown with red alert icon
- [ ] Compact: max 10 rows, sorted by date ascending

**Affected layers:** UI | Calendar (reuse)
**Dependencies:** TA-E2-S1

---

#### TA-E3-S4: Market summary from Claude

Status: Pending

**As a** swing trader,
**I want** a concise AI-generated market summary from today's emails and web articles,
**So that** I understand themes, risks, and notable movers in 2 minutes.

**Acceptance criteria:**
- [ ] Reuses `_claude.py:summarize_market()` — same prompt, same output model
- [ ] Displayed as rendered markdown in a scrollable text area
- [ ] Sections: Regime assessment, Active themes, Notable movers, Risks, Action items
- [ ] "Generating..." placeholder while Claude API call runs in background
- [ ] Cached per date — doesn't re-call Claude on refresh if already generated today

**Affected layers:** UI | Claude integration (reuse)
**Dependencies:** TA-E2-S2

---

#### TA-E3-S5: DRIFT regime and eligibility (absorbs Conditions E2-S1 through E2-S5)

Status: Pending

**As an** investor running the DRIFT portfolio,
**I want** to see the DRIFT regime tier and per-underlying eligibility in the same app,
**So that** I don't need a separate dashboard for DRIFT decisions.

**Acceptance criteria:**
- [ ] DRIFT regime tier: SPY drawdown × VIX → tier (Normal/Elevated/Correction/Deep/Bear) with BP%
- [ ] Underlying registry: Directional + Neutral blocks with tier (Core/Selective/Optional)
- [ ] Per-underlying checklist: IVP ≥ 50, Price vs 200d SMA, IV > HV → eligible/not
- [ ] Structure recommendation: short put / PDS / iron condor per underlying based on conditions
- [ ] BP guardrail: manual BP input, warning when > 50%
- [ ] Collapsible section — DRIFT is secondary to swing for the evening prep workflow

**Affected layers:** UI | Data (reuse from conditions)
**Dependencies:** TA-E2-S1
**Absorbs:** Conditions E2-S1 through E2-S5 (all Pending) — combined into one story since the logic is defined but no UI exists yet. Split during implementation if too large.
**Notes:** DRIFT eligibility requires options IV data from IBKR. If unavailable, show "IBKR required" placeholder.

---

### Epic 4 — Watchlist Table

Centre panel: sortable, filterable table of scored candidates with tag badges.

#### TA-E4-S1: Scored candidate table

Status: Pending

**As a** swing trader,
**I want** a sortable table of all scored candidates with their dimension scores and tags,
**So that** I can quickly identify the highest-conviction setups.

**Acceptance criteria:**
- [ ] QTableView with columns: Checkbox, Symbol, Score (0–100), D1 (Trend), D2 (RS), D3 (Base), D4 (Catalyst), D5 (Risk), Tags, Price, 5D %Chg, RVOL, Sector
- [ ] Sortable by any column (default: Score descending)
- [ ] Score cells colour-coded: green ≥70, amber 40–69, red <40
- [ ] Tag badges as coloured labels (e.g., `pead-long` in blue, `vol-spike` in orange)
- [ ] Row selection highlights and populates the Detail Panel (TA-E5)
- [ ] Row count in status bar: "147 candidates, 12 selected"

**Affected layers:** UI
**Dependencies:** TA-E2-S2

---

#### TA-E4-S2: Filtering and search

Status: Pending

**As a** swing trader,
**I want** to filter the watchlist by score range, tags, and sector,
**So that** I can focus on relevant subsets (e.g., only `pead-long` or only Tech).

**Acceptance criteria:**
- [ ] Score range slider (0–100) filters table in real time
- [ ] Tag filter: multi-select dropdown showing all active tags with counts
- [ ] Sector filter: multi-select dropdown
- [ ] Text search: symbol or sector substring match
- [ ] Filters are AND-combined; active filters shown as chips above the table
- [ ] "Reset filters" button clears all

**Affected layers:** UI
**Dependencies:** TA-E4-S1

---

#### TA-E4-S3: Row checkbox and batch selection

Status: Pending

**As a** swing trader,
**I want** to check/uncheck candidates in the table to build my daily watchlist,
**So that** I can select exactly which tickers to export.

**Acceptance criteria:**
- [ ] Checkbox column in table; header checkbox selects/deselects all visible (filtered) rows
- [ ] "Select top N" button (configurable, default 20) checks the top N by score
- [ ] Selected count shown in status bar
- [ ] Selection persists across sort/filter changes

**Affected layers:** UI
**Dependencies:** TA-E4-S1

---

### Epic 5 — AI Reasoning Panel

Right panel: detailed view of the selected candidate with Claude analysis.

#### TA-E5-S1: Candidate detail view

Status: Pending

**As a** swing trader,
**I want** to see the full scoring breakdown and AI reasoning for the selected candidate,
**So that** I can make an informed decision about whether to include it in my watchlist.

**Acceptance criteria:**
- [ ] Header: Symbol, Price, Score, Tags
- [ ] Dimension breakdown: 5 bars (0–weight each: 25/25/15/20/15) with sub-component detail expandable. Hard gate violations shown with alert icon. Partial data flagged.
- [ ] Scanner data: key fields from the CSV row (change %, RVOL, IV percentile, P/C ratio, earnings)
- [ ] AI reasoning section (from Claude): setup type, profit mechanism, thesis, entry/stop/target, confidence
- [ ] "No reasoning available" state when Claude hasn't been called yet
- [ ] Updates when a different row is selected in the watchlist table

**Affected layers:** UI
**Dependencies:** TA-E4-S1

---

#### TA-E5-S2: On-demand Claude analysis for single candidate

Status: Pending

**As a** swing trader,
**I want** to request Claude analysis for a specific candidate by clicking a button,
**So that** I can get AI reasoning for candidates outside the auto-analyzed top N.

**Acceptance criteria:**
- [ ] "Analyze" button in the detail panel
- [ ] Calls Claude with the single candidate's data + market context (reuses `_claude.py` prompt)
- [ ] Result cached per symbol per date — doesn't re-call if already analyzed today
- [ ] Loading state while API call runs in background
- [ ] Top N candidates (from pipeline run) are auto-analyzed; others are on-demand only

**Affected layers:** UI | Claude integration
**Dependencies:** TA-E5-S1, TA-E3-S4
**Notes:** Cost control — auto-analyze top 10 (configurable), manual trigger for the rest.

---

### Epic 6 — Ticker Export

Export selected tickers to formats usable in Barchart and TWS.

#### TA-E6-S1: Barchart ticker list export

Status: Pending

**As a** swing trader,
**I want** to export selected tickers as a comma-separated list I can paste into Barchart,
**So that** I can create a Barchart watchlist for chart review and intraday monitoring.

**Acceptance criteria:**
- [ ] "Export → Barchart" button in toolbar
- [ ] Copies comma-separated ticker list to clipboard (e.g., `AAPL,MSFT,NVDA,TSLA`)
- [ ] Also saves to `_data/assistant/watchlist-YYYY-MM-DD.txt`
- [ ] Toast notification: "12 tickers copied to clipboard"
- [ ] Only exports checked (selected) rows

**Affected layers:** UI | File I/O
**Dependencies:** TA-E4-S3

---

#### TA-E6-S2: TWS CSV file export

Status: Pending

**As a** swing trader,
**I want** to export selected tickers as a TWS-importable CSV file,
**So that** I can import them into a TWS watchlist via File → Import.

**Acceptance criteria:**
- [ ] "Export → TWS" button in toolbar
- [ ] Generates CSV in TWS import format: `DES,SYMBOL,STK,SMART,,,,` per line (all caps)
- [ ] Saves to `_data/assistant/tws-watchlist-YYYY-MM-DD.csv`
- [ ] Opens the file in the system file explorer for easy drag-and-drop into TWS
- [ ] Toast notification: "TWS file saved: {path}"

**Affected layers:** UI | File I/O
**Dependencies:** TA-E4-S3
**Notes:** TWS import via right-click watchlist → "Import/Export" → "Import Financial Instruments". The API does not support programmatic watchlist manipulation.

---

### Implementation Order

```
Phase 1 — Scoring engine (no UI, testable standalone):
  TA-E1-S1  Scoring model + config              (foundation)
  TA-E1-S2  Trend Template scoring              (parallel)
  TA-E1-S3  RS scoring                          (parallel)
  TA-E1-S4  Base Quality scoring                (parallel)
  TA-E1-S5  Catalyst scoring                    (parallel)
  TA-E1-S6  Risk scoring                        (parallel)
  TA-E1-S7  Tag bonus scoring                   (after S2–S6)

Phase 2 — App shell + data flow:
  TA-E2-S1  App skeleton + launcher             (scaffold)
  TA-E2-S2  Pipeline integration                (connects everything)

Phase 3 — Core UI panels:
  TA-E4-S1  Watchlist table                     (centre panel — most value)
  TA-E4-S3  Row checkbox + batch selection       (required for export)
  TA-E3-S1  Swing regime indicators             (left panel — absorbs Conditions E1)
  TA-E3-S2  Composite GO/NO-GO + stop counter   (left panel — absorbs Conditions E1-S4)
  TA-E3-S3  Events calendar                     (left panel)
  TA-E6-S1  Barchart export                     (immediate utility)
  TA-E6-S2  TWS export                          (immediate utility)

Phase 4 — AI enrichment + DRIFT:
  TA-E3-S4  Market summary (Claude)             (left panel)
  TA-E5-S1  Candidate detail view               (right panel)
  TA-E5-S2  On-demand Claude analysis           (right panel)
  TA-E3-S5  DRIFT regime + eligibility          (left panel collapsible — absorbs Conditions E2)

Phase 5 — Polish:
  TA-E4-S2  Filtering and search                (UX improvement)
```

### Resolved Questions

1. **Scoring weights** — Resolved 2026-04-22 via `/researcher`. Weights: 25/25/15/20/15 (D1 Trend / D2 RS / D3 Base / D4 Catalyst / D5 Risk). See scoring weights table above.
2. **Cache strategy** — Resolved 2026-04-22. One JSON file per day in `_data/assistant/YYYY-MM-DD.json`. Simple, auditable, no cleanup needed.
3. **Short-side scoring** — Resolved 2026-04-22 via `/researcher`. Single scoring function with `direction` parameter ("long" | "short"). D1 and D2 use flag-based inversion (`1 - raw_score`). D3 has direction-aware sub-components (BB expanding + volume expanding = good for shorts, opposite of longs). D4 has direction-aware logic (inverted surprise, 10-day earnings blackout vs 5-day, first-miss > consecutive). D5 adds Short Float sub-component (>20% = hard gate → 0). Same weights (25/25/15/20/15). Short scores NOT scaled down — 30% allocation cap is portfolio-level, not scoring-level. Full spec in `investing_framework/ScoringSystem.md`.
4. **IBKR enrichment dependency** — Resolved 2026-04-22. Prompt user to start IBKR Gateway; do nothing without it. Scanner-only fallback is not worth the complexity — the scoring engine needs IBKR data for D1 (SMA slopes), D2 (RS vs SPY), D3 (BB squeeze, VDU), and D5 (stop distance). Without it, 4 of 5 dimensions are degraded.
5. **Tradelog integration** — Resolved 2026-04-22. Standalone first. Tradelog integration deferred until the assistant has been proven in daily use. Revisit after 30 days of live use.
6. **Hard gate evaluation order** — Resolved 2026-04-22 via `/researcher` + `/architect`. Gates bypass sub-component averaging — checked FIRST, if any fires dimension = 0. Sub-components still computed for display. Precedent: Piotroski F-Score, Altman Z-Score.
7. **Earnings blackout threshold** — Resolved 2026-04-22. D4 long blackout = 5 days (not 3). Aligns with TradingPlaybook.md Box 4 and Barchart base filter. Short = 10 days (Layer 1).
8. **Sub-component weighting** — Resolved 2026-04-22. Equal within dimensions (intentional). Precedent: Piotroski F-Score, DeMiguel et al (2009) on 1/N outperformance. D5 dilution from Short Float is intentional.
9. **Scoring curve anchor points** — Resolved 2026-04-22. Full piecewise-linear anchor tables defined in ScoringSystem.md for all sub-components. Linear interpolation between anchors.
10. **Missing data policy** — Resolved 2026-04-22 via `/researcher`. Exclude unavailable sub-components from average, reweight over available only. Precedent: MSCI ESG Score. If ALL sub-components unavailable, dimension = 0 + "no data" flag.

