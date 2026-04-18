# Trade Analyst — Product Backlog

Daily pre-market pipeline that ingests Barchart scanner CSVs and market emails from Gmail,
scores candidates against the 5-box checklist, calls Claude API for trade reasoning and
historical compliance review, and pushes results to the Tradelog web app.

**Module:** `finance/apps/analyst/`
**Platform:** Python CLI → Tradelog (.NET/Angular) display

---

## Architecture Decisions (pending `/architect`)

1. **Gmail credential storage** — OAuth2 token location, refresh strategy
2. **Claude API prompt design** — System prompt structure, playbook context size, token budget per call
3. **Python → SQL Server bridge** — Direct `pyodbc` read vs calling the .NET API
4. **Daily Prep data model** — JSON columns vs normalized tables for analysis results
5. **Cost management** — Claude API call budget per daily run, caching strategy for historical reviews

---

## Implementation Order

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

## Epic 1 — Scanner Ingestion & 5-Box Screening

Parse Barchart scanner CSVs, apply the 5-box checklist programmatically,
and produce a scored watchlist.

### E1-S1: Scanner CSV parser

**As a** swing trader,
**I want to** load Barchart scanner CSVs with adaptable column mappings,
**So that** I can change my scanner configuration without breaking the pipeline.

**Acceptance criteria:**

- [ ] Reads one or more CSV files from a configurable directory or from email attachments
- [ ] Column mapping is configurable (YAML/JSON config mapping Barchart column names → internal field names)
- [ ] Extracts at minimum: symbol, price, volume, 5d change %, 1M change %, 52W high distance, sector
- [ ] Deduplicates symbols appearing across multiple scans
- [ ] Outputs a normalized DataFrame/list of scanner candidates
- [ ] Handles missing columns gracefully (warns, doesn't crash)

**Affected layers:** Python data pipeline
**Dependencies:** None
**Status:** Done

---

### E1-S2: Market data enrichment

**As a** swing trader,
**I want** scanner candidates enriched with technical data from my IBKR cache,
**So that** I can evaluate them against my 5-box checklist.

**Acceptance criteria:**

- [ ] For each candidate symbol, loads daily price history from IBKR Parquet cache (if available)
- [ ] Computes: 20/50/200 SMA, SMA slopes, Bollinger Band width, ATR(14), 52W high/low, 12-month return
- [ ] Computes RS line vs SPY (relative performance ratio)
- [ ] Computes volume metrics: current volume vs 50d avg (RVOL), volume contraction detection (VDU)
- [ ] Marks symbols with no IBKR data as "data missing" (included but flagged for manual review)

**Affected layers:** Python data pipeline
**Dependencies:** E1-S1
**Status:** Done

**Notes:** Reuses `finance/apps/conditions/_data.load_daily` and `classify_slope`. Not all scanned symbols will have IBKR history.

---

### E1-S3: Automated 5-box scoring

**As a** swing trader,
**I want** each candidate automatically scored against the 5-box checklist,
**So that** I can focus on stocks that pass all filters.

**Acceptance criteria:**

- [ ] **Box 1 — Trend Template:** Price > 20 SMA > 50 SMA, 50 SMA rising, within 25% of 52W high, positive 12-month return
- [ ] **Box 2 — RS/RW:** RS line vs SPY trending up (10d slope positive), outperforming SPY over 1M
- [ ] **Box 3 — Base Quality:** BB squeeze (BB width < 20d avg), volume contracting, ATR within 0–6× from base, SMA stack (5>10>20>50 all rising)
- [ ] **Box 4 — Catalyst:** Flagged as "requires manual review" (news/earnings are qualitative — evaluated by Claude in E3-S2)
- [ ] **Box 5 — Risk:** Computable stop distance (base low or 20 SMA), flags if stop > 7% from price, computes position size at 0.5% risk
- [ ] Each box scored as PASS / FAIL / MANUAL with reasoning text
- [ ] Overall score: count of passed boxes (0–5), Box 4 always MANUAL
- [ ] Output sorted by score descending, then by RS strength

**Affected layers:** Python data pipeline
**Dependencies:** E1-S2
**Status:** Done

**Notes:** Box 3 sub-criteria (base duration, contraction count) are approximations — flag edge cases for manual chart review.

---

## Epic 2 — Gmail Email Ingestion

Fetch daily market emails from a labeled Gmail folder and extract content + CSV attachments.

### E2-S1: Gmail API authentication setup

**As a** trader,
**I want** the pipeline to authenticate with my Gmail account,
**So that** it can fetch market emails automatically.

**Acceptance criteria:**

- [ ] OAuth2 flow for Gmail API (offline refresh token, stored securely)
- [ ] Credentials stored outside the repository (e.g. `~/.config/tradelog/gmail_credentials.json`)
- [ ] First run prompts browser-based auth; subsequent runs use refresh token
- [ ] Scopes limited to read-only + attachment download

**Affected layers:** Python infrastructure
**Dependencies:** None
**Status:** Pending

**Notes:** Flag for `/architect` — decide on credential storage approach.

---

### E2-S2: Fetch labeled emails and extract attachments

**As a** trader,
**I want** the pipeline to fetch emails from a specific Gmail label and download CSV attachments,
**So that** scanner results and market summaries arrive without manual effort.

**Acceptance criteria:**

- [ ] Fetches unread emails from a configurable Gmail label (e.g. `Trading/Scanners`)
- [ ] Downloads CSV attachments to a local staging directory
- [ ] Extracts email body text (plain text or HTML → text) for market commentary emails
- [ ] Records email metadata: sender, subject, date, attachment filenames
- [ ] Marks processed emails as read (or applies a "processed" label)
- [ ] Idempotent — re-running doesn't reprocess already-handled emails

**Affected layers:** Python data pipeline
**Dependencies:** E2-S1
**Status:** Pending

**Notes:** Barchart sends scanner results as CSV attachments. Other market emails (newsletters) contain text content for Claude to summarize.

---

### E2-S3: Email content classification

**As a** trader,
**I want** emails automatically classified by type,
**So that** each type feeds into the right analysis path.

**Acceptance criteria:**

- [ ] Classifies by sender/subject pattern matching (configurable rules in YAML)
- [ ] Categories: `scanner` (has CSV attachment), `market_commentary` (newsletter body), `research` (other)
- [ ] Scanner emails → CSV attachments fed to E1-S1 parser
- [ ] Market commentary → text content fed to Claude summarization (E3-S1)
- [ ] Unknown senders logged as warnings, content preserved for manual review

**Affected layers:** Python data pipeline
**Dependencies:** E2-S2
**Status:** Pending

**Notes:** Start with simple sender-based rules. Can add Claude-based classification later if needed.

---

## Epic 3 — Claude-Powered Analysis

Use the Claude API to synthesize market context, evaluate trade fit,
and provide coaching insights.

### E3-S1: Market context summarization

**As a** trader,
**I want** Claude to summarize my daily market emails into a structured brief,
**So that** I get a concise market overview without reading multiple newsletters.

**Acceptance criteria:**

- [ ] Sends email text content to Claude API with a structured prompt
- [ ] Output includes: market regime assessment, key themes/sectors, notable movers, risk events ahead
- [ ] Output references the GO/NO-GO framework (SPY trend, VIX level, breadth)
- [ ] Structured as JSON with sections: `regime`, `themes`, `movers`, `risks`, `action_items`
- [ ] Token-efficient: truncates email content to essential paragraphs before sending
- [ ] Handles rate limits and API errors gracefully (retry with backoff)

**Affected layers:** Python data pipeline + Claude API
**Dependencies:** E2-S2
**Status:** Pending

**Notes:** Prompt must include the playbook's regime rules as system context so Claude evaluates against your framework, not generic market commentary.

---

### E3-S2: Trade candidate reasoning

**As a** trader,
**I want** Claude to evaluate my top-scored candidates and explain why each fits (or doesn't fit) my strategies,
**So that** I understand the reasoning and learn to spot setups better.

**Acceptance criteria:**

- [ ] Sends top N candidates (configurable, default 10) with their 5-box scores + technical data to Claude
- [ ] Claude identifies which profit mechanism fits (PM-01 Breakout, PM-02 PEAD, PM-03 Pre-Earnings, PM-05 RS Divergence)
- [ ] For each candidate: trade thesis, setup type (A/B/C/D), recommended structure (from options decision tree), entry/stop/target levels, risk assessment
- [ ] Flags Box 4 (Catalyst) — Claude evaluates whether recent news/earnings qualify as a valid catalyst
- [ ] Output structured as JSON per candidate
- [ ] Includes a `confidence` field (high/medium/low) with reasoning

**Affected layers:** Python data pipeline + Claude API
**Dependencies:** E1-S3, E3-S1 (market context informs candidate reasoning)
**Status:** Pending

**Notes:** The prompt must include the full setup definitions (Type A/B/C/D) and the options structure selection matrix (by IVR). Claude should reason against these rules, not invent its own.

---

### E3-S3: Historical trade compliance review

**As a** trader,
**I want** Claude to review my closed trades and evaluate whether my entries, management, and exits followed my strategy,
**So that** I can identify rule violations and improve discipline.

**Acceptance criteria:**

- [ ] Loads closed trades from Tradelog via REST API (`GET /api/trades/export`)
- [ ] For each trade: entry date, symbol, strategy, structure, P/L, intended management, actual management, management rating, learnings, notes
- [ ] Enriches with market data at time of trade: SPY regime, VIX level, stock's 5-box status at entry
- [ ] Claude evaluates: Was entry consistent with the stated strategy? Was management consistent with intended? Was exit at the right signal?
- [ ] Claude produces a single markdown analysis covering: compliance assessment, what should have been done differently, optimal trade reconstruction, and strategy refinement suggestions (where intuition outperformed rules)
- [ ] Output per trade: score (1–5) + markdown analysis text
- [ ] Pushes each analysis to Tradelog via `POST /api/trades/{id}/analysis`
- [ ] Aggregate insights: patterns across trades (e.g. "You consistently exit too early on Type B setups", "Your Monday entries underperform") — included in the DailyPrep market summary

**Affected layers:** Python data pipeline + Claude API + Tradelog REST API
**Dependencies:** E1-S2 (market data enrichment)
**Status:** Pending

**Notes:** Highest-value feature for learning. Batch trades (e.g. last 30 days) to manage API costs.
Cache results to avoid re-analyzing the same trades.
The markdown analysis should include optimal trade reconstruction (entry, structure, stop, exit per playbook rules)
and flag where intuition outperformed rules. Uses intraday data when available (`finance/utils/intraday.py`),
falls back to daily OHLCV.

> E3-S4 (optimal trade reconstruction) merged into this story — it's a section within the analysis markdown,
> not a separate data structure.

---

## Epic 4 — Tradelog Integration & Display

Push analysis results to the Tradelog web app for persistence and display.

### E4-S1: Analysis results API endpoint

**As a** developer,
**I want** a Tradelog API endpoint that accepts daily analysis reports,
**So that** the Python pipeline can push results for storage and display.

**Acceptance criteria:**

- [ ] `POST /api/daily-prep` accepts a structured JSON report (market summary, watchlist, trade reviews)
- [ ] `GET /api/daily-prep?date={date}` retrieves a report by date
- [ ] `GET /api/daily-prep/latest` retrieves the most recent report
- [ ] Data model stores: date, market summary JSON, watchlist candidates JSON, trade reviews JSON, raw email count
- [ ] One report per day per account (upsert on date)

**Affected layers:** .NET data model + API + EF migration
**Dependencies:** None (can be built in parallel with Python pipeline)
**Status:** Pending

**Notes:** Store analysis as JSON columns — the structure will evolve. Avoid over-normalizing at this stage.

---

### E4-S2: Daily Prep frontend page

**As a** trader,
**I want** a "Daily Prep" page in Tradelog that shows today's analysis,
**So that** I can review scanner results, market context, and trade recommendations in one place.

**Acceptance criteria:**

- [ ] New nav link "Daily Prep" in sidebar (between Dashboard and Trades)
- [ ] Sections: Market Regime Summary, Watchlist (scored candidates table), Trade Recommendations (Claude reasoning), Historical Review (if available)
- [ ] Market summary rendered as structured cards (regime, themes, risks)
- [ ] Watchlist table: symbol, 5-box score breakdown, setup type, profit mechanism, confidence, recommended action
- [ ] Clicking a watchlist symbol opens a detail panel with Claude's full reasoning
- [ ] Date picker to view previous days' reports
- [ ] Shows "No report for this date" when pipeline hasn't run

**Affected layers:** Angular frontend
**Dependencies:** E4-S1
**Status:** Pending

**Notes:** Flag for `/architect` — decide on detail panel layout. Render Claude's reasoning as markdown → HTML.

---

### E4-S3: Trade analysis display and editing

**As a** trader,
**I want** to view and edit Claude's analysis on each trade,
**So that** I can review compliance, correct inaccuracies, and track improvement over time.

**Acceptance criteria:**

- [ ] Trade detail view (sidebar) shows an "Analysis" section listing all `TradeAnalysis` entries
- [ ] Each entry shows: analysis date, score (visual 1–5), model badge, rendered markdown
- [ ] Edit button switches analysis to rich-text editor (Quill, same pattern as trade notes)
- [ ] Saving an edit updates the analysis via `PUT /api/trades/{id}/analysis/{analysisId}`
- [ ] If no analysis exists, section shows "No analysis yet"
- [ ] Links from Daily Prep page to trade detail with analysis visible

**Affected layers:** Angular frontend + API (PUT endpoint for edits)
**Dependencies:** E4-S1, E3-S3
**Status:** Pending

---

## Epic 5 — Pipeline Orchestration & Launcher

Wire everything together as a launchable app with a single-click entry point.

### E5-S1: Pipeline orchestrator and CLI entry point

**As a** trader,
**I want** a single command that runs the full daily analysis pipeline,
**So that** my pre-market prep is one step.

**Acceptance criteria:**

- [ ] Pipeline stages run in order: fetch emails → parse scanners → enrich data → score 5-box → Claude analysis → push to Tradelog
- [ ] Each stage logs progress and timing to stdout
- [ ] Partial failures don't block subsequent stages (e.g. Gmail fails → still runs on any local CSVs)
- [ ] Dry-run mode: `--dry-run` skips Claude API calls and Tradelog push, outputs report to stdout
- [ ] Can be invoked via CLI: `uv run python -m finance.apps analyst`

**Affected layers:** Python CLI (`finance/apps/analyst/`)
**Dependencies:** E1, E2, E3, E4-S1
**Status:** Pending

---

### E5-S2: Configuration file

**As a** trader,
**I want** the pipeline configured via a YAML file,
**So that** I can adjust Gmail labels, scanner mappings, and Claude settings without changing code.

**Acceptance criteria:**

- [ ] Configuration file at `finance/apps/analyst/config.yaml`
- [ ] Sections: `gmail` (label, classification rules), `scanner` (column mapping, base filters), `claude` (model, top-N candidates, token budget), `tradelog` (API URL, account ID)
- [ ] Defaults provided — pipeline runs with zero config on first use (except Gmail OAuth)
- [ ] Config validated on startup with clear error messages for missing/invalid values

**Affected layers:** Python data pipeline
**Dependencies:** E5-S1
**Status:** Pending

---

### E5-S3: Launcher integration

**As a** trader,
**I want** to launch the daily analyst pipeline from the Finance Apps launcher with a single click,
**So that** I don't need to open a terminal or remember CLI commands.

**Acceptance criteria:**

- [ ] Register `analyst` in `finance/apps/__init__.py` APPS dict pointing to `finance.apps.analyst`
- [ ] Module exposes `APP_DESCRIPTION`, `APP_ICON_ID`, and `launch()` matching the launcher contract
- [ ] `launch()` runs the pipeline and keeps the console window open until complete (shows progress + final report summary)
- [ ] Custom icon in `_launcher_icons.py` (e.g. magnifying glass over chart, or clipboard with checkmarks)
- [ ] Launcher button appears alongside existing apps (swing-plot, momentum, conditions)

**Affected layers:** Python launcher + app registration
**Dependencies:** E5-S1
**Status:** Pending

**Notes:**

> **Flag for `/architect`:** The analyst is a non-GUI pipeline, unlike the existing PyQtGraph apps.
> The launcher spawns apps as subprocesses via `subprocess.Popen([sys.executable, "-m", "finance.apps", name])`.
> Key design questions:
>
> 1. **Console visibility** — The launcher currently spawns subprocesses with `start_new_session=True`, which hides the console on Windows. The analyst needs a visible console for progress output. Options: (a) spawn with `CREATE_NEW_CONSOLE` flag on Windows, (b) wrap output in a small Qt log window, (c) use `subprocess.Popen` with `creationflags=subprocess.CREATE_NEW_CONSOLE`.
> 2. **Blocking vs non-blocking** — GUI apps run indefinitely until closed. The analyst runs for 30–120 seconds then exits. Should the launcher button disable during execution and re-enable on completion, or is fire-and-forget fine?
> 3. **Output persistence** — Should the pipeline write a log file alongside pushing to Tradelog, so the console output is preserved after the window closes?
> 4. **Error handling** — If the pipeline partially fails (e.g. Gmail auth expired), how should this surface in the launcher? Toast notification? Log file?
