# Trade Analyst — Architecture

> Backlog: `finance/BACKLOG-ANALYST.md`
> Module: `finance/apps/analyst/`
> Date: 2026-04-18

---

## Overview

Daily pre-market pipeline that ingests Barchart scanner CSVs and market emails from Gmail,
scores candidates against the 5-box swing trading checklist, calls Claude API for trade
reasoning and historical compliance review, and pushes results to the Tradelog web app.

```
Gmail (market label)     Barchart CSVs (email attachments)
        │                          │
        ▼                          ▼
   E2: _gmail.py ──────────► E1: _scanner.py
        │                          │
        │ (email text)             │ (candidate symbols)
        ▼                          ▼
  E3-S1: Claude             E1-S2: _enrichment.py (IBKR Parquet)
  market summary                   │
        │                          ▼
        │                    E1-S3: _scoring.py (5-box)
        │                          │
        ▼                          ▼
        └──────────────► E3-S2: Claude trade reasoning
                                   │
                                   ▼
                         E3-S3/S4: Claude compliance + optimal trade
                                   │
                                   ▼
                         E4: _tradelog_push.py (REST API)
                                   │
                          ┌────────┴────────┐
                          ▼                 ▼
                    DailyPrep           TradeAnalysis
                  (market brief,      (per-trade review,
                   watchlist)          stored on trade)
```

---

## Decision Log

### D1: Data bridge — REST API only

**Decision:** Python communicates with Tradelog exclusively via its REST API. No direct
database access (`pyodbc` rejected).

**Why:**
- Single source of truth for data validation and authorization
- Account scoping (`X-Account-Id` header) handled consistently
- Schema changes only need to be reflected in one place
- Clean separation — Python pipeline is a client, not a privileged DB user

**Implications:**
- New read endpoint needed: `GET /api/trades/export` (bulk trade export for analysis)
- New write endpoints: `POST /api/daily-prep`, `POST /api/trades/{id}/analysis`
- Python needs the Tradelog backend running (or uses the staging Azure deployment)

---

### D2: API design — unversioned, backward-compatible JSON

**Decision:** All Tradelog API endpoints are unversioned. Changes must be backward-compatible.

**Rules:**
- No `/v1/`, `/v2/` prefixes — ever
- Adding fields to responses: always safe
- Removing fields: deprecate first (return null for 2 releases), then remove
- Adding optional query parameters: always safe
- Changing field types or semantics: never (add a new field instead)
- New endpoints: always safe

**Why:** Single consumer (Angular frontend + Python analyst). Versioning adds complexity
with no benefit for a personal project with one developer.

---

### D3: Per-trade analysis storage

**Decision:** Claude's trade analysis is stored per-trade in a `TradeAnalysis` entity,
not embedded in the `DailyPrep` JSON blob.

**Data model:**

```
TradeAnalysis
├── Id                int (PK, auto-increment)
├── TradeId           int (FK → Trade)
├── AccountId         int (global query filter)
├── AnalysisDate      date
├── Score             int (1–5)
├── Analysis          nvarchar(max)   -- markdown text (Claude's full analysis, editable)
├── Model             nvarchar(50)    -- e.g. "claude-opus-4-6"
├── CreatedAt         datetime
```

**Constraints:**
- One analysis per trade per date (unique on `TradeId + AnalysisDate`)
- Allows re-analysis over time — track if compliance improves
- `Analysis` is markdown — Claude writes structured sections (compliance, optimal trade,
  refinements) as headings. Human-readable and editable in the frontend.

**Frontend:**
- Trade detail view shows an "Analysis" section listing all `TradeAnalysis` entries for that trade
- Each entry shows: date, score (visual 1–5), model badge, rendered markdown
- Edit button switches to rich-text editor (Quill, same pattern as trade notes) for manual corrections
- If no analysis exists, the section is hidden or shows "No analysis yet"

**DailyPrep** (separate entity) holds only:

```
DailyPrep
├── Id                int (PK)
├── AccountId         int (global filter)
├── Date              date (unique per account)
├── MarketSummary     nvarchar(max)   -- JSON (regime, themes, movers, risks)
├── Watchlist         nvarchar(max)   -- JSON array of scored candidates
├── EmailCount        int
├── CandidateCount    int
├── CreatedAt         datetime
├── UpdatedAt         datetime
```

**Why separate:** Market summary and watchlist are daily snapshots (not tied to specific
trades). Trade analysis is per-trade coaching that belongs with the trade record.

---

### D4: Gmail integration

**Decision:** Label-based fetch with `gmail.readonly` scope. No email modification.

**Flow:**
1. User creates Gmail filter: market-related senders → apply label `market`, skip inbox
2. Pipeline queries Gmail API: `label:market after:{last_run_date}`
3. Downloads CSV attachments (scanner data) and extracts body text (market commentary)
4. Classifies emails by sender/subject pattern matching (configurable in `config.yaml`)
5. Tracks last successful run timestamp in `finance/apps/analyst/_state.json`

**Credentials:**
- Google Cloud OAuth2 desktop app credentials
- Client secret + token stored in `finance/apps/analyst/_credentials/` (committed — private repo)
- Scope: `gmail.readonly` only
- Gmail API does not support label-only restriction at OAuth scope level;
  code-level restriction (only query `market` label) provides the access boundary

**No modify scope needed:** Instead of marking emails as read, the pipeline tracks
`last_run_timestamp` in `_state.json` and fetches emails newer than that timestamp.

---

### D5: Claude API prompt architecture

**Decision:** Layered prompts with playbook as cached system context.

```
┌─────────────────────────────────────────────────┐
│ System prompt (~4K tokens, cached)              │
│ ├── Role definition                             │
│ ├── 5-box checklist rules                       │
│ ├── Setup types A/B/C/D with entry criteria     │
│ ├── Profit mechanisms PM-01 through PM-05       │
│ ├── Options structure matrix (by IVR)           │
│ ├── GO/NO-GO regime rules                       │
│ ├── Exit signals and management rules           │
│ └── Output format: strict JSON schema           │
├─────────────────────────────────────────────────┤
│ User prompt (per-call, task-specific)           │
│ ├── E3-S1: email text → market summary          │
│ ├── E3-S2: candidates + scores → trade reasoning│
│ ├── E3-S3: trade data + context → compliance    │
│ └── E3-S4: trade + prices → optimal trade       │
└─────────────────────────────────────────────────┘
```

**Prompt templates** stored as markdown files in `finance/apps/analyst/_prompts/`.
The system prompt is assembled from the playbook documents at build time (not at runtime)
to keep token count predictable.

**Model selection:**
- `claude-sonnet-4-6` — scanner analysis (E3-S1, E3-S2): speed + cost efficiency
- `claude-opus-4-6` — trade compliance review (E3-S3, E3-S4): deeper reasoning

**Configurable** in `config.yaml`:

```yaml
claude:
  model_scanner: claude-sonnet-4-6
  model_review: claude-opus-4-6
  max_candidates: 10
  max_trade_reviews: 5
```

---

### D6: Cost management

**Decision:** Cache results + batch limits + budget guard.

| Control | Mechanism |
|---------|-----------|
| **Result caching** | `_state.json` tracks analyzed trade IDs with timestamps. Skip unless `--force` flag. |
| **Batch limits** | Config controls `max_candidates` and `max_trade_reviews` per run. |
| **Model tiering** | Sonnet for volume work, Opus for deep analysis. |
| **Dry-run** | `--dry-run` skips Claude + Tradelog push, outputs 5-box scores only. |
| **Token tracking** | Log input/output tokens per call. Warn when monthly total exceeds configurable limit (default 500K tokens). |

**Estimated daily cost:** ~40K tokens/day ≈ $0.15–0.40 depending on model mix.

---

### D7: Launcher integration

**Decision:** Visible console subprocess + log file.

| Concern | Approach |
|---------|----------|
| **Console visibility** | Windows: `CREATE_NEW_CONSOLE` flag in `subprocess.Popen`. Non-Windows: default behavior. |
| **Blocking** | Fire-and-forget from launcher. Analyst runs in its own console. |
| **Output persistence** | Write to `finance/_data/analyst/logs/YYYY-MM-DD.log` alongside stdout. |
| **Error handling** | Partial failures logged, pipeline continues. Exit code: 0=success, 1=partial, 2=fatal. |
| **launch() contract** | Calls `_pipeline.run()` directly. Subprocess handles the rest. |

**Launcher modification:** `_launcher.py` needs a platform check to add `CREATE_NEW_CONSOLE`
for apps that don't have a GUI. Add an `APP_GUI = False` attribute to the analyst module;
the launcher reads this to decide subprocess flags.

---

## Module Structure

```
finance/apps/analyst/
├── __init__.py           # APP_DESCRIPTION, APP_ICON_ID, APP_GUI, launch()
├── config.yaml           # Default configuration (committed)
├── _config.py            # Config loader + validation
├── _state.json           # Runtime state (last run, processed IDs) — git-ignored
├── _credentials/         # Gmail OAuth2 client secret + token — committed (private repo)
├── _scanner.py           # CSV parsing with configurable column mapping
├── _enrichment.py        # IBKR Parquet data enrichment + technical indicators
├── _scoring.py           # 5-box checklist evaluation
├── _gmail.py             # Gmail API client (fetch by label, extract attachments)
├── _claude.py            # Claude API client (all analysis types)
├── _prompts/             # Prompt templates (markdown)
│   ├── system.md         # Playbook rules (cached system prompt)
│   ├── market_summary.md
│   ├── trade_reasoning.md
│   ├── compliance.md
│   └── optimal_trade.md
├── _tradelog.py           # Tradelog REST API client (read trades, push results)
├── _pipeline.py           # Orchestrator (stage execution order)
└── _models.py             # Data classes for pipeline internal data
```

---

## New API Endpoints

### Read (for Python pipeline)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/api/trades/export?status={status}&since={date}` | Bulk export closed trades with positions for analysis |

Response shape: array of `TradeExportDto` containing trade fields + linked option/stock
position summaries + existing analysis dates (to skip already-reviewed trades).

### Write (from Python pipeline)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/api/daily-prep` | Create/update daily market brief + watchlist |
| `GET` | `/api/daily-prep/latest` | Retrieve most recent report |
| `GET` | `/api/daily-prep?date={date}` | Retrieve report by date |
| `POST` | `/api/trades/{id}/analysis` | Attach Claude analysis to a specific trade |
| `GET` | `/api/trades/{id}/analysis` | Retrieve analysis history for a trade |
| `PUT` | `/api/trades/{id}/analysis/{analysisId}` | Update analysis (manual edits from frontend) |

---

## Dependencies

**New Python packages** (add to `pyproject.toml`):

| Package | Purpose |
|---------|---------|
| `anthropic` | Claude API SDK |
| `google-auth-oauthlib` | Gmail OAuth2 flow |
| `google-api-python-client` | Gmail API client |
| `pyyaml` | Config file parsing |

**Existing packages reused:** `pandas`, `numpy`, `pyarrow` (IBKR data), `requests` or `httpx` (Tradelog API calls).

**No `pyodbc` needed** — all DB access goes through the .NET REST API.

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Gmail OAuth testing mode limit (100 users) | None — personal use only | N/A |
| Azure SQL auto-pause delays | Pipeline may timeout on first trade export call | Retry with backoff (same pattern as backend startup) |
| IBKR data gaps for scanned symbols | Some candidates scored incompletely | Flag as "data missing", still include in watchlist |
| Claude output format deviation | JSON parse failure | Validate response against schema, fall back to raw text display |
| Claude API rate limits | Pipeline stalls | Sequential calls with configurable delay, retry on 429 |
| Prompt token budget exceeded | Truncated context | Pre-measure token count, trim email content if over budget |
