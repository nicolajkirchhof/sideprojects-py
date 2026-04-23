# Trading Assistant — Backlog

Desktop application for the evening prep workflow. Single app for market regime,
scored watchlist, AI analysis, and ticker export. Replaces both the conditions
dashboard (`finance/apps/conditions/`) and the analyst pipeline (`finance/apps/analyst/`),
which are deleted — all logic is migrated directly into the assistant.

Target: ~1 hour evening session → export watchlists → execute during first and last market hour.

**Module:** `finance/apps/assistant/`
**Platform:** PyQtGraph desktop (reuses `_qt_bootstrap.py`, dark theme, panel layout)

---

## Module Structure

All analyst modules move directly into the assistant. The `finance/apps/analyst/`
and `finance/apps/conditions/` directories are deleted. There is no shared-library
layer — migration is a straight move with import path updates.

```
finance/apps/assistant/
    __init__.py             launch() + APP_NAME / APP_DESCRIPTION / APP_ICON_ID
    config.yaml             unified config: gmail, scanner, claude, tradelog, scoring
    _config.py              moved from analyst — updated import paths
    _models.py              moved from analyst — updated; add CandidateScore / DimensionScore
    _gmail.py               moved from analyst
    _scanner.py             moved from analyst
    _enrichment.py          moved from analyst
    _claude.py              moved from analyst
    _calendar.py            moved from analyst (ForexFactory API — already implemented)
    _pipeline.py            moved from analyst — updated to call new _scoring.py
    _tradelog.py            moved from analyst — kept for future Tradelog integration
    _scoring.py             new — replaces analyst pass/fail scorer with weighted 0–100 engine
    _data.py                migrated from conditions/_data.py — SPY/QQQ/VIX loading + slope classify
    _web.py                 moved from analyst
```

The analyst CLI entry point (`python -m finance.apps analyst`) is deleted.
The conditions launcher registration is removed from `finance/apps/__init__.py`.

---

## Error Handling Policy

**All errors halt the pipeline and are surfaced immediately.** No silent swallowing,
no partial continuation, no retry.

- Any unhandled exception in any pipeline stage stops the QThread
- The UI displays: exception type, message, and full traceback in a scrollable error dialog
- The user resolves the underlying issue and re-runs manually
- This applies to: Gmail fetch, CSV parse, IBKR enrichment, scoring, Claude calls, file I/O

This policy replaces the prior "partial failures don't block subsequent stages" design.

---

## Scoring Weights

Evidence-based weights from `/researcher` evaluation 2026-04-22:

| Dimension | Weight | Rationale |
|-----------|--------|-----------|
| D1 Trend Template | **25** | Core momentum signal — century momentum (Geczy & Samonov 2017), cross-sectional momentum (Jegadeesh & Titman 1993). Strongest academic backing. |
| D2 Relative Strength | **25** | Cross-sectional momentum IS relative strength. Equally well-documented. Together with D1, captures 50% of the signal. |
| D3 Base Quality | **15** | Timing/entry quality — practitioner evidence (Minervini VCP, BB squeeze). Reduces whipsaw but doesn't generate alpha independently. |
| D4 Catalyst | **20** | PEAD sub-component is Grade A academic (Ball & Brown 1968). Options flow moderate (Pan & Poteshman 2006). Earnings blackout is a hard constraint (score 0 on violation). |
| D5 Risk | **15** | Survival constraint, not alpha. 7% stop is a hard gate (score 0 if exceeded). |
| **Total** | **100** | |
| Tag bonus | +2/tag, cap +12 | Multi-signal convergence reward. |

**Barchart scanner filter relaxation** (decided 2026-04-22):
- Removed: % 50D MA > 0%, Slope 50D Rising, Slope 200D > 0, Weighted Alpha > 0
- Retained: Price > $5, Vol > 1M, MktCap > $200M, No earnings within 5 days
- Relaxed: ADR > 3% → ADR > 2%
- Universe: ~150 → ~300–400 stocks (Stage 1/3/4 + turnarounds visible, scored low but present)

---

## User Workflow

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

## Epic 1 — Weighted Scoring Engine

Replace pass/fail 5-box scoring with continuous 0–100 weighted scoring. Each dimension
produces a 0–weight sub-score (25/25/15/20/15). Scanner tags add bonus points (+2 each,
cap +12). Full spec with anchor tables: `investing_framework/ScoringSystem.md`.

### TA-E1-S1: Scoring model and configuration

Status: Done

**As a** swing trader,
**I want** candidates scored on a 0–100 scale with configurable weights per dimension,
**So that** I see a ranked watchlist where near-misses aren't silently dropped.

**Acceptance criteria:**
- [x] `ScoringConfig` dataclass with weight per dimension (default: 25/25/15/20/15)
- [x] `ComponentScore` dataclass: `name`, `raw_score` (0.0–1.0), `available` (bool), `source` ("ibkr" | "scanner" | "none")
- [x] `DimensionScore` dataclass: `dimension` (1–5), `name`, `raw_score` (0.0–1.0), `weighted_score` (0–weight), `components: list[ComponentScore]`, `hard_gate_fired` (bool), `partial` (bool — true if any component unavailable)
- [x] `CandidateScore` dataclass: `direction` ("long" | "short"), `dimensions: list[DimensionScore]`, `tag_bonus: float`, `total: float` (0–100+), `tags: list[str]`
- [x] Config loaded from `config.yaml` scoring section; weights must sum to 100
- [x] Hard gate semantics: gates evaluated FIRST; if any fires, `weighted_score = 0` for that dimension. Sub-components still computed for display but do not contribute to total.
- [x] Missing data policy: unavailable sub-components excluded from average (reweight over available only). If ALL sub-components unavailable, dimension = 0 + flagged "no data".
- [x] Sub-component weighting within dimensions: equal (intentional — no differential weighting)

**Affected layers:** Data model | Config
**Dependencies:** None

---

### TA-E1-S2: Dimension 1 — Trend Template scoring

Status: Done

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

### TA-E1-S3: Dimension 2 — Relative Strength scoring

Status: Done

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

### TA-E1-S4: Dimension 3 — Base Quality scoring

Status: Done

**Acceptance criteria:**
- [ ] Sub-components:
  - BB squeeze: `ttm_squeeze` == "On" = 1.0, `ttm_squeeze` == "Off" AND `bb_rank` > 80 = 0.8 (proxy for recently fired), `ttm_squeeze` == "Off" AND `bb_rank` ≤ 80 = 0.3
  - Volume contraction (VDU): contracting = 1.0, flat = 0.5, expanding = 0.2
  - SMA stack (5>10>20>50): all aligned = 1.0, 3 of 4 = 0.6, fewer = linear decay
  - ADR%: 3–7% = 1.0, 7–10% = 0.6, <3% or >10% = 0.2
- [ ] Short scoring (direction-aware, NOT simple inversion):
  - BB: expanding (`bb_rank` > 80 AND `ttm_squeeze` == "Off") = 0.8 (distribution), squeeze on = 0.5
  - Volume: expanding/flat = 0.8 (institutional distribution), contracting = 0.4
  - SMA stack: 5<10<20<50 all falling = 1.0 (mirror)
  - ADR%: same as long
- [ ] Falls back to scanner fields (`ttm_squeeze`, `bb_rank`, `adr_pct_20d`)

**Affected layers:** Scoring logic
**Dependencies:** TA-E1-S1
**Spec:** `investing_framework/ScoringSystem.md` § D3

---

### TA-E1-S5: Dimension 4 — Catalyst scoring

Status: Done

**Acceptance criteria:**
- [ ] Sub-components (from scanner data, no IBKR needed):
  - Earnings proximity: >20d → 1.0, 10d → 0.8, 5d → 0.5, <5d = HARD GATE (D4=0)
  - Earnings surprise: ≥+10% → 1.0, +5% → 0.7, 0–5% → 0.3, miss → 0.0
  - Surprise history: 4/4 beats → 1.0, 3/4 → 0.8, 2/4 → 0.5, <2 → 0.2
  - Put/Call ratio: <0.3 → 1.0, 0.5 → 0.8, 1.0 → 0.3, >1.5 → 0.0
  - RVOL: ≥3.0 → 1.0, 2.0 → 0.7, 1.5 → 0.5, 1.0 → 0.3, <0.8 → 0.0
  - IV percentile: context only, no score contribution (used for structure selection)
- [ ] Scored as MANUAL-biased: raw score is a starting estimate, Claude upgrades/downgrades in TA-E5
- [ ] Earnings blackout: <5 days long (HARD GATE D4=0), <10 days short (HARD GATE D4=0)
- [ ] Short scoring (direction-aware):
  - Earnings blackout: 10 days (vs 5 for longs)
  - Surprise: miss ≤−10% → 1.0, −5% → 0.7, beat → 0.0
  - History: first miss → 1.0, consecutive → 0.7
  - P/C ratio: >2.0 → 1.0, 1.5 → 0.8, <0.3 → 0.0

**Affected layers:** Scoring logic
**Dependencies:** TA-E1-S1
**Spec:** `investing_framework/ScoringSystem.md` § D4

---

### TA-E1-S6: Dimension 5 — Risk scoring

Status: Done

**Acceptance criteria:**
- [ ] Sub-components:
  - Stop distance (ATR% or 20 SMA distance): ≤3% = 1.0, 3–5% = 0.7, 5–7% = 0.4, >7% = 0
  - ADR vs stop: stop < 1× ADR = 1.0, 1–2× = 0.5, >2× = 0.2
  - Market cap: >$2B = 1.0, $500M–$2B = 0.7, $200M–$500M = 0.4
- [ ] 7% hard stop is a hard gate — D5 = 0 if exceeded
- [ ] Short scoring adds Short Float sub-component:
  - <10% = 1.0, 10–15% = 0.5, 15–20% = 0.2, >20% = hard gate (D5 → 0)
  - Source: scanner field `short_float`

**Affected layers:** Scoring logic
**Dependencies:** TA-E1-S1
**Spec:** `investing_framework/ScoringSystem.md` § D5

---

### TA-E1-S7: Scanner tag bonus scoring

Status: Done

**As a** swing trader,
**I want** candidates appearing on multiple scanner tags to score higher,
**So that** multi-signal convergence is rewarded.

**Acceptance criteria:**
- [ ] Tag bonus: +2 points per tag (configurable), capped at +12
- [ ] Tags displayed as badges next to the candidate in the watchlist
- [ ] Total score = sum of 5 dimensions + tag bonus (can exceed 100)
- [ ] Tags assigned per the mapping table below — evaluated after CSV ingestion and IBKR enrichment, before scoring
- [ ] Tags are non-exclusive — a stock can carry multiple tags (e.g., `[52w-high, vol-spike, pead-long]`)
- [ ] Parser strips the Barchart footer row (`"Downloaded from Barchart.com..."`) before processing
- [ ] `TTM Squeeze` column parsed as text: `"On"` = in squeeze, `"Off"` = not in squeeze (no "Fired" state exists)
- [ ] Direction assigned from tags before scoring runs (see direction rules below)

**Tag mapping:**

| Tag | Source scanner | Internal field(s) | Condition |
|-----|---------------|-------------------|-----------|
| `52w-high` | Long Universe | `high_52w_distance_pct`, `ttm_squeeze`, `rvol_20d` | `high_52w_distance_pct` > −5% AND `ttm_squeeze` == "On" AND `rvol_20d` > 1.0 |
| `5d-momentum` | Long Universe | `change_5d_pct`, `rvol_20d`, `perf_vs_market_5d` | `change_5d_pct` > 5% AND `rvol_20d` > 1.0 AND `perf_vs_market_5d` > 0% |
| `1m-strength` | Long Universe | `change_1m_pct`, `perf_vs_market_1m`, `ttm_squeeze` | `change_1m_pct` > 10% AND `perf_vs_market_1m` > 0% AND `ttm_squeeze` == "On" |
| `vol-spike` | Long Universe | `rvol_20d` | `rvol_20d` > 1.75 |
| `trend-seeker` | Long Universe | `trend_seeker_signal` | `trend_seeker_signal` == "Buy" |
| `ttm-fired` | Long Universe | `ttm_squeeze`, `rvol_20d`, `atr_pct_20d`, `bb_rank` | `ttm_squeeze` == "Off" AND `bb_rank` > 80 AND `rvol_20d` > 1.0 AND `atr_pct_20d` < 7% |
| `pead-long` | PEAD Scanner | `earnings_surprise_pct`, `change_5d_pct`, `perf_vs_market_5d`, `weighted_alpha` | `earnings_surprise_pct` > 5% AND `change_5d_pct` > 10% AND `perf_vs_market_5d` > 0% AND `weighted_alpha` > 0 |
| `pead-short` | PEAD Scanner | `earnings_surprise_pct`, `change_5d_pct`, `pct_from_50d_sma`, `short_float` | `earnings_surprise_pct` < −5% AND `change_5d_pct` < −5% AND `pct_from_50d_sma` < 0% AND `short_float` < 20% |
| `consecutive-miss` | PEAD Scanner | `earnings_surprise_pct`, `earnings_surprise_q1/q2/q3` | `earnings_surprise_pct` < 0 AND ≥2 of q1/q2/q3 < 0 |
| `ep-gap` | EP Gap Scanner | membership | Any stock appearing in `ep-gap-scanner` file |
| `rw-breakdown` | RW Breakdown | membership | Any stock appearing in `rw-breakdown-candidates` file |
| `short-squeeze` | Short Squeeze | membership | Any stock appearing in `short-squeeze` file |
| `high-put-ratio` | High Put Ratio | membership | Any stock appearing in `high-put-ratio` file |
| `high-call-ratio` | High Call Ratio | membership | Any stock appearing in `high-call-ratio` file |

**Notes on specific tags:**
- `ttm-fired`: No "Fired" state exists in Barchart exports — proxy used: squeeze recently turned Off with BB expansion (`bb_rank` > 80) and volume (`rvol_20d` > 1.0). The ATRP guard (`< 7%`) avoids tagging already-extended stocks.
- `short-squeeze`: Watch flag only (PM-11 is Grade B research, not yet validated). The +2 bonus is correctly sized — it can break ties but cannot rescue a weak candidate.
- `high-put-ratio`: **Direction-neutral.** Does not determine direction on its own — amplifies whichever direction is already assigned by other tags. On a long candidate it signals squeeze fuel; on a short candidate it confirms institutional distribution.

**Direction assignment (evaluated in order, first match wins):**
1. Has `pead-short` OR `rw-breakdown` → **short**
2. Has `consecutive-miss` AND no `pead-long` AND no `ep-gap` → **short**
3. Has any long tag (`pead-long`, `ep-gap`, `52w-high`, `5d-momentum`, `1m-strength`, `vol-spike`, `trend-seeker`, `ttm-fired`, `short-squeeze`, `high-call-ratio`) → **long**
4. Has `high-put-ratio` only (no other tags) → **long** (default)
5. Conflict (both long and short tags present) → **long** (lower-risk default)
6. No tags → **long** (default)

**Affected layers:** Scoring logic
**Dependencies:** TA-E1-S1
**Spec:** Column mapping validated against `finance/_data/barchart/screener/` CSV exports (2026-04-22). Screener definitions in `investing_framework/BarchartScreeners.md`.

---

## Epic 2 — App Shell & Data Pipeline

### TA-E2-S1: App skeleton and launcher registration

Status: Done

**As a** user,
**I want to** launch the trading assistant from the launcher or CLI,
**So that** it replaces the conditions dashboard and analyst pipeline as the single evening prep tool.

**Acceptance criteria:**
- [x] Module at `finance/apps/assistant/` with `APP_NAME = "assistant"`, `APP_DESCRIPTION`, `launch()`
- [x] Registered in `finance/apps/__init__.py`; `conditions` and `analyst` entries removed
- [x] Launches via `python -m finance.apps assistant`
- [x] Dark-themed window using `apply_dark_palette()`, 1400×900 default size
- [x] Window geometry (size, position, splitter widths) persisted via `QSettings` across restarts
- [x] Three-panel layout in QSplitter: left (300px) = Market Context, centre (flexible) = Watchlist Table, right (350px) = Detail Panel

**Affected layers:** App structure | UI
**Dependencies:** None

---

### TA-E2-S2: Data pipeline integration

Status: Done

**As a** swing trader,
**I want** the app to fetch and process scanner data on launch or on-demand,
**So that** I have fresh scored candidates each evening.

**Acceptance criteria:**
- [x] "Run Pipeline" toolbar button triggers: CSV discover → parse → IBKR enrich → score → cache (Gmail + Claude deferred to later stories)
- [x] Uses migrated modules: `_scanner.py`, `_enrichment.py`; `_scoring.py` (TA-E1 weighted engine)
- [x] Pipeline runs in a background `QThread` — UI remains responsive with progress indicator in status bar
- [x] **Error handling:** any exception in any stage stops the pipeline immediately; a scrollable error dialog shows the exception type, message, and full traceback; no retry, no partial continuation
- [x] On successful run: results stored in-memory and cached to `_data/assistant/YYYY-MM-DD.json`
- [x] On launch: if today's cache file exists, load it without re-running the pipeline
- [x] "Load CSV" button for manual CSV import (bypasses discovery; scoring runs on the imported file)
- [x] Status bar shows: current pipeline stage, candidate count, last run timestamp
- [x] "Connect to IBKR" prompt shown if IBKR Gateway is not reachable at pipeline start; pipeline halts until resolved

**Affected layers:** Data pipeline | UI
**Dependencies:** TA-E2-S1, TA-E1-S1 through TA-E1-S7

---

## Epic 3 — Market Context Panel

Left panel showing market conditions, economic events, AI summary, and DRIFT eligibility.

### TA-E3-S1: Swing regime indicators

Status: Done

**As a** swing trader,
**I want** the GO/NO-GO regime displayed in the assistant,
**So that** I have one app for all evening prep.

**Acceptance criteria:**
- [x] Logic migrated from `conditions/_data.py` into `assistant/_data.py` (conditions app deleted)
- [x] Compact display: single GO/NO-GO banner + trend rows + VIX row
- [x] Colour-coded: green banner for GO, amber for CAUTION, red for NO-GO
- [x] SPY/QQQ each show: price, 50d SMA status (green/red + slope arrow), 200d SMA status
- [x] VIX shows: level, zone badge (green/amber/red), direction arrow, spike warning

**Affected layers:** UI | Data
**Dependencies:** TA-E2-S1

---

### TA-E3-S2: Composite GO/NO-GO with stop-out counter

Status: Pending

**As a** swing trader,
**I want** a composite GO/NO-GO status with a manual stop-out counter and macro warnings,
**So that** I have a definitive answer before scanning.

**Acceptance criteria:**
- [ ] Composite: GO (all green), CAUTION (mixed), NO-GO (any red trigger)
- [ ] Each sub-signal shown with individual status
- [ ] Manual counter for consecutive stopped-out trades; ≥3 triggers NO-GO override
- [ ] Counter persists within session, resets on app restart
- [ ] Macro event proximity from economic calendar (TA-E3-S3): "FOMC in 2 days" shown inline

**Affected layers:** UI | State
**Dependencies:** TA-E3-S1, TA-E3-S3

---

### TA-E3-S3: Economic events calendar

Status: Pending

**As a** swing trader,
**I want** to see high-impact economic events for the next 5 trading days,
**So that** I know when to pause entries or reduce size.

**Acceptance criteria:**
- [ ] Uses `_calendar.py:fetch_upcoming_events(days_ahead=5, impact_filter="High")`
- [ ] Table: date, time, event name, country, impact badge (red/amber), forecast/previous
- [ ] Events within 48h highlighted with warning colour
- [ ] NO-GO keywords (FOMC, CPI, NFP) shown with red alert icon
- [ ] Compact: max 10 rows, sorted by date ascending
- [ ] Fetched once at pipeline run; cached with the rest of the daily results

**Affected layers:** UI | Calendar
**Dependencies:** TA-E2-S1
**Notes:** `_calendar.py` is already implemented using the FairEconomy ForexFactory JSON API
(`nfs.faireconomy.media`). It fetches this-week and next-week calendars automatically and
filters by impact. Moves from analyst to assistant unchanged.

---

### TA-E3-S4: Market summary from Claude

Status: Pending

**As a** swing trader,
**I want** a concise AI-generated market summary from today's emails,
**So that** I understand themes, risks, and notable movers in 2 minutes.

**Acceptance criteria:**
- [ ] Uses `_claude.py:summarize_market()` (migrated from analyst)
- [ ] Displayed as rendered markdown in a scrollable text area
- [ ] Sections: Regime assessment, Active themes, Notable movers, Risks, Action items
- [ ] "Generating..." placeholder while Claude API call runs in background
- [ ] Cached per date — doesn't re-call Claude on refresh if already generated today

**Affected layers:** UI | Claude integration
**Dependencies:** TA-E2-S2

---

### TA-E3-S5: DRIFT regime and eligibility

Status: Pending

**As an** investor running the DRIFT portfolio,
**I want** to see the DRIFT regime tier and per-underlying eligibility in the assistant,
**So that** I don't need a separate dashboard for DRIFT decisions.

**Acceptance criteria:**
- [ ] DRIFT regime tier: SPY drawdown × VIX → tier (Normal/Elevated/Correction/Deep/Bear) with BP%
- [ ] Underlying registry: Directional + Neutral blocks with tier (Core/Selective/Optional)
- [ ] Per-underlying checklist: IVP ≥ 50, Price vs 200d SMA, IV > HV → eligible/not
- [ ] Structure recommendation: short put / PDS / iron condor per underlying based on conditions
- [ ] BP guardrail: manual BP input, warning when > 50%
- [ ] Collapsible section — DRIFT is secondary to swing for the evening prep workflow
- [ ] If IBKR IV data unavailable, show "IBKR required" placeholder per underlying

**Affected layers:** UI | Data
**Dependencies:** TA-E2-S1

---

## Epic 4 — Watchlist Table

Centre panel: sortable, filterable table of scored candidates with tag badges.

### TA-E4-S1: Scored candidate table

Status: Done

**As a** swing trader,
**I want** a sortable table of all scored candidates with their dimension scores and tags,
**So that** I can quickly identify the highest-conviction setups.

**Acceptance criteria:**
- [x] `QTableView` with columns: Checkbox, Symbol, Direction (Long/Short), Score (0–100), D1 (Trend), D2 (RS), D3 (Base), D4 (Catalyst), D5 (Risk), Tags, Price, 5D %Chg, RVOL, Sector
- [x] Sortable by any column (default: Score descending)
- [x] Score cells colour-coded: green ≥70, amber 40–69, red <40
- [x] Tag badges as comma-joined string (full coloured chips deferred to Phase 5 polish)
- [ ] Row selection highlights and populates the Detail Panel (TA-E5)
- [x] Row count in status bar: "N candidates"

**Affected layers:** UI
**Dependencies:** TA-E2-S2

---

### TA-E4-S2: Filtering and search

Status: Pending

**As a** swing trader,
**I want** to filter the watchlist by score range, tags, and sector,
**So that** I can focus on relevant subsets (e.g., only `pead-long` or only Tech).

**Acceptance criteria:**
- [ ] Score range slider (0–100) filters table in real time
- [ ] Direction filter: All / Long / Short
- [ ] Tag filter: multi-select dropdown showing all active tags with counts
- [ ] Sector filter: multi-select dropdown
- [ ] Text search: symbol or sector substring match
- [ ] Filters are AND-combined; active filters shown as chips above the table
- [ ] "Reset filters" button clears all

**Affected layers:** UI
**Dependencies:** TA-E4-S1

---

### TA-E4-S3: Row checkbox and batch selection

Status: Done

**As a** swing trader,
**I want** to check/uncheck candidates in the table to build my daily watchlist,
**So that** I can select exactly which tickers to export.

**Acceptance criteria:**
- [x] Checkbox column in table; header checkbox selects/deselects all visible (filtered) rows
- [x] "Select top N" button (configurable, default 20) checks the top N by score
- [x] Selected count shown in status bar
- [x] Selection persists across sort/filter changes (tracked at model level, not view level)

**Affected layers:** UI
**Dependencies:** TA-E4-S1

---

## Epic 5 — AI Reasoning Panel

Right panel: detailed view of the selected candidate with Claude analysis.

### TA-E5-S1: Candidate detail view

Status: Pending

**As a** swing trader,
**I want** to see the full scoring breakdown and AI reasoning for the selected candidate,
**So that** I can make an informed decision about whether to include it in my watchlist.

**Acceptance criteria:**
- [ ] Header: Symbol, Direction, Price, Score, Tags
- [ ] Dimension breakdown: 5 bars (0–weight each: 25/25/15/20/15) with sub-component detail expandable; hard gate violations shown with alert icon; partial data flagged
- [ ] Scanner data: key fields from the CSV row (change %, RVOL, IV percentile, P/C ratio, earnings date)
- [ ] AI reasoning section (from Claude): setup type, profit mechanism, thesis, entry/stop/target (absolute prices), confidence
- [ ] "No reasoning available — click Analyze to generate" state when Claude hasn't been called yet
- [ ] Updates when a different row is selected in the watchlist table

**Affected layers:** UI
**Dependencies:** TA-E4-S1

---

### TA-E5-S2: On-demand Claude analysis for single candidate

Status: Pending

**As a** swing trader,
**I want** to request Claude analysis for a specific candidate by clicking a button,
**So that** I can get AI reasoning for candidates outside the auto-analyzed top N.

**Acceptance criteria:**
- [ ] "Analyze" button in the detail panel
- [ ] Calls Claude with the single candidate's data + market context (reuses `_claude.py` prompt)
- [ ] Result cached per symbol per date — doesn't re-call if already analyzed today
- [ ] Loading state while API call runs in background QThread
- [ ] **Error handling:** exception stops the call and shows error dialog (same unified policy)
- [ ] Top N candidates (configurable, default 10) are auto-analyzed during the pipeline run; others are on-demand only

**Affected layers:** UI | Claude integration
**Dependencies:** TA-E5-S1, TA-E3-S4

---

## Epic 6 — Ticker Export

### TA-E6-S1: Barchart ticker list export

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

### TA-E6-S2: TWS CSV file export

Status: Pending

**As a** swing trader,
**I want** to export selected tickers as a TWS-importable CSV file,
**So that** I can import them into a TWS watchlist via File → Import.

**Acceptance criteria:**
- [ ] "Export → TWS" button in toolbar
- [ ] Generates CSV in TWS import format: `DES,SYMBOL,STK,SMART,,,,` per line (all caps)
- [ ] Saves to `_data/assistant/tws-watchlist-YYYY-MM-DD.csv`
- [ ] Opens the saved file location in Windows Explorer (`explorer /select, <path>`)
- [ ] Toast notification: "TWS file saved: {path}"
- [ ] Only exports checked (selected) rows

**Affected layers:** UI | File I/O
**Dependencies:** TA-E4-S3
**Notes:** TWS import via right-click watchlist → "Import/Export" → "Import Financial Instruments".

---

## Implementation Order

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
  TA-E2-S1  App skeleton + launcher             (scaffold; remove conditions + analyst registrations)
  TA-E2-S2  Pipeline integration                (connects everything)

Phase 3 — Core UI panels:
  TA-E4-S1  Watchlist table                     (centre panel — most value)
  TA-E4-S3  Row checkbox + batch selection       (required for export)
  TA-E3-S1  Swing regime indicators             (left panel)
  TA-E3-S2  Composite GO/NO-GO + stop counter   (left panel)
  TA-E3-S3  Events calendar                     (left panel — _calendar.py already done)
  TA-E6-S1  Barchart export                     (immediate utility)
  TA-E6-S2  TWS export                          (immediate utility)

Phase 4 — AI enrichment + DRIFT:
  TA-E3-S4  Market summary (Claude)             (left panel)
  TA-E5-S1  Candidate detail view               (right panel)
  TA-E5-S2  On-demand Claude analysis           (right panel)
  TA-E3-S5  DRIFT regime + eligibility          (left panel collapsible)

Phase 5 — Polish:
  TA-E4-S2  Filtering and search                (UX improvement)
```

---

## Resolved Questions

1. **Scoring weights** — 25/25/15/20/15 (D1/D2/D3/D4/D5). Decided 2026-04-22 via `/researcher`.
2. **Cache strategy** — One JSON file per day in `_data/assistant/YYYY-MM-DD.json`. Decided 2026-04-22.
3. **Short-side scoring** — Single scoring function with `direction` parameter. D1/D2 use flag-based inversion. D3 direction-aware. D4 direction-aware (inverted surprise, 10-day blackout). D5 adds Short Float sub-component. Same weights. Decided 2026-04-22.
4. **IBKR enrichment dependency** — Prompt user to start IBKR Gateway; halt pipeline without it. Decided 2026-04-22.
5. **Tradelog integration** — Deferred. `_tradelog.py` migrated to assistant but not wired up. Revisit after 30 days of live use. Decided 2026-04-22.
6. **Hard gate evaluation order** — Gates bypass sub-component averaging — checked FIRST, dimension = 0 if any fires. Decided 2026-04-22.
7. **Earnings blackout threshold** — D4 long = 5 days, short = 10 days. Decided 2026-04-22.
8. **Sub-component weighting** — Equal within dimensions. Decided 2026-04-22.
9. **Scoring curve anchor points** — Full piecewise-linear anchor tables in `ScoringSystem.md`. Decided 2026-04-22.
10. **Missing data policy** — Exclude unavailable sub-components, reweight over available. Decided 2026-04-22.
11. **Module integration** — Analyst and conditions apps deleted; all code migrated directly into `finance/apps/assistant/`. No shared-library layer. Analyst CLI removed. Decided 2026-04-22.
12. **Error handling** — Unified STOP + show error policy. Any pipeline exception halts and surfaces full traceback. No partial continuation. Decided 2026-04-22.
13. **Calendar source** — FairEconomy ForexFactory JSON API (`nfs.faireconomy.media`). Already implemented in `_calendar.py`. Not manual input. Decided 2026-04-22.
14. **Tag-to-column mapping** — Derived from scanner CSV column values, not filenames. Full mapping TBD pending CSV review session. Decided 2026-04-22.
15. **Window state persistence** — `QSettings` for geometry and splitter widths. Decided 2026-04-22.
16. **Historical runs** — Today's cache auto-loaded on launch. No explicit UI for viewing previous days in Phase 1. Decided 2026-04-22.
17. **TTM Squeeze encoding** — Binary text values: `"On"` (in squeeze) and `"Off"` (not in squeeze). No "Fired" state exists in Barchart CSV exports. `ttm-fired` tag uses proxy: `ttm_squeeze` == "Off" AND `bb_rank` > 80 AND `rvol_20d` > 1.0 AND `atr_pct_20d` < 7%. Decided 2026-04-22.
18. **Tag mapping — full column conditions** — Validated against actual CSV exports from all 7 scanners (2026-04-22). Three schemas (Standard/PEAD-EP/Options-Flow). Membership-based tags for EP Gap, RW Breakdown, Short Squeeze, High Put Ratio, High Call Ratio. Column conditions for Long Universe and PEAD scanner tags. Full table in TA-E1-S7.
19. **`high-put-ratio` direction** — Direction-neutral tag. Does not assign direction independently; amplifies whichever direction other tags already establish. Decided 2026-04-22 via `/researcher`.
20. **Schema C extra columns** — `1M P/C Vol`, `Short Int`, `Days2Cover`, `5D IV Chg`, `1M IV Chg` are display-only context columns. Not scoring inputs. `Short Float` (already in D5 as hard gate) is the only Schema C column that contributes to scoring. Decided 2026-04-22 via `/researcher`.
