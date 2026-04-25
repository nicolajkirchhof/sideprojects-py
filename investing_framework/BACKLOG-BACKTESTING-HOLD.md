# Backtesting Backlog — On Hold

Strategies that are either permanently excluded from backtesting or deferred pending a specific data blocker.

Two types of entries:
- **Deferred** — blocked by missing data; will move to main backlog when unblocked
- **Permanently excluded** — backtesting is not applicable by nature; stated reason is final

---

## Category 1 — Options Strategies (Verified via Optionsomega)

DRIFT options strategies have been simulated in detail in **Optionsomega**. These are considered
verified for go/no-go purposes. No code-based backtest is required.

A **dolt options database** is available and may be explored to cross-validate specific structures
or regime adjustments, but this is supplementary research, not a verification blocker.

| Strategy | Source | Simulation status | Notes |
|----------|--------|-------------------|-------|
| Short Put (45–60 DTE, 20–30Δ) | `investor.md` | Verified — Optionsomega | Core DRIFT structure |
| XYZ Trade (PDS + short put) | `investor.md` | Verified — Optionsomega | Preferred DRIFT structure |
| Synthetic Long (ATM call + put) | `investor.md` | Verified — Optionsomega | Bullish regime only |
| PMCC | `investor.md` | Verified — Optionsomega | Income on existing longs |
| Iron condors ESTX50/XSP (regime-conditional) | `StrategyIdeasAssessment.md` | Verified — Optionsomega | Pilot within DRIFT |

**Dolt options database — exploration candidates:**
- Cross-validate ESTX50 iron condor EV vs Optionsomega simulation
- Explore IVP timeseries for ESTX50 underlyings to confirm IVP ≥ 50 filter frequency
- Test XYZ structure on ESTX50 specifically (lower drift vs SPY — does put-side VRP still dominate?)

---

## Category 2 — Fundamental Data (Dolt Earnings DB Available — Unblocked)

The **dolt earnings database** is available. H2-01 and H2-02 (accruals, F-score) are unblocked
and should move to the main backtesting backlog.

H2-03 (insider drift) remains deferred — requires a separate data source (Form 4 filings).

### Action required
- [ ] Confirm which fields are available in the dolt earnings database:
  - For accruals: `net_income`, `operating_cash_flow`, `total_assets` (quarterly)
  - For F-score: `roa`, `cfo`, `delta_roa`, `accruals`, `delta_leverage`, `delta_liquidity`, `equity_offering`, `delta_margin`, `delta_turnover`
- [ ] If fields confirmed → create BT-3-S3 (Accruals) and BT-3-S4 (F-Score) stories in `BACKLOG-BACKTESTING.md`

---

### H2-01 → TO BE MOVED: Accruals Anomaly Filter

**Strategy:** High operating accruals (net income − operating cash flow, scaled by assets) as Layer 3 filter for short candidates.
**Source:** `StrategyIdeasAssessment.md` — Idea 1, Layer 3
**Status:** Unblocked pending field confirmation from dolt earnings DB.

---

### H2-02 → TO BE MOVED: Piotroski F-Score Filter

**Strategy:** F-Score ≤ 2 as Layer 3 filter for short candidates (9 binary signals across profitability, leverage, efficiency).
**Source:** `StrategyIdeasAssessment.md` — Idea 1, Layer 3
**Status:** Unblocked pending field confirmation from dolt earnings DB.

---

### H2-03: Insider & Corporate Action Drift (PM-10) — Deferred

**Strategy:** Long stocks with significant insider purchases (Form 4, C-suite, > $100k) or buyback announcements (> 5% of float); stock in Stage 2 with RS positive.
**Source:** `TradingPlaybook.md` — PM-10 Research
**Blocker:** Form 4 filing history and buyback announcements require SEC EDGAR or OpenInsider API. Not in pipeline.
**Unblock condition:** Integrate OpenInsider API or SEC EDGAR bulk data for Form 4 filings.

---

## Category 3 — Alternative / Social Data (Deferred)

**Blocker:** Data sources are either no longer public, paid/restricted, or not yet acquired.

---

### H3-01: Retail Attention Contrarian (PM-07)

**Strategy:** Short stocks with extreme retail buying frenzies (no fundamental catalyst), entering day 2–3; 20-day hold.
**Source:** `TradingPlaybook.md` — PM-07 Research
**Blocker:** Robinhood popularity data restricted since 2021. Proxy via OTM call vol/OI ratio is possible but requires options chain history.
**Unblock condition:** Access to Quiver Quant retail flow data, OR define OTM vol/OI ratio proxy and validate against H3-02 infrastructure.

---

### H3-02: OTM Informed Flow Standalone (PM-04)

**Strategy:** Validate OTM call vol > 3× OI as a standalone directional signal (currently used as a confirmation filter only).
**Source:** `TradingPlaybook.md` — PM-04
**Blocker:** Historical OI by strike/expiry at individual stock level not in pipeline. Dolt options DB may partially cover this — explore.
**Unblock condition:** Confirm dolt options DB contains historical OI data at strike level for individual equities.

---

### H3-03: News-Driven Drift (PM-06)

**Strategy:** Long on news-confirmed > 3% moves; short on no-news > 3% moves. Hold 5–20 days.
**Source:** `TradingPlaybook.md` — PM-06 Research
**Blocker:** No news corpus in pipeline. Requires ticker-tagged news events with date and headline.
**Unblock condition:** Integrate Benzinga/Refinitiv news API, or build LLM headline classifier on IBKR news feed (Claude API).

---

## Category 4 — Permanently Excluded (Inherently Discretionary)

These strategies involve judgment that is inherently non-rule-specifiable. Market themes,
macro developments, and narrative context will always require discretionary entry decisions.
No mechanical rule can fully substitute for this without removing the edge.
These entries are final — they will not move to the main backlog.

---

### H4-01: Range-Bound Commodity Trades (NG, CL)

**Strategy:** Enter at observable multi-year range top/bottom with reversal confirmation when fundamental context supports.
**Source:** `investor.md` — opportunistic, non-core

**Why permanently excluded:** Entry requires simultaneous alignment of:
- Price at a structural range extreme (partially mechanisable)
- Fundamental context: supply/demand dynamics, seasonal cycle (requires domain judgement)
- Reversal confirmation: bar pattern at the extreme (partially mechanisable)

Mechanising all three produces a rule that fires on too many false positives. The edge
is precisely that only a subset of range touches are tradeable — and that subset requires
reading the fundamental context, which changes each time.

**What can be done instead:** Use structured checklists in the playbook to make the
decision process explicit and reviewable, without pretending it is mechanical.

---

### H4-02: Market Theme / Narrative-Driven Entries

**Strategy:** Entries driven by developing macro themes, sector rotations, or catalytic developments
(e.g., AI infrastructure buildout, energy transition, rate cycle turns) across any instrument class.
**Source:** Embedded in PM-01 (catalyst requirement), PM-06 (news-driven drift)

**Why permanently excluded:** A market theme is identified by synthesising diverse information —
news flow, earnings revisions, fund flow data, commentary — that does not reduce to a signal.
The entry timing within a theme is equally discretionary. Any attempt to backtest "theme trades"
forces a definition that either over-specifies (misses the actual trades made) or under-specifies
(fires on everything).

**What can be done instead:** Log theme-driven trades in the tradelog with a narrative tag;
review win rate and EV by tag over time. This produces empirical evidence about skill at
theme identification without requiring the theme to be defined in advance.

---

### H4-03: Long-Term ETF Portfolio (by design — no backtest applicable)

**Strategy:** Buy-and-hold globally diversified ETF portfolio on Five Factor Model. Add-only. Never sell.
**Source:** `investor.md` — Tanker sleeve

**Why permanently excluded:** This is not an alpha strategy. Its expected return is the factor
premium itself, not skill in execution. Backtesting would answer the wrong question.

**What is validated instead:**
- Factor loading of each ETF (does AVUV actually deliver small-cap value exposure?)
- TER-weighted cost vs expected factor premium (is the cost drag acceptable?)
- Geographic correlation (is diversification genuine?)
These are academic validations, not backtests.

---

## Summary

| ID | Strategy | Type | Status |
|----|----------|------|--------|
| H1-01 | DRIFT Short Put | Options | Verified — Optionsomega |
| H1-02 | DRIFT XYZ Trade | Options | Verified — Optionsomega |
| H1-03 | DRIFT Synthetic Long | Options | Verified — Optionsomega |
| H1-04 | DRIFT PMCC | Options | Verified — Optionsomega |
| H1-05 | Iron condors ESTX50/XSP | Options | Verified — Optionsomega |
| H2-01 | Accruals filter | Fundamental | **Unblocked** — confirm dolt DB fields, then move to BT-3 |
| H2-02 | F-Score filter | Fundamental | **Unblocked** — confirm dolt DB fields, then move to BT-3 |
| H2-03 | Insider drift (PM-10) | Corporate actions | Deferred — OpenInsider API needed |
| H3-01 | Retail contrarian (PM-07) | Social data | Deferred — Robinhood data restricted |
| H3-02 | Informed flow standalone (PM-04) | Options data | Deferred — check dolt options DB for OI data |
| H3-03 | News-driven drift (PM-06) | News data | Deferred — news corpus needed |
| H4-01 | Range-bound commodities | Permanently excluded | Discretionary fundamental context |
| H4-02 | Market theme / narrative entries | Permanently excluded | Themes are inherently discretionary |
| H4-03 | Long-term ETF portfolio | Permanently excluded | Not an alpha strategy; factor validation instead |
