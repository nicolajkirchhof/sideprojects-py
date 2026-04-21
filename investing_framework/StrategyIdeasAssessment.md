# Strategy Ideas Assessment

> Independent evaluation of new strategy directions against documented edges.
> Lens: does a real edge exist, and can it be systematized?

**Date:** 2026-04-18
**Context:** Ideas from `IDEAS.md` evaluated by the `/researcher` framework.
**Key constraint:** Strategies are only viable as automated systems — no manual intraday execution.
The evaluation lens is therefore *automation feasibility*, not trader attention cost.

---

## Evaluation Framework

Each idea is rated on four dimensions (1–5 each):

| Dimension | 1 | 5 |
|-----------|---|---|
| Edge quality | Practitioner lore only | Replicated academic edge |
| Risk/reward | Unclear or negative EV | Clear positive EV, defined max loss |
| Automation feasibility | Discretionary — cannot be rules-based | Fully rule-based, backtestable |
| Portfolio diversification | High overlap with current book | Uncorrelated alpha source |

**Score ≤ 12: Do not pursue. 13–16: Pilot cautiously. 17–20: Add to playbook.**

---

## Active Strategy Context

Any new idea is evaluated for fit against the current book:

| Strategy | What it captures | Gaps |
|----------|-----------------|------|
| **Momentum Swing** — 5–50 day, stocks + options | Momentum, PEAD, pre-earnings drift, EP/VCP | Short exposure, futures, intraday |
| **DRIFT** — 40–60 DTE short puts/XYZ on SPY/QQQ/IWM/ESTX50/GLD/SLV | VRP on major indices, positive drift | Intraday options, single-stock, iron condors |
| **Long-Term ETF** — Five Factor, add-only | Market beta, size, value, EM, gold | Not a short-term alpha source |

---

## Idea 1 — Factor-Driven Short Strategy

**Concept:** Build short exposure from documented negative-return factors — not from price pattern triggers.
The trigger is a measurable fundamental or event signal, not a support break.
Price action is a confirmation layer, not the edge source.

> Short selling works when you know *why* a stock should decline, not just that it has declined.

---

### The Framework: Four Signal Layers

A short candidate must pass through layers in order. Each subsequent layer adds conviction; a candidate can be acted on with Layer 1 + 2 confirmed. Layers 3–4 increase size or shorten hold.

---

#### Layer 1 — Universe Filter (always on)

Eliminates structural short-squeeze and gap-risk candidates before any signal is assessed.

| Filter | Rule | Rationale |
|--------|------|-----------|
| Trend stage | Price below declining 30-week MA (Weinstein Stage 3/4) | Institutional distribution already underway |
| Short interest | < 20% of float short | Above this: squeeze risk dominates any edge |
| Liquidity | Avg daily volume > 500k shares | Ensures borrow availability and manageable slippage |
| Binary events | No earnings, FDA, M&A within 10 days | Eliminates gap risk from scheduled announcements |
| Borrowability | Locate available at < 5% annualised | Cost directly reduces EV |

---

#### Layer 2 — Primary Trigger (event-driven, academically documented)

These are the core edge sources. At least one must be present to enter.

**Trigger A — Negative PEAD**

> The mirror of the long PEAD trade already in the swing playbook.

- Academic basis: Ball & Brown (1968), Bernard & Thomas (1989), Skinner & Sloan (2002)
- Signal: negative earnings surprise (negative SUE) + stock closes in bottom 25% of day's range on earnings day
- Mechanism: institutions reduce positions gradually over weeks as analysts revise estimates downward; the drift is as persistent on the downside as the upside
- Hold period: 20–60 days; drift exhausts as consensus converges to the new earnings level
- Entry: first recovery day after the earnings gap (not the gap itself — avoids the initial panic spike and improves fill quality)

**Trigger B — Consecutive Miss + Guidance Cut**

- Academic basis: Chordia & Shivakumar (2006); momentum in analyst revisions
- Signal: 2+ consecutive earnings misses AND management guides below consensus for next quarter
- Mechanism: successive misses indicate a structural deterioration, not a one-off; guidance cut forces institutional selling as fund mandates are violated
- Hold period: 30–90 days (analyst revision cycle)

---

#### Layer 3 — Fundamental Quality Filter (increases conviction)

These signals confirm the fundamental thesis. Use to increase size, not as standalone triggers.

**Accruals Anomaly (Sloan 1996)**
- One of the most replicated anomalies in finance (Richardson et al 2005; Hirshleifer et al 2004)
- Signal: operating accruals in the top quartile of the universe = reported earnings significantly exceed cash earnings
- Formula: Accruals = Net Income − Operating Cash Flow, scaled by total assets
- Interpretation: high accruals mean earnings are being inflated by accounting choices; mean-reversion is the expected outcome over 1–4 quarters
- Source: quarterly 10-Q filings; data available from Compustat / financial data providers

**Short Interest Flow (Cohen, Diether & Malloy 2007)**
- The *change* in short interest is more predictive than the level
- Signal: short interest increasing over last two reporting periods (bi-monthly FINRA data)
- Interpretation: informed shorters (typically hedge funds with fundamental research) are adding — not retail panic
- Avoid: stocks where short interest is already at extreme levels (>20% float) — that is squeeze fuel, not signal

**Low F-Score (Piotroski 2000)**
- F-Score ≤ 2 out of 9 = deteriorating fundamentals across profitability, leverage, and efficiency
- Nine binary signals; fully codeable from public financial statements
- Most predictive as a short signal when combined with a price-momentum trigger

---

#### Layer 4 — Price Confirmation (entry timing)

Not the edge — just timing the entry to avoid buying into the initial selling panic.

| Condition | Rule |
|-----------|------|
| Entry timing | Enter on first 1–3% recovery from the post-event low (reduces initial spike fill) |
| Stage confirmation | Price below 30-week MA, MA slope negative or flat |
| Sector context | Sector in Stage 3/4 overall — avoids shorting individual weakness in a rising sector |

---

### Tail Risk Mitigation

The structural problem with short selling — uncapped upside gap risk — is not eliminated but can be managed:

| Risk | Mitigation |
|------|------------|
| Overnight gap up (takeover, FDA, beat) | Use **put spreads** (debit spreads) instead of naked shorts — max loss is spread width, not infinite |
| Short squeeze | Layer 1 squeeze filter (< 20% float short); exit immediately at +15% against position |
| Borrow recall | Size conservatively; treat borrow availability as a real constraint, not a given |
| Correlation with long book | Long PEAD longs and short PEAD shorts will both move on the same earnings season — this is a *feature*, not a bug, if managed as a long/short pair |

**Preferred structure for automation:** Put debit spreads (buy ATM put, sell OTM put at 1–2 sigma below) define max loss precisely, eliminate the overnight gap tail, and make position sizing deterministic. The trade-off is reduced credit vs naked short — but for a systematic strategy, defined loss is worth the cost.

---

### Assessment

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Edge quality | 4/5 | Negative PEAD + accruals are among the most replicated factor anomalies |
| Risk/reward | 3/5 | Defined with put spread structure; tail risk managed but borrow cost real |
| Automation feasibility | 4/5 | All signals are quantifiable from public data; borrow availability check is the one non-trivial piece |
| Portfolio diversification | 4/5 | Currently zero short exposure; adds genuine negative-delta diversification |
| **Total** | **15/20** | ~~Pilot cautiously~~ → **Validated (April 2026)** |

### Backtest Results (April 2026)

All backtests run on 2016–2026 data, screener-aligned universe (Price>$5, Vol>1M).

**Layer 2A — Negative PEAD (primary trigger):**
- All misses (N=5,169): -2.57%/1d, -2.20%/10d, fades by day 40. **Short PEAD is a 10–20 day trade.**
- Strong short (miss + gap<=-5% + bottom 25% close, N=418): -15.97%/1d, -15.41%/40d, -12.00%/60d. **Persistent drift for the filtered signal.**

**Layer 2B — Consecutive Miss:**
- Counter-intuitive: 1st miss (-3.17%/1d) is the **strongest** short signal. 2nd miss: -1.82%/1d. 3rd+: -1.08%/1d. Bad news is already priced in by the repeat miss. **Focus on first-miss quality, not miss count.**

**Layer 3 — Accruals (Sloan 1996):**
- 59,536 quarterly observations, 10 deciles. Long-short spread: +3.38%/252d. Negative at shorter horizons. **Annual rebalance signal — use as quarterly conviction filter, not swing entry.**

**Layer 3 — F-Score (Piotroski 2000):**
- 197,250 observations. F>=8 outperforms F<=2 by ~0.5%/quarter. **Quality filter, not standalone alpha. Use F>=7 to upgrade swing conviction, F<=3 to upgrade short conviction.**

**SPY context matters:** Non-supporting regime amplifies the miss drift. Short misses in non-supporting SPY: -3.90%/1d vs -1.23% in supporting.

**Verdict upgraded:** The condition for upgrade (positive net EV on negative PEAD after costs) is met. The strong short signal (-12%/60d at 76.8% win rate) exceeds the threshold. **Promote to active trading — paper trade 30 positions, then live at half size.**

---

### Academic References

| Signal | Paper |
|--------|-------|
| Negative PEAD | Ball & Brown (1968); Bernard & Thomas (1989); Skinner & Sloan (2002) |
| Consecutive miss / guidance | Chordia & Shivakumar (2006) |
| Accruals anomaly | Sloan (1996); Richardson, Sloan et al (2005) |
| Short interest flow | Cohen, Diether & Malloy (2007) |
| F-Score | Piotroski (2000) |
| Short-side momentum (general) | Israel & Moskowitz (2013) |

---

## Idea 2 — Intraday Breakout Strategy (Futures + Stocks)

**Concept:** Automate breakout entries at the opening range or intraday consolidation breakouts for same-session trades.

### Edge

The opening range breakout has the strongest academic foundation of any intraday pattern:

- Toby Crabel (1990): original documentation of ORB edge in futures
- Lou, Polk & Skouras (2019): first-30-minute return predicts last-30-minute direction in SPX — aggregate microstructure edge confirmed independently
- Overnight vs intraday return split (Cliff, Cooper & Gulen 2008): equities earn positive returns overnight, negative intraday — the ORB is capturing the inflection at open

**Critical qualification:** These are aggregate statistical effects. They describe a marginal edge across many observations — not a reliable per-trade signal. The edge exists in expectation over many trades; individual ORB trades are noisy.

### Risk/Reward

- EV is positive in the literature but thin — transaction costs are the primary variable. A 2-tick bid-ask on ES ($25 round-trip) vs a 4-tick stop ($50) requires the entry to have a genuine edge just to break even net of friction
- Tail risk is defined — intraday stops bound the loss per trade. The strategy-level risk is a run of correlated stops on high-volatility days
- Futures provide leverage: correct sizing is essential; one over-leveraged session can distort the statistics

### Automation Feasibility

This is the highest-automation-feasibility idea on the list:

- Entry trigger: mathematically defined (high/low of first N candles)
- Stop: rule-based (below ORB low)
- Target: rule-based (fixed R multiple or close-of-session)
- No discretion required — the rules are fully specifiable

The primary engineering challenge is execution quality: slippage on fast ORB moves is the difference between edge and no-edge. Limit orders may not fill; market orders may fill at poor prices on fast breaks.

### Assessment

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Edge quality | 3/5 | Academic backing exists; thin but real after costs if execution is tight |
| Risk/reward | 3/5 | Defined loss, positive EV in expectation — sensitive to execution quality |
| Automation feasibility | 5/5 | Fully rule-based; highest automation potential of all four ideas |
| Portfolio diversification | 4/5 | Different timeframe and instrument class from current book |
| **Total** | **15/20** | **Pilot cautiously** |

**Path to pilot:**
1. Backtest on futures tick data with realistic fill assumptions (not mid-price)
2. Define universe: one or two instruments only (ES/NQ or ESTX50); not all instruments simultaneously
3. Paper-trade for 60 sessions with live data before committing capital
4. The go/no-go metric: net P&L after slippage > 0 over the pilot period

**The condition that would change this verdict upward:** Demonstrated positive net-of-costs EV in backtest. This is the one idea where automation is straightforward enough to test rigorously before any capital commitment.

---

## Idea 3 — Situational Intraday Strategies (Hougaard Research)

**Concept:** Automate Hougaard's named "situational" strategies — time- and event-triggered bracket setups on European index futures.
The original verdict ("fundamentally discretionary") was wrong. After reading the source material, several strategies are **fully mechanical**: fixed signal bar, OCO bracket entry, defined stop. The psychological framework is irrelevant to automation.

> Full extraction in `_research/TomHougaard/RESEARCH_SUMMARY.md`

---

### The Named Strategies — Automation Feasibility

| Strategy | Instrument | Signal Bar | Entry | Stop | Exit | Automation |
|----------|-----------|------------|-------|------|------|------------|
| **SRS** | DAX | 2nd 15-min bar after 09:00 | OCO bracket ±2pts | Bar extreme or 20% ATR | Mechanical trail | Full |
| **ASRS** | DAX | 4th 5-min bar after 09:00 | OCO bracket ±2pts | Opposite bracket side | 2-bar trailing stop | Full |
| **1BN / 1BP** | FTSE, Nasdaq | 1st 5-min bar at open | OCO bracket on bar close | Bar range | ATR trail or 2-bar stop | Full |
| **Rule of 4** | Dow, SP500 | 4th 10-min bar post-FOMC | OCO bracket | Opposite bracket side | Mechanical trail | Full (event-triggered) |
| **4-Bar Fractal** | Any | Close > prior bar 2 high AND bar 4 high | On bar close | Opposite fractal level | Trail | Full |

All five use the same structural template: **wait for signal bar → bracket both sides → whichever triggers is the trade → defined stop from day one**. The discretionary element in Hougaard's personal trading is exit management — but three mechanical exit options are documented and usable:

- **ATR trailing stop**: 20% of daily ATR(14) for intraday; update each bar
- **2-bar stop**: trail stop to the high/low of the two most recent completed bars
- **Over-balance stop**: exit when current pullback exceeds the size of prior pullbacks (structural trend break)

---

### Edge Assessment

The strategies are not academically replicated in the traditional sense, but they have structural backing:

**SRS / ASRS — DAX opening bracket**
- Mechanism: the first 15–30 minutes of the DAX session are dominated by market-maker order execution for clients; the "real" trend emerges once that flow is absorbed
- This aligns with Lou, Polk & Skouras (2019): first-period returns at session open predict subsequent direction
- 57% historical win rate on SRS (stated in source material); EV positive if avg win > avg loss at that rate

**1BN / 1BP — First Bar Direction**
- 58-month FTSE dataset (1,286 trading days): both signals documented with statistics
- August 2024 1BP on FTSE: 60% win rate, 194 pts profit on winners vs 31 pts on losers (6.2:1 payoff)
- August 2024 1BP on Nasdaq: 8 of 10 triggered days produced "great or better" results
- DAX 1BP: explicitly flagged as not working — instrument-specific effect, not universal

**Rule of 4 — Post-FOMC bracket**
- Mechanism: post-announcement price action resolves the uncertainty in the 4th bar as the initial noise clears
- Structurally identical to the ORB hypothesis on a news-triggered session open
- Small sample (handful of live FOMC dates documented) — treat as a hypothesis, not a validated edge

---

### Risk/Reward

- Max loss is defined on every trade by the bracket structure — no undefined tail risk
- Stop is set at the time of order placement, before any fill; cannot be moved wider
- Leverage risk: futures position sizing using the ATR formula (stake = monetary risk ÷ points at risk) contains this
- Strategy-level risk: correlated losses on high-volatility gap opens — the bracket is triggered but market immediately reverses through the stop; this is the main failure mode on news days
- Mitigation: do not run SRS/ASRS on days with red-flag news at or before 09:00 Frankfurt (pre-market filter using Forex Factory)

---

### Instrument Notes Relevant to Automation

| Instrument | Character | Recommended Strategies | Notes |
|-----------|-----------|----------------------|-------|
| **DAX** | Momentum-oriented, large moves sustain | SRS, ASRS | 40-pt SRS stop scales with DAX level; reduce size 50% when signal bar range >30 pts |
| **FTSE** | Back-and-fills, overlapping bars, commodity-heavy | 1BN, 1BP | Aggressive trailing (1:1 trail); move stop to BE at 6–10 pts profit; do not scale in |
| **Nasdaq** | High volatility, fast moves | 1BP | Wide parameters; 8/10 August 2024 result; validate over longer window |
| **Dow / SP500** | Rule of 4 only | Rule of 4 | FOMC dates only; insufficient data to run as systematic strategy |

---

### Relationship to Idea 2 (Generic Intraday ORB)

These strategies overlap with Idea 2 in principle but differ in specificity:

| | Idea 2 — Generic ORB | Idea 3 — Hougaard Situationals |
|--|---------------------|-------------------------------|
| Signal bar | Configurable (first N minutes) | Fixed by strategy rule (4th 5-min, 2nd 15-min, etc.) |
| Instruments | ES, NQ, ESTX50 (US focus) | DAX, FTSE, Nasdaq (European open focus) |
| Data | Requires tick-data backtest | 58-month FTSE dataset already exists in source material |
| Validation | No published statistics | Win rate and payoff documented for 1BP |

They are complementary, not competing. The Hougaard strategies fill the **European session open window** (08:00–11:00 UK); a generic ORB strategy covers the **US session open** (14:30 UK).

---

### Assessment

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Edge quality | 3/5 | Practitioner-documented with multi-year data; structural backing from opening-range microstructure; not independently replicated |
| Risk/reward | 4/5 | Fully defined max loss on all strategies; exit rules available mechanically; FTSE 1BP shows strong payoff asymmetry |
| Automation feasibility | 4/5 | Entry and stop are fully mechanical; exit rule needs one of three mechanical options selected per strategy |
| Portfolio diversification | 4/5 | European session, index futures, intraday — zero overlap with current book |
| **Total** | **15/20** | **Pilot cautiously** |

**Recommended pilot — 1BP on FTSE:**
- Most data available (58-month dataset, 1,286 days)
- Simplest entry: classify first 5-min bar direction → place bracket
- Stop is fully defined (bar range) before any fill
- Exit rule: 2-bar trailing stop (fully mechanical)
- Run against historical FTSE 5-min data first; if EV > 0 net of spread, move to live paper trading

**The condition that changes this verdict upward:** Independent validation of the 1BP edge over a full market cycle (including 2022 bear + 2020 volatility spike), not just the August 2024 sample used in the source documents. If EV holds across regimes, this is a playbook addition.

---

### Source Material

Full rule extraction: `_research/TomHougaard/RESEARCH_SUMMARY.md`

Key source documents:
- `School Run Strategy December 2022.pdf` — SRS full rules
- `Advanced School Run Strategy.pdf` — ASRS full rules
- `1BN & 1BP FTSE STRATEGY.pdf` — 58-month dataset, entry/stop/exit rules
- `ALL-YOU-EVER-NEED-TO-KNOW-ABOUT-STOP-LOSS-PLACEMENT.pdf` — mechanical exit options
- `Tom Hougaard - Scalping Stock Indices (Trade on BarClose).pdf` — volatility-adaptive parameters

---

## Idea 4 — Intraday Options: Iron Condors on ESTX50 / XSP

**Concept:** Sell iron condors (OTM call + OTM put) on ESTX50 and XSP to capture the variance risk premium at shorter timeframes.

### Edge

The variance risk premium (VRP) is one of the best-documented edges in options:

- IV systematically overstates realised volatility (Carr & Wu 2009; Bollerslev et al 2009)
- Tastytrade empirical data (2012–2022, SPX): iron condors at 45 DTE managed at 50% profit produce ~70% win rate; 45 DTE outperforms ≤21 DTE on Sharpe basis due to gamma risk near expiry

**This is the same edge as DRIFT.** The question is not whether a new edge exists — it is whether the iron condor structure is superior to the current XYZ structure for these underlyings.

### Iron Condor vs XYZ — The Key Comparison

| | Iron Condor | XYZ / Short Put |
|--|-------------|-----------------|
| Delta | Neutral | Slightly negative to neutral |
| VRP capture | Both sides (put + call) | Put side only (larger premium) |
| Positive drift | Caps upside — works against drift | Benefits from drift |
| Max loss | Defined (spread width − credit) | Undefined (naked put) or defined (spread) |
| ESTX50 fit | Better — drift is weaker in EU equities | Better for SPY/QQQ where drift is stronger |

**The put-side VRP is larger than the call-side VRP** due to persistent put skew. Adding a short call leg captures a smaller premium while capping upside — a net negative in trending/drifting markets.

**ESTX50 is the exception:** European equity drift is materially lower than US equity drift. The cost of selling the call wing (capping upside) is lower when the underlying drifts less. Iron condors are more defensible on ESTX50 than on SPY/QQQ.

### Automation Feasibility

- Entry rules: fully specifiable (DTE, delta strikes, IVP filter)
- Management rules: fully specifiable (50% profit close, 200% stop, 21 DTE roll)
- This is the lowest-friction addition to an automated book because the infrastructure already exists in DRIFT

### Assessment

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Edge quality | 4/5 | VRP is academically robust; iron condor is a valid harvesting structure |
| Risk/reward | 3/5 | Defined max loss is an improvement over naked puts; capped upside is a cost |
| Automation feasibility | 5/5 | Fully rule-based; same infrastructure as existing DRIFT trades |
| Portfolio diversification | 2/5 | Same edge as DRIFT — regime variation, not diversification |
| **Total** | **14/20** | **Pilot cautiously** |

**Recommended framing:** Do not add iron condors as a separate new strategy. Add them as a **regime-conditional structure within DRIFT**:

- ESTX50 range-bound (price inside 20-day ATR, no defined trend): use iron condor
- ESTX50 trending up: use XYZ or synthetic long (positive delta benefits)
- SPY/QQQ/IWM: keep XYZ as baseline; the US drift premium makes short calls costly

**The condition that would change this verdict:** If ESTX50 demonstrated negative or flat drift over the test period, the iron condor becomes the clear structure of choice for that underlying permanently.

---

---

## Proprietary Backtested Research — `finance/intraday_pm`

Strategies already implemented and evaluated in code. Three tiers:

| Tier | Content |
|------|---------|
| **Tier 1 — Tradeable candidates** | Full entry/exit/evaluation framework already built; can be wrapped for live execution |
| **Tier 2 — Filter signals** | Research outputs useful as regime or timing filters for other strategies |
| **Tier 3 — ML explorations** | Hypothesis stage; no actionable signal yet |

---

### Tier 1 — Tradeable Candidates

#### Following Range Break (`future_following_range_break.py`)

**What it does:** Trend-following entry on candle breakout; trail stop through subsequent candles until price fails to continue.

**Entry logic:**
- Long: current candle high > prior candle high → enter at prior candle high; stop at `min(prior candle low, current candle low)`
- Short: current candle low < prior candle low → mirror image

**7 stop variants tested** (offset applied to prior candle low/high as stop buffer):

| Variant | Stop offset |
|---------|------------|
| `S_cbc` | At prior candle low/high (zero offset) |
| `S_cbc_10_pct` | +10% of prior candle ATR (looser) |
| `S_cbc_20_pct` | +20% of ATR (loosest) |
| `S_cbc_10_pct_up` | −10% of ATR (tighter than candle close) |
| `S_cbc_20_pct_up` | −20% of ATR (tightest) |
| `S_01_pct` | 0.1% of prior candle open |
| `S_02_pct` | 0.2% of prior candle open |

**Instruments:** IBDE40, IBGB100, IBES35, IBJP225, IBUS30, IBUS500, IBUST100
**Timeframes:** 2m, 5m, 10m bars
**Evaluation coverage:** 2023–2025; per-variant metrics: loss sum/count/%, candle duration, move mean/median/std, max favourable excursion vs loss

**Status:** Active. Evaluation runs against 2024 pkl files. Results are rendered at runtime — no hardcoded EV.

**For the assessment:** Entry logic is fully codeable. The 7-variant study is designed to find the optimal stop-tightness per timeframe and instrument. Once the optimal variant is identified from evaluation output, the strategy is ready for live execution wrapping.

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Edge quality | 3/5 | Standard breakout trend-following; no academic replication but evaluation framework exists |
| Risk/reward | 3/5 | Defined stop per trade; 7 variants under evaluation — optimal parameters not yet selected |
| Automation feasibility | 5/5 | Fully rule-based; code already written; needs only a live execution wrapper |
| Portfolio diversification | 4/5 | Intraday European indices; zero overlap with current book |
| **Total** | **15/20** | **Pilot — read evaluation output, select optimal variant, wrap for live execution** |

---

#### VWAP Extrema Bracket (`futures_vwap_extrema.py`)

**What it does:** For each 5-minute candle, simulate a bracket entry (buy-stop above high, sell-stop below low) and measure which side triggers first and how far price travels to the next VWAP3 swing extremum.

**Key metrics computed per candle and time slot:**
- `bracket_entry_in_trend` — did the triggered bracket side match the actual direction of the next swing?
- `success_rate = bracket_entry_in_trend / all_count` — effectively the directional win rate at each time of day
- `pts_move` — absolute points from extremum to next extremum (the "win" if in-trend)
- `sl_pts_offset` — adverse excursion during the move (the "loss" if against trend)
- In-trend aggregate points vs out-of-trend aggregate points — total EV estimation by time slot

**Candle pattern flags:** `is_doji`, `is_oii` (outside-inside-inside 3-bar), `is_high_lh`/`is_high_oc` (ATR expansion)

**Instruments:** IBDE40, IBEU50, IBES35, IBGB100, IBUS30, IBUS500, IBUST100, IBJP225, USGOLD
**Timeframes:** 5m, 10m, 15m bars; extended-hours variant (`_ad`)

**Status:** Active. Results rendered to PNG files per symbol and time slot. The success-rate output identifies which times of day have directional bracket edge.

**For the assessment:** This strategy is generating a per-time-slot win-rate map across 9 instruments. Once the high-edge time slots are identified from saved PNGs, those slots can be isolated as the entry universe for a live bracket strategy — essentially a time-filtered ORB.

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Edge quality | 3/5 | Microstructure-based; bracket-at-VWAP-extrema aligns with ORB academic backing; not independently replicated |
| Risk/reward | 3/5 | Defined per bracket; time-slot filtering from evaluation output should improve EV |
| Automation feasibility | 4/5 | Data framework already built; entry trigger needs time-slot filter derived from evaluation output |
| Portfolio diversification | 4/5 | 9 instruments, intraday, different from all current strategies |
| **Total** | **14/20** | **Pilot — extract high-edge time slots from evaluation PNGs; implement filtered bracket** |

---

#### Noon Iron Butterfly (`paused/__noon_to_close_evaluation.py`)

**What it does:** Short iron butterfly at noon (≈3 hours before close), using Black-Scholes to price the wings at ±2σ of expected same-day move. Profits if price stays inside the wings at close.

**Structure:**
- Buy wing put at noon close − 2σ (protection)
- Sell ATM put + ATM call at current spot (premium collection)
- Buy wing call at noon close + 2σ (protection)
- Hold to close; P&L = credit received − intrinsic value at close

**σ formula:** `IV / sqrt(252) × sqrt(0.5/365)` — scaling IV to the remaining half-day

**Instruments:** DAX, ESTX50, SPX (where IV data available)
**P&L tracking:** Per trade, aggregated by year/month/week/weekday; win/loss summary with avg win and avg loss

**Current status:** Paused. Most developed backtest in the directory — Black-Scholes pricing is implemented and working. Paused likely due to data/IV feed issues.

**Relationship to DRIFT:** Same edge source (VRP) but different timeframe (intraday vs 45–60 DTE). The iron butterfly is a same-day theta harvest — much higher gamma risk than DRIFT's 45 DTE structures. The win rate will be high (price usually stays near noon) but tail losses on high-volatility days will be larger as a percentage of premium collected.

**For the assessment:** The infrastructure is built. The paused status should be investigated — if it is a data-feed issue rather than a strategy failure, this may be worth resuming with the existing IBKR intraday data pipeline.

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Edge quality | 4/5 | VRP edge — same academic basis as DRIFT; Black-Scholes pricing already implemented |
| Risk/reward | 2/5 | Same-day expiry = high gamma; tail losses on volatile days can exceed premium collected; wings are further OTM than DRIFT stops |
| Automation feasibility | 4/5 | Backtest already priced with Black-Scholes; needs live options data feed |
| Portfolio diversification | 2/5 | Same VRP edge as DRIFT; intraday vs overnight is a timeframe difference, not a strategy difference |
| **Total** | **12/20** | **Investigate why paused; resume only if tail risk on high-IV days is acceptable after reviewing P&L distribution** |

---

### Tier 2 — Filter Signals

These are not standalone strategies but generate signals that improve other strategies.

#### Day-of-Week Bias (`thursday_friday_monday.py`)

- Measures probability of price making a new high above prior close (`hc`) vs new low below prior close (`lc`) on each weekday, conditioned on prior day's bar structure
- Instruments: DAX, ESTX50, SPX, INDU, NDX, Gold; 2017–2025 (8 years)
- **Use as:** Pre-trade directional bias filter for intraday bracket strategies — if Thursday shows >60% `hc_pct` historically, weight long brackets higher on Thursday

#### Prior Day's Close Proximity (`futures_close_to_min.py`)

- Measures how close price returns to prior day's close (PDC) at each 30-minute window through the session
- Instruments: ESTX50, SPX, INDU, NDX, N225
- **Use as:** Entry timing filter — identifies the session windows where price is most likely to be near PDC, enabling better entry timing for mean-reversion setups near prior day's close

#### Post-Extreme-Day Behaviour (`underlying_extreme_days.py`)

- Identifies days with ±2% daily move; measures forward returns 1–4 weeks
- Instruments: DAX, ESTX50, SPX, INDU, NDX, Gold
- **Use as:** DRIFT entry timing signal — if extreme-day mean-reversion is positive in weeks 1–2, extreme-down days are elevated-priority entry points for new DRIFT short put positions

---

### Tier 3 — ML Explorations (not actionable yet)

| File | Hypothesis | Status |
|------|-----------|--------|
| `school_run_evaluation.py` | First 60-min bar shape predicts DAX close direction | Feature R² near zero; no predictive signal found |
| `change_to_close.py` | Morning 30-min candle dynamics predict afternoon move size | ML models fitted; results printed at runtime; no published findings |
| `futures_experimenting.py` | Intraday swing structure mapping with pullback geometry | Exploratory visualization; no trading signals defined |

These remain research hypotheses. Revisit when evaluation outputs indicate a predictive signal above noise.

---

## Summary

| Idea | Edge | R/R | Automation | Diversification | Score | Verdict |
|------|------|-----|------------|-----------------|-------|---------|
| Factor-driven short (neg. PEAD + accruals) | 4/5 | 4/5 | 4/5 | 4/5 | **16** | **Validated** — paper trade → live |
| Intraday ORB (US session, generic) | 3/5 | 3/5 | 5/5 | 4/5 | **15** | Pilot — tick backtest first |
| Situational intraday (Hougaard, EU session) | 3/5 | 4/5 | 4/5 | 4/5 | **15** | Pilot — 1BP on FTSE first |
| Iron condors ESTX50/XSP | 4/5 | 3/5 | 5/5 | 2/5 | **14** | Pilot — within DRIFT |
| Following Range Break (`intraday_pm`) | 3/5 | 3/5 | 5/5 | 4/5 | **15** | Read evaluation output → wrap for live |
| VWAP Extrema Bracket (`intraday_pm`) | 3/5 | 3/5 | 4/5 | 4/5 | **14** | Extract high-edge time slots → filtered bracket |
| Noon Iron Butterfly (`intraday_pm`) | 4/5 | 2/5 | 4/5 | 2/5 | **12** | Investigate why paused; review P&L tail |

### Priority order

**Options / income strategies (lowest friction, existing infrastructure):**
1. **Iron condors on ESTX50** — same infrastructure as DRIFT; add as regime-conditional structure. Lowest risk to attempt first.
2. **Noon Iron Butterfly** — investigate why paused; if it is a data-feed issue, Black-Scholes pricing is already implemented. Review P&L tail distribution before resuming.

**Intraday bracket strategies (EU + US session):**
3. **Hougaard 1BP on FTSE** — 58-month dataset exists; mechanical entry and stop; validate on historical data. Fills the EU open window.
4. **Following Range Break** — code is already written; read evaluation output to select optimal stop variant per instrument and timeframe; add live execution wrapper.
5. **VWAP Extrema Bracket** — extract high-edge time slots from saved PNGs; implement time-filtered bracket strategy.
6. **Intraday ORB (US session)** — generic US open complement; requires tick-data backtest with realistic fills.

**Multi-day / swing strategies:**
7. **Factor-driven shorts (negative PEAD)** — ~~mirrors existing long PEAD infrastructure; pilot as put debit spreads on negative earners~~ **VALIDATED.** Backtest confirms -12%/60d on strong filtered signals. Barchart scanners #13 and #14 configured. Paper trade → live.

---

## Next Steps

**Immediate — short framework go-live:**
- [x] Backtest negative PEAD — **validated** (strong short: -12%/60d, 76.8% WR)
- [x] Backtest accruals + F-Score — **validated** as Layer 3 filters
- [x] Backtest consecutive miss — **validated** (1st miss strongest, not consecutive)
- [x] Barchart scanners #13 (Negative PEAD) and #14 (RW Breakdown) configured
- [ ] Paper trade 30 short positions using put debit spreads
- [ ] After 30 paper trades: promote to live at half size

**Deferred (intraday — pursue after short framework is live):**
- [ ] Validate Hougaard 1BP on FTSE — data exists but BT-4-S1 result was no-go
- [ ] Define iron condor entry rules for ESTX50 within DRIFT
- [ ] Generic intraday ORB on ES/NQ
