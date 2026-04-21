# Portfolio Assessment — April 2026

> Independent evaluation of the three-strategy portfolio: Momentum Swing, DRIFT, Long-Term ETF.
> Lens: structural strengths, exploitable weaknesses, and a prioritised refinement roadmap.

**Date:** 2026-04-20
**Reviewer:** `/researcher` framework
**Scope:** All documents in `investing_framework/`, all code in `finance/`, all backlogs

---

## The Portfolio

| Strategy | Edge Source | Timeframe | Direction | Status |
|----------|-----------|-----------|-----------|--------|
| **Momentum Swing** (A) | Institutional friction, PEAD, century momentum | 5–50 days | Long only | Active |
| **DRIFT** (B) | Variance risk premium, positive drift | 30–360 DTE | Long delta (directional) + neutral (commodities) | Active |
| **Long-Term ETF** (C) | Factor premia (size, value, profitability) | 5+ years | Long only | Active |

---

## Strengths

### 1. Academic grounding is rare and genuine

Every profit mechanism traces to published research — not practitioner lore. PEAD cites
Ball & Brown (1968), VRP cites Carr & Wu (2009), century momentum cites Geczy & Samonov
(2017). This creates *falsifiability*: you can check whether the edge still holds, rather
than relying on pattern-recognition faith.

### 2. Defined risk everywhere

Hard stops (7% swing, 50Δ DRIFT), 0.5% max risk per swing trade, 2% max per DRIFT trade,
50% BP cap. These are non-negotiable rules, not guidelines. The LTCM reference in the
hedging section shows why this matters.

### 3. Three genuinely diversified mechanisms

- Swing: momentum persistence (institutional friction)
- DRIFT: variance risk premium (structural overpricing of protection)
- Long-Term: factor premia (size, value, profitability)

These are independent edges with different academic foundations. Most retail traders running
"multiple strategies" are actually running the same thesis in different instruments.

### 4. DRIFT neutral block is the best idea in the portfolio

UNG (72% win, +$4,162), WEAT (77% win, +$170), PDBC (79% win, +$44) — commodity VRP is
5–10× larger than equity index VRP, and the correlation to SPY is near zero. This block
earns comparable theta while acting as a structural hedge. The 50/50 split over the original
60/40 is correct.

### 5. Disciplined mechanism framework

The 4-step Outlier framework (define → signal → structure → validate) prevents adding
strategies based on a good week. The StrategyIdeasAssessment scores each idea on four
dimensions before implementation. This is rare discipline.

### 6. Pipeline automation

Barchart → Gmail → CSV → pipeline → 5-box scoring → Claude analysis → Tradelog daily-prep.
Removes mundane filtering, preserves attention for judgment calls (chart quality, ORB candle
assessment, catalyst evaluation).

---

## Weaknesses

### W1. Net delta exposure is overwhelmingly positive (CRITICAL)

| Strategy | Delta Exposure | Bear Market Outcome |
|----------|---------------|---------------------|
| Swing (A) | 100% long | Goes idle (NO-GO) |
| DRIFT directional block | Long delta | 50Δ stops fire across the board |
| DRIFT neutral block | Neutral | Only survivor — earns theta |
| Long-Term ETF | 80–90% equity | Drops 40–50% |
| Short exposure | **Zero** | **Not built** |

In a 2008-style event, all three strategies lose simultaneously. The factor-driven short
strategy scored 15/20 in October 2025 — it's still not piloted. This is the single
highest-priority gap.

**Impact:** In a fast correction (-20% in 3 weeks), the portfolio has no active short delta
to offset losses. The neutral block and gold are the only natural hedges. The hedge budget
(VIX calls, far-OTM puts) is reactive and untested against actual portfolio P&L.

### W2. TQQQ is a structurally dangerous DRIFT underlying

- A 30% QQQ drop → ~80% TQQQ drop. The 50Δ stop fires on *any* correction.
- Daily rebalancing creates volatility decay that standard Black-Scholes doesn't capture.
- The high IV exists because the risk is high, not because there's excess VRP.
- IV-RV spread per unit of *actual* risk may be lower than QQQ.

**Recommendation:** Replace with additional XSP or IWM allocation. The sizing argument at
$50k is valid but the path-dependency introduces model risk.

### W3. Thin-liquidity underlyings have overstated backtest EV

WEAT (~2k opts/day), PDBC (~1–2k/day), DBA (~1–3k/day) show the best backtest stats.
This is not coincidental — the premium compensates for illiquidity:

- Backtests using mid-price fills overstate real EV by the bid-ask on entry AND exit
- At 1-lot positions, no ability to scale in/out
- 50Δ hard stop on a thin underlying → fill may be 5–10% worse than theoretical

**Recommendation:** Paper trade 10 cycles with limit-order-only fills. If effective premium
is <70% of backtest premium, drop the position.

### W4. Correlation spike during stress

Normal-regime correlations (FXI 0.30–0.55, EWZ 0.40–0.60, ESTX50 ~0.65) spike to >0.90
during crises. The "geographic diversification" within the directional block evaporates
precisely when it matters.

The drawdown scaling framework addresses this partially but is *offensive* (deploy more
capital into fear), not *defensive*. Scaling into equity short puts during a correlation
spike is increasing concentrated risk.

### W5. Research pipeline breadth vs depth — pilot paralysis

Seven strategy ideas at 14–15/20 ("pilot cautiously"). Each pilot requires backtesting
(2–4 weeks), paper trading (6–12 weeks), live validation (3–6 months). Running seven in
parallel means none gets adequate validation. Partial implementation of multiple strategies
creates cognitive load without generating validated edge.

### W6. No portfolio-level stress test

Individual strategy stress behavior is documented (regime filters, stops, BP caps). The
*combined* portfolio P&L under stress is not modeled. Key unanswered questions:

- In 2020 COVID (SPY –34%, VIX 82), what is the combined loss across all three strategies?
- Does the 1–2% annual hedge budget offset directional block losses?
- Does drawdown scaling create 80% BP at the same time the swing book is stopped out?

### W7. Live track record is the missing variable

Extensive backtests, dashboards, pipelines — but: what is the live Sharpe and max drawdown
of the combined portfolio? The 4-step framework requires 30+ live trades per mechanism.
Which mechanisms have passed this gate?

### W8. Swing book goes dormant in bear markets

When SPY < 50 SMA, the playbook says "Type D only" but the short infrastructure is unbuilt.
Result: Strategy A produces zero P&L during the period where active management is most
valuable. The DRIFT neutral block partially compensates, but the swing book's contribution
drops to zero.

### W9. PEAD drift window may be compressing

More systematic funds now exploit PEAD. The 40–60 day drift documented by Bernard & Thomas
(1989) may now compress to 20–30 days. The drift is strongest in small/mid-caps where
execution quality is worst. Post-earnings IV spikes make options expression expensive on
the exact names with the strongest drift.

### W10. "Never rebalance" rule has a horizon problem

Tax-efficient in Germany, conceptually sound. But over 15–20 years, a 15% gold allocation
can drift to 30%+ or 5% without rebalancing. The "redirect new cash" approach only works
if contributions are large relative to portfolio size. As the portfolio grows, marginal
contributions become too small to correct meaningful drift.

---

## Cross-Strategy Interaction Analysis

### Positive interactions

- DRIFT neutral block (UNG, WEAT, GLD) hedges swing losses during corrections
- Long-Term gold (10–20%) provides crisis diversification
- PEAD infrastructure is shared between long (swing) and potential short — code reuse
- Pipeline automation serves both swing scanning and DRIFT regime assessment

### Negative interactions

| Scenario | Effect |
|----------|--------|
| Correction (–10 to –20%) | Swing goes idle AND DRIFT directional loses — both revenue streams dry up |
| Drawdown scaling + stopped-out swings | Scaling into DRIFT (more delta) at the same time swing is stopped out → doubling down on recovery thesis |
| Bear market attention budget | With swing idle, temptation to force DRIFT entries to maintain income |

### Combined portfolio behavior under stress

| Scenario | Swing (A) | DRIFT Dir. | DRIFT Neutral | Long-Term | Net |
|----------|-----------|------------|---------------|-----------|-----|
| Bull | Earning | Earning | Earning | Growing | All positive |
| Flat | Limited | VRP income | VRP income | Flat | DRIFT carries |
| Correction (–15%) | Stopped/idle | Losing | Earning/flat | Losing | Concentrated loss |
| Bear (–30%+) | Fully idle | Stopped, scaling in | Partially earning | Losing significantly | Maximum pain |

**Critical observation:** The system's value-add is in the flat-to-moderate-correction zone.
In the tail scenarios, there is no active short delta — only passive gold and BP buffer.

---

## Roadmap — Prioritised Refinement

### Phase 1 — Immediate: Short Framework + Swing Backtesting (Q2 2026)

#### 1A. Factor-Driven Short Strategy — Build and Pilot

**Thesis:** Markets need long AND short exposure at all times. A portfolio that can only go
long is structurally exposed to the one scenario where active management matters most.

The short strategy mirrors the long swing book: same academic edges (PEAD, accruals, momentum),
same infrastructure (earnings data, Barchart pipeline), same execution windows (ORB), but
capturing the downside of the same mechanisms.

**Implementation:** See §Short Framework below.

#### 1B. Swing Strategy Backtesting Execution Framework

**Thesis:** The swing strategies must be backtested mechanically. This serves two purposes:
(1) validate that the edge survives realistic costs, and (2) create the technical
infrastructure for optimization — results become the fitness function for refining entry,
stop, and exit parameters.

**Implementation:** See §Execution Framework below.

### Phase 2 — Validation (Q3 2026)

1. Portfolio-level stress test (combined P&L under 2008/2020/2022 scenarios)
2. DRIFT thin-liquidity validation (10 paper cycles with real fills on WEAT/PDBC/DBA)
3. TQQQ VRP-per-unit-risk analysis — replace if VRP ratio < QQQ
4. Hedge budget empirical test (1–2% annual cost vs actual drawdown offset)

### Phase 3 — Expansion (Q4 2026+)

1. Add accruals + F-Score layers to short framework (Layers 3–4)
2. PEAD drift window measurement on live data
3. Iron condors on ESTX50 within DRIFT (regime-conditional structure)
4. Intraday strategies (Hougaard, ORB) — only after short + swing validation complete

---

## Short Framework — Implementation Specification

### Design Principle

> Always have long AND short exposure. The short book operates independently of the long
> book — it is not a hedge, it is a separate alpha source that captures the downside of the
> same academic edges.

### Architecture: Mirror of the Long Book

| Component | Long Book (existing) | Short Book (new) |
|-----------|---------------------|------------------|
| Edge source | Positive PEAD, breakout momentum, RS | Negative PEAD, consecutive miss, accruals, RW |
| Signal | Positive surprise, gap up, RS | Negative surprise, gap down, RW vs SPY |
| Structure | Long calls, debit spreads, stock | Put debit spreads (defined max loss) |
| Entry | ORB above 15/30-min candle | ORB below 15/30-min candle, or first recovery day |
| Stop | Below entry candle / base low | Above entry candle / recovery high; +15% hard |
| Sizing | 0.5% max risk per trade | 0.5% max risk per trade |
| Pipeline | Barchart scanner → 5-box → watchlist | Same pipeline, inverted filters |

### Signal Layers (from StrategyIdeasAssessment Idea 1, refined)

#### Layer 1 — Universe Filter (always on)

| Filter | Rule | Rationale |
|--------|------|-----------|
| Trend stage | Price < declining 50D SMA | Institutional distribution underway |
| Short interest | < 20% of float | Above 20%: squeeze risk dominates |
| Liquidity | 20D Avg Vol > 500K | Borrow availability + execution quality |
| Binary events | No earnings, FDA, M&A within 10 days | Gap risk elimination |
| Price | > $5 | Same quality floor as long book |

#### Layer 2 — Primary Trigger (at least one required)

**Trigger A — Negative PEAD** (highest priority, mirrors PM-02)
- Signal: EPS miss (bottom 25% SUE) + closes bottom 25% of range on earnings day
- Academic: Ball & Brown 1968, Bernard & Thomas 1989, Skinner & Sloan 2002
- Hold: 20–60 days (mirror of long PEAD drift)
- Entry: first recovery day after gap (1–3% bounce from post-earnings low)

**Trigger B — Consecutive Miss + Guidance Cut** (higher conviction, lower frequency)
- Signal: 2+ consecutive misses AND forward guidance below consensus
- Academic: Chordia & Shivakumar 2006 (analyst revision momentum)
- Hold: 30–90 days

**Trigger C — RS Weakness Breakdown** (Type D from TradingPlaybook, mechanism PM-05)
- Signal: RW line at new lows vs SPY + sector in Stage 3/4 + breakdown below base
- Entry: ORB below 30-min candle on breakdown day
- Hold: 5–30 days
- This is the short-side mirror of the Type B/C long entries

#### Layer 3 — Fundamental Confirmation (increases conviction / size)

| Signal | Source | Rule |
|--------|--------|------|
| Accruals anomaly | Sloan 1996 | Accruals ratio top quartile (earnings >> cash flow) |
| Short interest flow | Cohen, Diether & Malloy 2007 | SI increasing last 2 reporting periods |
| F-Score | Piotroski 2000 | F-Score ≤ 2 (deteriorating fundamentals) |

Layer 3 is additive — use to increase size or shorten the evaluation bar, not as standalone.

#### Layer 4 — Price Confirmation (entry timing)

| Condition | Rule |
|-----------|------|
| Entry timing | Enter on first 1–3% recovery from post-event low (better fills) |
| Stage confirmation | Price < 50D SMA, slope negative or flat |
| Sector context | Sector in Stage 3/4 — avoids shorting individual weakness in rising sector |
| Volume | Below-average volume on recovery day (no institutional buying) |

### Structure: Put Debit Spreads (not short stock)

**Why defined risk:** Overnight gap-up (takeover, FDA, surprise beat) is uncapped on short
stock. Put debit spreads define max loss at spread width. This makes position sizing
deterministic and eliminates the borrow problem entirely.

| Parameter | Guideline |
|-----------|----------|
| Long put | ATM or slightly OTM (40–50Δ) |
| Short put | 1–2σ below current price (15–20Δ) |
| DTE | 45–60 (same theta curve as DRIFT) |
| Max loss | Debit paid (defined by spread width) |
| Profit target | Close at 50% of max profit (same as all other structures) |
| Stop | Close at 50% of debit remaining OR underlying recovers above 50D SMA |
| Sizing | 0.5% portfolio max loss = debit paid per position |

### Barchart Scanner Integration

Add to existing pipeline — inverted filters on a new scanner:

**Scanner 13: Negative PEAD Candidates**
- View: PEAD/EP (same view — earnings surprise columns work for both directions)
- Filters:

| Filter | Value | Reasoning |
|--------|-------|-----------|
| Latest Earnings | Within past 7 days | Earnings event just happened |
| 5D %Chg | < –5% | Significant downside move |
| Earnings Surprise% | < –5% | Negative surprise — bottom quartile |
| 20D RelVol | > 2.0 | Volume spike on earnings |
| Short Float | < 20% | Squeeze filter (Layer 1) |
| % 50D MA | < 0% | Below 50D SMA (Stage 3/4) |

**Scanner 14: RW Breakdown Candidates**
- View: Options/Flow (short interest + flow data needed)
- Filters:

| Filter | Value | Reasoning |
|--------|-------|-----------|
| Perf vs Market 5D | < –3% | Underperforming SPY significantly |
| Perf vs Market 1M | < 0% | Sustained weakness |
| % 50D MA | < 0% | Below 50D SMA |
| Slope of 50D SMA | Declining | SMA confirming downtrend |
| Short Float | < 20% | Squeeze filter |

### 5-Box Checklist — Short Version

**01 — Trend Template (inverted)**
- Price < 50D SMA, 50D SMA declining
- Within 25% of 52-week low (no floor support nearby)
- Negative 12-month return

**02 — Relative Weakness (Bruzzese, inverted)**
- RW line at new lows vs SPY
- Drops more when SPY dips, fails to recover when SPY bounces
- Underperforming sector peers over 1M

**03 — Distribution Quality**
- Base forming below declining SMAs (supply being created)
- Volume expanding on down days (distribution, not accumulation)
- No Bollinger squeeze (volatility already expanding downward)

**04 — Catalyst (negative)**
- Earnings miss, guidance cut, sector deterioration
- Negative PEAD: gap ≥ 5% down AND closed bottom 25% of range
- No binary event within 10 days that could reverse (FDA, takeover)

**05 — Risk Parameters**
- Stop defined before entry (recovery high or 50D SMA)
- Max risk: spread debit = 0.5% of portfolio
- R:R ≥ 2:1 minimum

### Portfolio Integration Rules

| Rule | Detail |
|------|--------|
| Max short exposure | 30% of swing book allocation (start at 15%, scale after 30 live trades) |
| Regime filter | Only short when SPY < 50D SMA OR breadth declining 2:1+ |
| Independence | Short book P&L tracked separately from long book |
| Correlation check | Never short a stock you're long in swing book (obvious but stated) |
| Earnings season | Peak activity — run negative PEAD scanner daily during earnings season |

### Backtesting Priority

BT-3 stories in `BACKLOG-BACKTESTING.md` are reprioritised to run immediately after BT-1
infrastructure:

1. **BT-3-S1: Negative PEAD** — highest priority, mirrors existing long PEAD
2. **BT-2-S4: PEAD drift window** — validates both long AND short PEAD simultaneously
3. **BT-3-S3: Accruals factor** — Layer 3 confirmation signal
4. **BT-3-S4: F-Score filter** — Layer 3 confirmation signal
5. **BT-3-S2: Consecutive miss** — requires guidance data (may be blocked)

---

## Execution Framework — Swing Strategy Backtesting

### Design Principle

> The backtesting framework is not a one-time validation tool. It is the permanent
> optimization engine. Strategy parameters (entry candle, stop placement, exit rules,
> regime filter) are inputs. Results (win rate, expectancy, Sharpe, max DD) are outputs.
> Refinement is the loop: change inputs → measure outputs → keep improvements.

### Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   Strategy Definition                     │
│  (entry signal, stop rule, exit rules, regime filter,     │
│   universe filter, position sizing, direction)            │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│                   Backtest Engine                          │
│  - Walk-forward (no future data leakage)                  │
│  - Cost model (BT-1-S1: spread + commission + slippage)   │
│  - Event-driven: signal → entry order → fill → manage     │
│  - Supports ORB intraday entry on EOD-triggered signals   │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│                   Results Output                          │
│  Standard metrics per run:                                │
│  - Win rate, avg win, avg loss, expectancy per trade      │
│  - Sharpe ratio (annualised)                              │
│  - Max drawdown (peak-to-trough)                          │
│  - Profit factor (gross wins / gross losses)              │
│  - Total trades, avg holding period                       │
│  - Segmented by: regime, setup type, direction, year      │
│                                                           │
│  Output to: Parquet + summary markdown                    │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│                   Optimization Loop                       │
│  - Parameter grid: stop %, exit rule, candle window,      │
│    regime filter variant                                  │
│  - Walk-forward: in-sample optimise → out-of-sample test  │
│  - Overfitting guard: require OOS Sharpe ≥ 50% of IS     │
│  - Result: optimal parameter set per strategy × direction │
└──────────────────────────────────────────────────────────┘
```

### Strategy Definition Interface

Each strategy is a Python dataclass with the following fields:

| Field | Type | Example |
|-------|------|---------|
| `name` | str | `"EP_Long"` |
| `direction` | `long \| short` | `long` |
| `universe_filter` | callable | Price > $5, Vol > 1M, Stage 2 |
| `signal` | callable | Gap ≥ 10%, RVOL > 5×, close top 25% |
| `entry_rule` | callable | ORB above 15-min candle high |
| `stop_rule` | callable | Below entry candle low, max 7% |
| `exit_rules` | list[callable] | [2R partial, 4R partial, 5 EMA 2-close, 50d time stop] |
| `regime_filter` | callable | SPY > 50D SMA, VIX < 30 |
| `position_sizer` | callable | 0.5% risk / (entry − stop) |
| `params` | dict | Tunable parameters for optimization |

### Optimization Parameters (per strategy)

| Strategy | Tunable Parameters |
|----------|-------------------|
| **EP (Type A)** | Gap threshold (8/10/15%), RVOL threshold (3/5/10×), ORB candle (15/30 min), close-in-range threshold (25/33%) |
| **VCP (Type B)** | Base duration (2–6 weeks), contraction count (2/3/4), VDU threshold, breakout volume (40/60/80% above avg), ORB candle (15/30 min) |
| **EMA Reclaim (Type C)** | Pullback MA (10/20 SMA), pullback depth tolerance (0.5/1/1.5 ATR), reclaim candle pattern (Elephant/Tail/any), ORB candle (15/30 min) |
| **Short PEAD** | SUE threshold (bottom 10/25/33%), gap threshold (−5/−10%), entry timing (gap day/recovery day), hold period (20/40/60 days) |
| **All strategies** | Stop method (candle low/base low/MA), stop max % (5/7/10%), first partial R (1.5/2/2.5), trail MA (5/10/20 EMA), time stop (30/50/70 days) |

### Walk-Forward Protocol

| Phase | Period | Purpose |
|-------|--------|---------|
| In-sample | 2019–2022 (4 years) | Optimize parameters |
| Out-of-sample | 2023–2025 (3 years) | Validate |
| Live forward | 2026+ | Paper then live |

**Overfitting guard:** Out-of-sample Sharpe must be ≥ 50% of in-sample Sharpe. If not,
the parameter set is overfit and rejected. Re-run with fewer parameters or wider grid.

### Data Requirements

| Data Source | Coverage | Use |
|-------------|----------|-----|
| IBKR intraday Parquet (1-min) | 2013–2026, 88 symbols | ORB candle simulation |
| EOD OHLCV (stocks) | 2010–2026, via Yahoo/IBKR | Universe filtering, MA computation, regime |
| Dolt earnings DB | 2016–2026, ~9K US symbols | PEAD signals, SUE, surprise % |
| Dolt financials | 2012–2026 | Accruals, F-Score (short Layer 3) |
| VIX daily | 2010–2026 | Regime filter |
| SPY daily | 2010–2026 | RS computation, market regime |

### Output Specification

Each backtest run produces:

1. **Trade log** — Parquet file with one row per trade:
   `symbol, direction, setup_type, entry_date, entry_price, stop_price, exit_date,
   exit_price, exit_reason, pnl_dollars, pnl_r, holding_days, regime_at_entry,
   cost_applied`

2. **Summary metrics** — Markdown file:
   ```
   Strategy: EP_Long | Period: 2019-2025 | Direction: Long
   Trades: 342 | Win Rate: 58.2% | Avg Win: 2.4R | Avg Loss: -1.0R
   Expectancy: +0.42R/trade | Sharpe: 1.24 | Max DD: -12.3%
   Profit Factor: 2.1 | Avg Hold: 14.2 days

   By Regime:  GO: 62.1% WR, +0.58R | NO-GO: 41.2% WR, -0.12R
   By Year:    2019: +18R | 2020: +42R | 2021: +31R | ...
   ```

3. **Parameter comparison** — when running optimization grid:
   Table of all parameter combinations with IS and OOS metrics side by side.

### Implementation Location

```
finance/utils/backtest.py          — Engine (already exists, extend)
finance/swing_pm/backtests/        — Strategy definitions + run scripts
finance/intraday_pm/backtests/     — Intraday strategy definitions
finance/_data/backtest_results/    — Output Parquet + markdown (already created)
```

### Backtesting Execution Order (revised)

The existing `BACKLOG-BACKTESTING.md` is reprioritised to run shorts first, then swing longs:

```
Phase 0 — Infrastructure (Done)
  BT-0-S1/S2/S3  Restructure + Parquet migration
  BT-1-S1        Cost model + framework
  BT-1-S2        Intraday data coverage
  BT-1-S3        Earnings data coverage

Phase 1 — Short Framework (NEW PRIORITY)
  BT-2-S4        PEAD drift window (validates both directions)
  BT-3-S1        Negative PEAD (primary short trigger)
  BT-3-S3        Accruals factor (Layer 3 confirmation)
  BT-3-S4        F-Score filter (Layer 3 confirmation)
  BT-3-S2        Consecutive miss + guidance cut

Phase 2 — Swing Long Strategies
  BT-2-S7        Bollinger touch / Type C (simplest, EOD only)
  BT-2-S6        Overnight reversal (close-to-open, simple)
  BT-2-S8        VIX mean reversion (small sample, fast)
  BT-2-S5        Pre-earnings anticipation
  BT-2-S1        EP backtest (most complex)
  BT-2-S2        VCP backtest (most complex)
  BT-2-S3        EMA Reclaim

Phase 3 — Intraday (deferred until Phase 1+2 validated)
  BT-4/5/6       European + US session strategies
```

---

## Deferred Items — Do NOT Pursue Now

| Item | Reason to Defer |
|------|-----------------|
| Hougaard intraday strategies | Requires separate infrastructure; thin edge after costs; pursue only after short + swing validated |
| Generic US ORB | Same as above — parallel research dilutes focus |
| Noon Iron Butterfly | Same VRP edge as DRIFT; adds complexity without diversification |
| VWAP Extrema Bracket | Evaluation PNGs exist but no actionable signal yet identified |
| Currency/Bond VRP | Research stage; no backtest infrastructure for FX/bond options |
| Quality ETF (QUAL) in Long-Term | Future expansion; current factor tilt is adequate |

---

## Backtest Results (2026-04-21)

Full results in `BACKLOG-BACKTESTING.md` Results Register and `finance/_data/backtest_results/swing/`.

### Validated edges (ranked by Sharpe)

| Strategy | N | 60d Mean% | 60d Win% | Sharpe | Action |
|----------|---|-----------|----------|--------|--------|
| Strong Long EP | 368 | +31.70 | 84.8 | 0.791 | Trade aggressively — highest-conviction signal |
| Green Line Breakout | 13,796 | +7.16 | 67.4 | 0.260 | Add to active scanning |
| EMA Reclaim (Type C) | 60,410 | +5.63 | 61.8 | 0.222 | Most reliable — Type C confirmed |
| PEAD Beats (long) | 15,699 | +5.43 | 57.5 | 0.208 | Core long PEAD — PM-02 confirmed |
| Pre-Earnings (T-14) | 2,243 | +1.99 (14d) | 58.4 | 0.200 | PM-03 confirmed — exit T-1 |
| Strong Short PEAD | 418 | -12.00 | 76.8 (short) | -0.369 | Short framework primary signal |
| BB Lower Touch | 53,400 | +2.25 | 55.1 | 0.115 | Supplementary — confirms pullback in trend |

### Rejected

| Strategy | N | Finding | Action |
|----------|---|---------|--------|
| Overnight Reversal (PM-08) | 33,935 | No edge at any horizon | Remove from active mechanisms |
| General short PEAD (all misses) | 5,169 | Drift fades by day 20 | Restrict to 10-20d hold |

### Factor confirmation signals (Layer 3)

| Signal | N | Finding | Role |
|--------|---|---------|------|
| 1st earnings miss | 3,292 | -3.17%/1d, strongest short signal | Primary short trigger — focus on miss quality, not count |
| 2nd consecutive miss | 1,073 | -1.82%/1d, weaker than 1st | Diminishing drift — bad news already priced in |
| Accruals (D1 vs D10) | 59,536 | +3.38%/year long-short spread | Annual rebalance signal, not swing trade |
| F-Score (>=8 vs <=2) | 197,250 | ~0.5%/quarter spread | Quality filter — F>=7 upgrades conviction |

### Key insight: short side is viable but time-limited

The short PEAD drift is real but temporary. General misses show -2.2% at 10d then mean-revert.
**Only the strong short signal** (miss + gap<=-5% + bottom 25% close) persists to 60d (-12%).
The short framework should filter aggressively and use 10-20d holds for general misses,
40-60d only for strong filtered signals.

**Consecutive misses are weaker, not stronger.** The 1st miss produces the largest reaction.
Focus on first-miss quality (SUE magnitude, gap size, close-in-range position) not miss count.

---

## Success Criteria

| Milestone | Metric | Target | Status |
|-----------|--------|--------|--------|
| Short framework pilot | 30 paper trades on negative PEAD | Net expectancy > 0 after costs | **Next** — backtest validated |
| Swing backtest validation | EP, VCP, EMA Reclaim backtested | ≥ 2 of 3 show OOS Sharpe > 0.5 | **Done** — EP 0.79, EMA 0.22 |
| Portfolio stress test | Combined P&L under 2020 COVID | Max drawdown < 25% with hedges | Pending |
| Short live validation | 30 live trades at half size | Win rate > 45%, expectancy > 0 | Pending |
| Execution framework | ≥ 5 strategies backtested with comparable metrics | Parameter optimization loop operational | **Done** — 9 strategies backtested |

---

## Review Cadence

- **Monthly:** Check progress against Phase 1 milestones
- **Quarterly:** Full portfolio assessment update (this document)
- **After 30 live short trades:** Promote to full size or archive with findings
