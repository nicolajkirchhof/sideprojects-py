# Candidate Scoring System

Weighted scoring system for the Trading Assistant app. Replaces the pass/fail 5-box
checklist with a continuous 0–100 scale. Scanner tag overlap adds bonus points. Nothing
is silently dropped — low-scoring candidates are visible but deprioritized.

**Decided:** 2026-04-22 via `/researcher` evaluation
**Implementation:** `finance/apps/assistant/` (backlog in `finance/BACKLOG.md`, Epic TA-E1)

---

## Scoring Formula

```
total = sum(dimension_weighted_scores) + tag_bonus

where:
  dimension_weighted_score =
    0                                  if any hard gate fires
    mean(available_sub_components) × weight   otherwise

  tag_bonus = min(count(tags) × 2, 12)
```

**Range:** 0–100 base + up to 12 tag bonus = theoretical max 112.

---

## Scoring Weights

| Dimension | Weight | Academic Basis |
|-----------|--------|---------------|
| D1 Trend Template | **25** | Century momentum (Geczy & Samonov 2017), cross-sectional momentum (Jegadeesh & Titman 1993). Strongest backing, longest documented persistence. |
| D2 Relative Strength | **25** | Cross-sectional momentum IS relative strength. Equally well-documented. D1+D2 = 50% of signal. |
| D3 Base Quality | **15** | Timing/entry quality. Practitioner evidence (Minervini VCP, BB squeeze) > academic. Reduces whipsaw but doesn't generate alpha independently. |
| D4 Catalyst | **20** | PEAD is Grade A academic (Ball & Brown 1968). Options flow moderate (Pan & Poteshman 2006). Earnings blackout is a hard constraint (→ 0). |
| D5 Risk | **15** | Survival constraint, not alpha. 7% stop is a hard gate. Within range, tighter stops don't predict better outcomes (Kaminski & Lo 2014). |
| **Total** | **100** | |
| Tag bonus | +2/tag, cap +12 | Multi-signal convergence reward. |

**Sub-component weighting within dimensions: equal.** No differential weighting. Precedent:
Piotroski F-Score uses 9 equally-weighted signals; DeMiguel, Garlappi & Uppal (2009) show
equal weights outperform optimized weights out-of-sample due to estimation error. Adding
Short Float to D5 dilutes other sub-components from 5.0 to 3.75 points — this is intentional
(shorts carry additional squeeze risk that should reduce per-component contribution).

---

## Hard Gates

Hard gates are evaluated **before** sub-component averaging. If any gate fires, the
dimension `weighted_score = 0`. Sub-components are still computed for display (the user
should see *why* the gate fired) but do not contribute to the total.

Precedent: Piotroski F-Score and Altman Z-Score treat binary constraints as absolute —
partial credit is not appropriate for untradeable conditions.

| Gate | Condition | Affected dimension |
|------|-----------|--------------------|
| Earnings blackout (long) | Upcoming earnings within 5 days | D4 → 0 |
| Earnings blackout (short) | Upcoming earnings within 10 days | D4 → 0 |
| Stop distance | >7% from entry | D5 → 0 |
| Short Float (shorts only) | >20% of float | D5 → 0 |

A candidate with any hard gate violation can still score up to 80 (other dimensions intact)
but the zeroed dimension makes it clearly flagged. The watchlist shows the violation.

> Long earnings blackout = 5 days, matching TradingPlaybook.md Box 4 ("no binary event
> within 5 days") and the Barchart scanner base filter. Short = 10 days per Layer 1.

---

## Missing Data Policy

When IBKR data is unavailable for a sub-component, **exclude it from the average** and
reweight over available sub-components only. Do not penalize (0.0) or assume neutral (0.5).

Precedent: MSCI ESG Score uses "best available" — missing sub-pillars are excluded and the
score is reweighted. Penalizing missing data creates false negatives; assuming neutral
creates false precision.

- If 5 of 6 D1 sub-components are available: `raw_score = mean(5 available) × 25`
- If ALL sub-components of a dimension are unavailable: `weighted_score = 0`, flagged as
  "no data" (Altman Z-Score approach: can't compute = can't score)
- Each `ComponentScore` carries an `available: bool` flag

**Data source mapping:**

| Sub-component | Primary | Scanner fallback | Fallback available? |
|---------------|---------|-----------------|-------------------|
| D1 Price vs 50d SMA | IBKR daily bars | `pct_from_50d_sma` | Yes |
| D1 50d SMA slope | IBKR slope calc | `slope_50d_sma` | Yes (categorical) |
| D1 200d SMA slope | IBKR slope calc | `slope_200d_sma` | Yes (numeric) |
| D1 52W high distance | IBKR daily bars | `high_52w_distance_pct` | Yes |
| D1 12-month return | IBKR daily bars | `weighted_alpha` (proxy) | Partial |
| D2 RS slope vs SPY | IBKR (stock + SPY) | None | **No** |
| D2 Perf vs Market 5D | — | `perf_vs_market_5d` | Yes (scanner only) |
| D2 Perf vs Market 1M | — | `perf_vs_market_1m` | Yes (scanner only) |
| D2 Perf vs Market 3M | — | `perf_vs_market_3m` | Yes (scanner only) |
| D3 BB squeeze | IBKR BB width calc | `ttm_squeeze` | Yes (categorical) |
| D3 Volume (VDU) | IBKR daily bars | None | **No** |
| D3 SMA stack | IBKR daily bars | partial from scanner | Partial |
| D3 ADR% | — | `adr_pct_20d` | Yes (scanner only) |
| D4 (all) | Scanner fields | — | Yes |
| D5 Stop distance | IBKR (20 SMA) | `atr_pct_20d` (proxy) | Partial |
| D5 ADR vs stop | IBKR + scanner | `adr_pct_20d` | Yes |
| D5 Market cap | — | `market_cap_k` | Yes (scanner only) |
| D5 Short Float | — | `short_float` | Yes (scanner only) |

---

## Direction: Long vs Short

Single scoring function with a `direction` parameter ("long" | "short"). The academic
evidence says long and short momentum are the SAME factor (Jegadeesh & Titman 1993
long-short portfolio). What differs is execution friction and tail risk — captured in D5.

**Direction assignment** (from scanner tags, evaluated during tag assignment step):
- `pead-short` → short
- `consecutive-miss` (without `pead-long`) → short
- RW Breakdown scanner (#14) results → short
- Everything else → long (default)
- Conflict (`pead-long` AND `pead-short`): direction = long (momentum asymmetry favours longs)

**Short scores are NOT scaled down.** The 30% allocation cap is a portfolio-level constraint
applied after scoring. In correction regimes, shorts should rank at the top of the watchlist.

---

## D1 — Trend Template (weight 25)

Captures momentum factor: price trend, SMA alignment, proximity to highs/lows, 12M return.
6 sub-components, equally weighted.

### Long anchor table

| Sub-component | Anchors (linear interpolation between points) | Source |
|---------------|----------------------------------------------|--------|
| Price vs 50d SMA | ≥0% → 1.0; −2% → 0.5; −10% → 0.0 | IBKR / `pct_from_50d_sma` |
| 50d SMA slope | Rising → 1.0; Flat → 0.5; Falling → 0.0 | IBKR / `slope_50d_sma` |
| Price vs 200d SMA | ≥0% → 1.0; −5% → 0.5; −20% → 0.0 | IBKR / `slope_200d_sma` for direction |
| 200d SMA slope | Rising → 1.0; Flat → 0.5; Falling → 0.0 | IBKR / `slope_200d_sma` |
| 52W high distance | 0% → 1.0; 5% → 0.8; 15% → 0.4; 25% → 0.1; ≥30% → 0.0 | IBKR / `high_52w_distance_pct` |
| 12-month return | ≥+20% → 1.0; 0% → 0.5; ≤−20% → 0.0 | IBKR / `weighted_alpha` (proxy) |

### Short anchor table (inverted)

| Sub-component | Anchors |
|---------------|---------|
| Price vs 50d SMA | ≤0% (below) → 1.0; +2% → 0.5; +10% → 0.0 |
| 50d SMA slope | Falling → 1.0; Flat → 0.5; Rising → 0.0 |
| Price vs 200d SMA | ≤0% (below) → 1.0; +5% → 0.5; +20% → 0.0 |
| 200d SMA slope | Falling → 1.0; Flat → 0.5; Rising → 0.0 |
| **52W low distance** | 0% → 1.0; 5% → 0.8; 15% → 0.4; 25% → 0.1; ≥30% → 0.0 |
| 12-month return | ≤−20% → 1.0; 0% → 0.5; ≥+20% → 0.0 |

**Implementation:** `1.0 - raw_score` on SMA sub-components. Swap 52W high → low reference.
Flip 12M return curve.

**Rationale for zero points:** −10% below 50d SMA = deep Stage 4. −20% below 200d SMA =
bear market. These are the Minervini Trend Template boundaries where momentum is absent.

---

## D2 — Relative Strength (weight 25)

Captures cross-sectional momentum: performance vs SPY across timeframes.
4 sub-components, equally weighted.

### Long anchor table

| Sub-component | Anchors | Source |
|---------------|---------|--------|
| RS slope vs SPY (10d) | ≥+0.5%/day → 1.0; 0%/day → 0.3; ≤−0.5%/day → 0.0 | IBKR (`rs_slope_10d`) |
| Perf vs Market 5D | ≥+5% → 1.0; 0% → 0.5; ≤−5% → 0.0 | Scanner `perf_vs_market_5d` |
| Perf vs Market 1M | ≥+5% → 1.0; 0% → 0.5; ≤−5% → 0.0 | Scanner `perf_vs_market_1m` |
| Perf vs Market 3M | ≥+10% → 1.0; 0% → 0.5; ≤−10% → 0.0 | Scanner `perf_vs_market_3m` |

### Short anchor table (clean mirror)

All sub-components: `1.0 - long_raw_score`. Negative RS slope = high score. Underperforming
SPY = high score.

**RS slope threshold rationale:** +0.5%/day over 10 days = +5% relative outperformance,
matching the 5D Perf vs Market threshold for internal consistency. Uses existing
`rs_slope_10d` from `_enrichment.py`.

---

## D3 — Base Quality (weight 15)

Captures entry timing quality: volatility contraction, volume patterns, SMA alignment.
4 sub-components, equally weighted.

**This is the only dimension where long/short are NOT a clean mirror.**

### Long anchor table

| Sub-component | Anchors | Source |
|---------------|---------|--------|
| BB squeeze | On → 1.0; Fired → 0.8; Off → 0.3 | Scanner `ttm_squeeze` / IBKR BB width |
| Volume (VDU) | Contracting (`volume_contracting=True`) → 1.0; RVOL <0.8 → 0.8; RVOL 0.8–1.2 → 0.5; RVOL >1.5 → 0.2 | IBKR `volume_contracting` / `rvol_20d` |
| SMA stack (5>10>20>50) | 4/4 aligned → 1.0; 3/4 → 0.6; 2/4 → 0.3; <2 → 0.0 | IBKR |
| ADR% | 3–7% → 1.0; 2–3% → 0.6; 7–10% → 0.6; <2% → 0.2; >10% → 0.2 | Scanner `adr_pct_20d` |

### Short anchor table (direction-aware)

| Sub-component | Anchors |
|---------------|---------|
| BB squeeze | Off + BB width > 20d avg → 0.8 (distribution expanding); On → 0.5; Off + narrow → 0.3 |
| Volume | RVOL >1.5 (distribution) → 0.8; RVOL 0.8–1.5 → 0.5; RVOL <0.8 (no selling) → 0.3 |
| SMA stack (5<10<20<50) | 4/4 aligned down → 1.0; 3/4 → 0.6; 2/4 → 0.3; <2 → 0.0 |
| ADR% | Same as long |

**Why different:** For longs, low volume = supply exhaustion (VDU). For shorts, high volume
= institutional distribution (Saar 2001; Chordia, Roll & Subrahmanyam 2001).

---

## D4 — Catalyst (weight 20)

Captures event-driven signals: earnings, options flow, volume activity.
5 sub-components, equally weighted.

### Long anchor table

| Sub-component | Anchors | Source |
|---------------|---------|--------|
| Earnings proximity | >20d → 1.0; 10d → 0.8; 5d → 0.5; **<5d → HARD GATE (D4=0)** | Scanner `latest_earnings` |
| Earnings surprise | ≥+10% → 1.0; +5% → 0.7; 0–5% → 0.3; miss → 0.0 | Scanner `earnings_surprise_pct` |
| Surprise history | 4/4 beats → 1.0; 3/4 → 0.8; 2/4 → 0.5; <2 → 0.2 | Scanner `earnings_surprise_q1/q2/q3` |
| Put/Call ratio | <0.3 → 1.0; 0.5 → 0.8; 1.0 → 0.3; 1.5 → 0.2; >1.5 → 0.0 | Scanner `put_call_vol_5d` |
| RVOL | ≥3.0 → 1.0; 2.0 → 0.7; 1.5 → 0.5; 1.0 → 0.3; <0.8 → 0.0 | Scanner `rvol_20d` |

### Short anchor table (direction-aware)

| Sub-component | Anchors |
|---------------|---------|
| Earnings proximity | >20d → 1.0; 10d → 0.5; **<10d → HARD GATE (D4=0)** |
| Earnings surprise | ≤−10% → 1.0; −5% → 0.7; 0 to −5% → 0.3; beat → 0.0 |
| Surprise history | First miss (current < 0, prev ≥ 0) → 1.0; Consecutive miss → 0.7; Beats → 0.0 |
| Put/Call ratio | >2.0 → 1.0; 1.5 → 0.8; 1.0 → 0.3; 0.5 → 0.2; <0.3 → 0.0 |
| RVOL | ≥2.0 → 1.0; 1.5 → 0.7; 1.0 → 0.3; <0.8 → 0.0 |

**Key differences:** Short earnings blackout is 10 days (vs 5 for longs, per Layer 1).
First miss drifts hardest; consecutive misses show weaker drift (backtest: general misses
−2.2% to day 10 then mean-revert; only strong filtered signals persist to 60 days).

---

## D5 — Risk (weight 15)

Survival constraint. Same base for long and short, plus Short Float for shorts.
3 sub-components (long) or 4 (short), equally weighted.

### Long anchor table

| Sub-component | Anchors | Source |
|---------------|---------|--------|
| Stop distance | ≤2% → 1.0; 3% → 0.8; 5% → 0.5; 7% → 0.1; **>7% → HARD GATE (D5=0)** | IBKR 20 SMA / `atr_pct_20d` |
| ADR vs stop | Stop < 0.5× ADR → 1.0; 1× → 0.7; 1.5× → 0.4; ≥2× → 0.1 | IBKR + `adr_pct_20d` |
| Market cap | ≥$10B → 1.0; $2B → 0.8; $500M → 0.5; $200M → 0.3 | Scanner `market_cap_k` |

### Short anchor table (adds Short Float)

| Sub-component | Anchors |
|---------------|---------|
| Stop distance | Same as long |
| ADR vs stop | Same as long |
| Market cap | Same as long |
| **Short Float** | <5% → 1.0; 10% → 0.7; 15% → 0.4; 20% → 0.1; **>20% → HARD GATE (D5=0)** |

**Why Short Float in D5 (not D4):** >20% SI is a RISK constraint (squeeze tail risk), not
a catalyst. Asquith, Pathak & Ritter (2005) show heavily shorted stocks experience violent
reversals.

---

## Tag Bonuses

Tags from consolidated Barchart scanners add +2 points each, capped at +12.

### Long tags (from Long Universe scanner)

| Tag | Condition |
|-----|-----------|
| `52w-high` | 52W %/High < 5% AND TTM Squeeze = On AND RVOL > 1.0 |
| `5d-momentum` | 5D %Chg > 5% AND RVOL > 1.0 AND Perf vs Market 5D > 0% |
| `1m-strength` | 1M %Chg > 10% AND Perf vs Market 1M > 0% AND TTM Squeeze = On |
| `vol-spike` | RVOL > 1.75 |
| `trend-seeker` | Trend Seeker Signal = Buy |
| `ttm-fired` | TTM Squeeze = Fired AND RVOL > 1.0 AND ATRP < 7% |

### Earnings tags (from PEAD scanner)

| Tag | Condition |
|-----|-----------|
| `pead-long` | Earnings Surprise% > 5% AND 5D %Chg > 10% AND Perf vs Market 5D > 0% AND Weighted Alpha > 0 |
| `pead-short` | Earnings Surprise% < −5% AND 5D %Chg < −5% AND % 50D MA < 0% AND Short Float < 20% |
| `consecutive-miss` | Earnings Surprise% < 0 AND ≥2 of Q-1/Q-2/Q-3 also < 0 |

---

## Score Interpretation

| Score Range | Interpretation | Action |
|-------------|---------------|--------|
| 80–100+ | High conviction — multiple dimensions strong + tag convergence | Top of watchlist, auto-analyze with Claude |
| 60–79 | Solid candidate — most dimensions pass, minor weakness | Review in detail panel |
| 40–59 | Mixed — one or two dimensions weak, may be early-stage setup | Lower priority, manual review |
| 20–39 | Weak for current direction — may be strong for opposite direction | Check if short candidate scoring as long (or vice versa) |
| 0–19 | Hard gate violation or structurally unsuitable | Visible but deprioritized |

---

## Barchart Scanner Filter Configuration

The scoring system requires a **broad universe** from Barchart — trend filters are removed
from the scanner and handled by the scorer.

**Retained hard filters** (structural/liquidity):
- Price > $5
- Volume > 1M
- Market Cap > $200M
- No earnings within 5 days
- ADR > 2%

**Removed from scanner** (now scoring inputs):
- % 50D MA > 0%
- Slope of 50D SMA: Rising
- Slope of 200D SMA > 0
- Weighted Alpha > 0

Full scanner configuration: `BarchartScreeners.md`
