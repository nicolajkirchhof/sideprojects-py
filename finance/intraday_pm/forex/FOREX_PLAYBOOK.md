# Forex Playbook

Generated: 2026-04-21
Research basis: Researcher evaluation + academic literature review

---

## Overview

Evaluation of whether the existing intraday and swing strategies transfer to forex,
and what additional forex-specific strategies have academic backing.

Key structural difference from equity index futures:
**Forex is a continuous OTC market — there is no exchange auction.** All session-open
edges in the equity book (Hougaard, VWAP Extrema) depend on opening-auction mechanics
that are absent in forex. Strategies are evaluated against this constraint.

---

## Go Strategies

None. Both backtested strategies failed to reach their Go thresholds.

### 1. OCO Session-Open Bracket (BT-FX-1)

**Verdict: Pilot — backtest required before deploying capital**

**Mechanism:** Institutional order flow concentrates at the London open (07:00 UTC)
and NY open (13:30 UTC). A directional bracket on the first bar of the session
captures any sustained move from that order flow.

**Academic backing (partial):**
- Andersen & Bollerslev (1998): intraday volatility spikes at London and NY open — documented
- Evans & Lyons (2002): order flow is directional at session opens — documented
- Directional profitability of the breakout: practitioner only — not independently replicated

**Pairs:** EURUSD, GBPUSD (London + NY), USDJPY (Tokyo + London), AUDUSD (Sydney + Tokyo)

**Parameters:**
| Parameter | Value |
|-----------|-------|
| Sessions | London 07:00, NY 13:30, Tokyo 00:00, Sydney 22:00 (UTC) |
| Timeframes | 5min, 15min, 30min |
| Stop methods | ATR trail (SRS style), bar range (ASRS style) |
| Entry offset | 0.5 pips beyond signal bar high/low |
| Cost model | 1.5 pips round-trip |
| Scan window | First 2 hours of each session |

**Instruments:** IBKR CASH (CFD/spot); micro futures M6E, M6B, M6J if exchange access confirmed

**Threshold for Go:** Sharpe > 0.06 on at least one pair × session combination.
Below that: do not deploy.

**Results (2020-01-01 -> 2026-04-01, 576 combinations per pair):**

| Pair | Session | Positive EV combos | Best Sharpe | Best EV (pips) | Verdict |
|------|---------|-------------------|-------------|----------------|---------|
| EURUSD | London | 0 / 72 | -0.019 | -0.36 | No-go |
| EURUSD | NY | 2 / 72 | +0.022 | +0.44 | No-go |
| GBPUSD | London | 0 / 72 | -0.037 | -0.45 | No-go |
| GBPUSD | NY | 5 / 72 | +0.005 | +0.11 | No-go |
| USDJPY | Tokyo | 10 / 72 | +0.033 | +0.52 | No-go |
| USDJPY | London | 3 / 72 | +0.019 | +0.27 | No-go |
| AUDUSD | Sydney | 0 / 72 | -0.019 | -0.20 | No-go |
| AUDUSD | Tokyo | 0 / 72 | -0.019 | -0.20 | No-go |

**Verdict: No-go — all pairs, all sessions.** Best Sharpe across 576 combinations is +0.033
(USDJPY Tokyo, 15min/bar2 bar_range). The Go threshold of +0.06 was not met by any
combination. The researcher's prediction was correct: the absence of an opening auction
means win rates stay near 36–44% vs 43–46% in equity index futures. At 1.5 pips cost,
this is insufficient to generate positive EV systematically.

See `RESULTS.md` -> Forex OCO Candle Scan for full per-combination tables.

---

### 2. Currency Momentum (BT-FX-2)

**Verdict: Pilot cautiously — academic evidence is strong, portfolio fit is good**

**Mechanism:** 12-1 month cross-sectional momentum in currency pairs vs USD.
Long the top 3 outperforming currencies, short the bottom 3. Weekly rebalancing.
Menkhoff et al (2012) document robust Sharpe of 0.5–0.7 gross in major pairs.

**Academic backing (strong):**
- Menkhoff, Sarno, Schmeling & Schrimpf (2012) — JFE 106 — independently replicated
- Moskowitz, Ooi & Pedersen (2012) — time-series momentum in FX also confirmed
- Survives realistic transaction costs in major pairs (1–2 pips/side)

**Universe:** EURUSD, GBPUSD, AUDUSD, CHFUSD (direct); USDJPY, USDCAD (sign-inverted)

**Parameters:**
| Parameter | Value |
|-----------|-------|
| Lookback | 52 weeks |
| Skip (reversal buffer) | 4 weeks |
| Portfolio | Long top 3, short bottom 3 |
| Rebalance | Weekly (Friday close) |
| Cost | 1.5 pips per pair when position changes |

**Execution instrument:** CME micro FX futures (M6E, M6B, M6J, M6A) — preferred
over spot CFD (no overnight financing, exchange-traded margin, clean fills).

**Threshold for Go:** Net Sharpe > 0.30 in backtest.

**Results (2014-05-30 -> 2026-03-27, 618 weeks):**

| Metric | Gross | Net |
|--------|-------|-----|
| Annualised return | +0.9% | +0.5% |
| Sharpe | +0.154 | +0.091 |
| Max drawdown | -15.2% | -16.9% |
| Win rate (weekly) | 52.1% | 51.1% |

**Verdict: No-go.** Net Sharpe +0.091 is well below the +0.30 pilot threshold.
The academic result (Sharpe ~0.5–0.7) does not replicate with this 6-pair universe.
CHFUSD was heavily long (76% of weeks), suggesting the CHF safe-haven flight-to-quality
in 2014-2016 and 2020 drove most of the gross return, not clean momentum.
See `RESULTS.md` -> Currency Momentum (BT-FX-2) for full breakdown.

---

## No-Go Strategies

| Strategy | Score | Verdict | Primary reason |
|----------|-------|---------|----------------|
| OCO session-open bracket | 12/20 | **No-go (backtested)** | Best Sharpe +0.033 across 576 combos; threshold +0.06 not met; no auction mechanic |
| Currency Momentum | 14/20 | **No-go (backtested)** | Net Sharpe +0.091; threshold +0.30 not met; CHF dominance, not clean momentum |
| VWAP Extrema | 7/20 | Do not pursue | VWAP is institutional in equities; broker-calculated in forex — mechanism absent |
| Hougaard SRS/ASRS | 5/20 | Do not pursue | Opening auction mechanics required; no auction in forex |
| Carry Trade | 11/20 | Do not pursue | Strong academic edge but crash risk perfectly correlated with DRIFT short-vol |

### Why VWAP Extrema does not transfer

The equity-index VWAP Extrema edge (Sharpe +0.43–+0.53 across 11 instruments)
derives from institutions placing and defending orders at exchange-calculated VWAP.
In forex, VWAP is broker-calculated from their internal tick stream — banks do not
manage orders relative to any "VWAP" benchmark. The causal chain is broken.
Testing it in forex would be pure pattern-matching with no mechanistic basis.

### Why Hougaard patterns do not transfer

The SRS/ASRS patterns identify exhaustion/continuation of the opening auction's
first directional impulse. The DAX opens with an auction at 09:00 Frankfurt; the
first 5–15 min bar captures that auction momentum. Forex has no auction — the
London open at 07:00 UTC is simply when European bank desks start routing orders.
The pattern has no equivalent trigger in forex.

### Why Carry Trade is excluded despite strong evidence

The carry trade is a real, academically documented alpha source (Burnside 2011,
Lustig & Verdelhan 2007). It is excluded not for lack of edge but for portfolio fit:
DRIFT is already a short-volatility strategy. Carry unwinds coincide with volatility
spikes — both positions would draw down simultaneously in 2008, 2020, 2022, Aug 2024.
Adding carry to a book that already has DRIFT would stack correlated crash risk,
not diversify it.

---

## Research Gaps

| Gap | Priority |
|-----|----------|
| News-driven momentum (NFP, CPI) — 60-second window after release | Low — requires co-location |
| London close (17:00 UTC) mean reversion | Low — unclear mechanism |
| Commodity FX momentum (AUD, CAD) as inflation proxy | Low — niche |

---

## Automation Notes

If OCO scan produces a Go verdict:
- Prefer CME micro FX futures (M6E/M6B/M6J) over IBKR CASH CFD
- Same bracket order infrastructure as FDXS/MNQ/MYM (IBKR Gateway)
- Session-open scheduler: same APScheduler setup; UTC times hardcoded
- Position sizing: single contract per pair

If Currency Momentum produces a Go verdict:
- Weekly rebalance: Friday 21:30 UTC (30 min after last weekly close)
- Orders: market-on-close or limit at last-traded price
- 6 positions: 3 long + 3 short micro FX futures
- Monitor turnover: high-momentum environments -> low turnover, low costs
