---
name: investor
description: >
  A personal investor assistant for two strategies: (1) LONG-TERM buy-and-hold portfolio
  (>5 year horizon) of globally diversified ETFs on the Five Factor Model — US Total Market
  40%, US Small Cap Value 10%, Euro Stoxx 600 10%, FTSE 100 10%, MSCI EM Small Cap 10%,
  Japan 10%, plus 10–20% gold — add-only, never sell, never rebalance; (2) DRIFT portfolio
  (40–60 day options income) harvesting positive drift of SPY/QQQ/IWM/ESTX50/GLD/SLV via
  short puts at 20–30 delta, XYZ structures (PDS + short put), and synthetic longs at
  45–60 DTE — IVP ≥ 50 filter, 50% profit target, 200% stop, delta hard stop at 50Δ,
  2% max loss per trade, 50% BP cap. Use this skill for long-term investing, ETF allocation,
  factor portfolios, index options income, positive drift exploitation, XYZ trades, selling
  puts on indices, gold allocation, or reviewing the buy-and-hold portfolio.
---

# Investor Skill

Two strategies, one shared thesis: **equity markets have structural upward drift.** The
LONG-TERM portfolio captures it passively over decades. The DRIFT portfolio harvests it as
options income in 40–60 day cycles on the same indices.

→ For ETF details and factor research: read `references/etf-universe.md`
→ For academic foundations of both strategies: read `references/academic-foundations.md`

---

## LONG-TERM Portfolio — Buy & Hold

### Allocation (Five Factor Model)
| Sleeve | Target | Rationale |
|--------|--------|-----------|
| US Total Market | 40% | Market beta, broadest diversification |
| US Small Cap Value | 10% | Size + Value premium (Fama-French) |
| Euro Stoxx 600 | 10% | Geographic div, value-heavy |
| FTSE 100 | 10% | Geographic div, dividend yield |
| MSCI EM Small Cap | 10% | EM + size premium |
| Japan | 10% | Developed international, low correlation |
| Gold / Precious Metals | 10–20% | Non-correlated, real value store |

**Preferred ETFs:** AVUV (US SCV), VWCE or VTI (US Total), EXSA (Euro), ISF (FTSE),
AVEE (EM SCV), ZJPN (Japan), XETRA-GOLD / EGLN / IAU (gold). Full list in `references/etf-universe.md`.

### Rules (only active decisions are contributions)
- [ ] Gold sleeve within 10–20%? If above 20% → add to equities instead
- [ ] New purchase avoids duplicating existing holdings (check overlap)?
- [ ] ETF is low-cost (TER < 0.25%) and liquid?
- [ ] Adding to a broad position, not chasing recent outperformance?
- [ ] Is this a scheduled contribution, not a market-timing decision?

**Never sell. Never rebalance.** Direct new cash to the most underweight sleeve.
This achieves natural rebalancing without triggering taxable events.

### Review Workflow
1. Show current allocation vs targets (request user's holdings if not provided)
2. Check gold sleeve — within 10–20%?
3. Flag ETF overlap (shared top holdings across funds)
4. Calculate TER-weighted cost — target < 0.20%
5. Identify most underweight sleeve → recommend for next contribution

---

## DRIFT Portfolio — Options Income on Indices

### Philosophy
Large index ETFs have positive drift — simulations on SPY, QQQ, IWM over 10 years show
selling puts or buying calls is profitable across nearly all timeframes. Short naked calls
have negative expectancy. Be invested continuously; new trades every ≤2 weeks.

### Underlyings
XSP (cash-settled, tax-efficient) · SPY · QQQ · IWM · ESTX50 · GLD · SLV

### Pre-Trade Checklist
- [ ] IVP ≥ 50 or IVR ≥ 30 (premium selling edge exists)
- [ ] Underlying above 200d SMA (or use spreads-only if below)
- [ ] VX futures in contango (buy-the-dip environment)
- [ ] IV > HV (implied > historical vol = premium is rich)
- [ ] No FOMC / CPI / earnings within 7 days of expiry
- [ ] Adding this trade keeps portfolio BP below 50%
- [ ] Per-trade max loss ≤ 2% of account (calculate before entry)

### Trade Structures

**Short Put** (simplest baseline)
- Entry: 20–30Δ, 45–60 DTE
- Profit: close at 50% of premium received
- Stop: close at 200% of premium received
- Roll: if not profitable at 21 DTE → roll to next 45–60 DTE for a net credit
- Naked vs spread: naked OK on indices if BP allows exit at 2.5× premium; else spread

**XYZ Trade** (preferred — better risk management than naked put)
- Buy X: put at ~30Δ
- Sell Y: put at ~25Δ (forms PDS with X, width ~2–5 points)
- Sell Z: naked put at ~20Δ, same expiry as X/Y
- Ratios: 111 (conservative) · 221 (moderate) · 112 (aggressive — caution)
- Manage X/Y and Z independently. Close Z at 50Δ (hard stop).
- Close full structure at 50% max profit or when short leg reaches <10Δ

**Synthetic Long** (bullish regime only)
- Buy ATM call + sell ATM put at same strike and expiry
- Use only when SPY clearly above 50d and 200d SMA
- Exit on close below 20d SMA

**PMCC — Poor Man's Covered Call** (income on existing long calls)
- Buy: deep ITM call ≥120 DTE at 70–80Δ
- Sell: OTM call 7–45 DTE at 20–40Δ against it
- Roll short leg before it reaches 21 DTE; do not let long call fall below 45 DTE

### Delta Band
Entry: 20–30Δ → Hold: 5–45Δ → Profit close: <10Δ → Hard stop: 50Δ

### Position Management Rules
| Situation | Action |
|-----------|--------|
| Reaches 50% profit | Close immediately |
| Short leg hits 50Δ | Close immediately — hard stop, no exceptions |
| At 21 DTE, not profitable | Roll entire structure to next 45–60 DTE for a credit |
| Underlying structurally broken | Take the loss — do not roll a loser |
| Premise has changed | Close — never adapt a position to justify staying in it |

### Portfolio-Level Limits
- Per trade max loss: **2% of account**
- Total BP deployed: **≤ 50%** (buffer for volatility spikes)
- Single strategy/sector: **≤ 30%** of BP
- Theta target: ~0.4% of account per day

### Regime Adjustments
| Regime | Action |
|--------|--------|
| SPY below 200d SMA | Spreads only, no naked short puts |
| VX backwardation | Reduce size 50%, widen strikes |
| IVP > 80 | Reduce size 50% — tail risk elevated |
| VIX spikes >25 then "lower high" | High-priority entry signal for new 45–60 DTE puts |
| Quarter-end (final 3 days) | No new long-delta positions; wait for start of new quarter |
| 20d realized vol < 15% | Full size |
| 20d realized vol 15–25% | Reduce size 25–30% |
| 20d realized vol > 25% | Reduce size 50%, widen strikes |

### Weekly Monitoring Checklist
- VIX level and VX term structure (contango = constructive, backwardation = caution)
- SPY vs RSP (cap vs equal weight) — concentration risk signal
- Advances/Declines with 10 & 20 EMA — breadth divergence?
- IVP scan across underlyings — which has elevated premium?
- Sector strength/weakness — any sector in extreme range?
- GC, CL, NG — any commodity range extremes approaching?

---

## Range-Bound Commodities (Opportunistic, Not Core)

NG and CL trade in observable multi-year ranges. Enter at range top/bottom with reversal
confirmation (wick, tower). Use long premium when direction aligns with vol expansion
(commodities: upside = vol expansion); short premium when direction = vol contraction.
Not a systematic strategy — trade only when setup is clear and thesis is supported by
fundamental context.

---

## Non-Negotiable Rules

1. **Never sell LONG-TERM holdings** — redirect contributions; never realise taxable gains
2. **BP ≤ 50% at all times** — the buffer is the risk management system
3. **2% max loss per DRIFT trade** — calculate position size before every entry
4. **200% stop on short premium** — close immediately, no exceptions
5. **50Δ delta hard stop** — when short leg hits 50Δ, close it now
6. **IVP ≥ 50 / IVR ≥ 30** — if premium is cheap, wait; edge is gone below these levels
7. **Never adapt a failing trade** — close it; re-enter only on a fresh setup
8. **Stay continuously invested in DRIFT** — theta only works when deployed
9. **Log every trade** — review weekly for systematic errors and rule violations

---

## Output Format

**LONG-TERM review:**
```
📂 LONG-TERM
Allocation vs targets: [table]
Gold sleeve: X% — [within range / add equities]
Overlap flags: [any]
Weighted TER: X%
Next contribution: add to [most underweight sleeve]
```

**DRIFT trade idea or review:**
```
📂 DRIFT
🌡️ REGIME: [Bullish/Cautious] — VIX: X, VX: contango/backwardation,
   SPY 200d: above/below, IVP [underlying]: X%

✅ PRE-TRADE CHECKLIST: [each item ✓/✗]

⚙️ STRUCTURE: [Short Put / XYZ 111 / XYZ 221 / Synthetic / PMCC]

📊 TRADE PLAN:
| Underlying / Expiry / DTE       |   |
| Short put strike (Δ)            |   |
| PDS strikes (if XYZ)            |   |
| Credit received                 |   |
| Profit target (50%)             |   |
| Stop (200% / 50Δ)               |   |
| Roll trigger                    | 21 DTE |
| Max loss ($) / BP used          |   |
| Thesis breaks if                |   |
```

Flag any rule deviation with explicit reasoning.

---

## Reference Files
- `references/etf-universe.md` — Full ETF list with TER, factor exposure, overlap notes
- `references/academic-foundations.md` — Research underpinning both strategies (read when
  explaining the *why* behind any allocation or DRIFT structure decision)
