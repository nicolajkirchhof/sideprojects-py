---
marp: true
theme: default
paginate: true
header: 'Investing Playbook'
footer: 'Personal Strategy Document'
style: |
  /* ── Global ── */
  section {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    font-size: 20px;
    color: #2C3E50;
    background: #FFFFFF;
    padding: 40px 50px 30px 50px;
  }

  /* ── Title slides ── */
  section.title {
    background: linear-gradient(135deg, #1B2A4A 0%, #2E86AB 100%);
    color: #F8F9FA;
  }
  section.title h1 {
    color: #FFFFFF;
    font-size: 42px;
    font-weight: 700;
    border-bottom: 3px solid #E8B931;
    padding-bottom: 12px;
  }
  section.title table { color: #F8F9FA; }
  section.title th { color: #E8B931; }
  section.title a { color: #E8B931; }

  /* ── Section divider slides ── */
  section.divider {
    background: #1B2A4A;
    color: #F8F9FA;
    display: flex;
    flex-direction: column;
    justify-content: center;
    text-align: center;
  }
  section.divider h2 {
    color: #E8B931;
    font-size: 38px;
    border: none;
  }

  /* ── Headings ── */
  h1 {
    font-size: 34px;
    color: #1B2A4A;
    font-weight: 700;
    margin-bottom: 8px;
  }
  h2 {
    font-size: 28px;
    color: #1B2A4A;
    font-weight: 600;
    border-bottom: 2px solid #2E86AB;
    padding-bottom: 6px;
    margin-bottom: 16px;
  }
  h3 {
    font-size: 22px;
    color: #2E86AB;
    font-weight: 600;
    margin-bottom: 8px;
  }

  /* ── Tables ── */
  table {
    font-size: 16px;
    border-collapse: collapse;
    width: 100%;
    margin: 8px 0;
  }
  th {
    background: #1B2A4A;
    color: #F8F9FA;
    font-weight: 600;
    padding: 6px 10px;
    text-align: left;
  }
  td {
    padding: 5px 10px;
    border-bottom: 1px solid #DEE2E6;
  }
  tr:nth-child(even) td {
    background: #F0F4F8;
  }

  /* ── Blockquotes ── */
  blockquote {
    border-left: 4px solid #2E86AB;
    background: #EBF5FB;
    padding: 12px 16px;
    margin: 10px 0;
    font-size: 18px;
    color: #1B2A4A;
    border-radius: 0 6px 6px 0;
  }

  /* ── Lists ── */
  ul, ol { margin: 4px 0; padding-left: 24px; }
  li { margin: 3px 0; line-height: 1.4; }

  /* ── Bold highlights ── */
  strong { color: #1B2A4A; }

  /* ── Code blocks ── */
  code {
    background: #F0F4F8;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 15px;
  }
  pre {
    background: #1B2A4A;
    color: #F8F9FA;
    padding: 16px;
    border-radius: 6px;
    font-size: 14px;
  }

  /* ── Header/Footer ── */
  header {
    font-size: 12px;
    color: #7F8C8D;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  footer {
    font-size: 11px;
    color: #95A5A6;
  }
---
<!-- _class: title -->

# Investing -- Personal Strategy Document

> Two strategies, one shared thesis: equity markets have structural upward drift. The LONG-TERM portfolio captures it passively over decades. The DRIFT portfolio harvests it through short volatility on the same indices and commodities.

**Instruments: indices and commodity ETFs only.** No individual equities. No stock picking.

| Strategy | Horizon | Approach |
|----------|---------|----------|
| LONG-TERM | 5+ years | Buy & hold globally diversified ETFs -- never sell, never rebalance |
| DRIFT | 30--360 DTE | Sell premium on index and commodity ETFs -- managed weekly |
| Range-Bound Commodities | Opportunistic | Mean reversion on NG/CL at range extremes |

---

## Why I Invest

- Capture long-term proven trends through diversified ETFs
- Tax-efficient: B&H without reallocation is highly tax-positive in Germany
- Harvest shorter-term drift through short volatility on indices and commodities -- especially when the market goes sideways or slightly drops
- Use structures like XYZ (111) to hedge for softer short-term swings
- Stay continuously invested -- theta only works when deployed

---

## 01 -- LONG-TERM Portfolio -- Buy & Hold

### Allocation (Five Factor Model + Quality)

Based on the Fama-French five factor model and the Buffett's Alpha paper. All Avantis funds actively tilt toward value + profitability.

| Sleeve | Target | Rationale | Change from v1 |
|--------|--------|-----------|-----------------|
| US Equity | 35% | Market beta + value/quality tilt | Was 40% pure market cap; reduced 5% |
| Global Small Cap Value | 10% | Size + Value + Profitability premium | Now includes international small value |
| Developed Europe | 15% | Geographic div, value-heavy, includes UK | Merged FTSE 100 into Europe |
| Developed Pacific | 10% | Japan + Australia + HK + Singapore | Broader diversification per dollar |
| Emerging Markets | 10% | EM all-cap with factor tilt | Now full EM coverage |
| Gold / Precious Metals | 10--20% | Crisis hedge, real value store | No change |

---

## 01 -- ETF Selection (UCITS, Accumulating, XETRA)

| Sleeve | Primary ETF | TER | Alternative | TER |
|--------|------------|-----|-------------|-----|
| **US Equity** | AVAE (Avantis America Equity) | 0.20% | VWRA (Vanguard FTSE All-World) | 0.22% |
| **Global SCV** | AVWS (Avantis Global SCV) | 0.39% | ZPRV (SPDR MSCI USA SCV) | 0.30% |
| **Dev. Europe** | AVEU (Avantis Europe Equity) | ~0.20% | MEUD (Lyxor STOXX Europe 600) | 0.07% |
| **Dev. Pacific** | AVPE (Avantis Pacific Equity) | 0.25% | ZJPN (iShares Core MSCI Japan) | 0.15% |
| **Emerging Mkts** | AVEM (Avantis EM Equity) | 0.35% | EMIM (iShares Core MSCI EM) | 0.18% |
| **Gold** | XETRA-GOLD (EWG2) | 0.36% | EGLN (iShares Physical Gold) | 0.12% |

**Weighted TER (primary picks):** ~0.27%. Cost justified by Fama-French and Buffett's Alpha research showing 1--3% expected annual premium.

---

## 01 -- Simplified Alternative & Contribution Rules

**Simplified alternative:** Replace all regional funds with AVWC (0.22%) at 60% + AVWS (10%) + AVEM (10%) + Gold (10--20%). Three equity funds, one gold, full factor tilt. Weighted TER: ~0.26%.

### Contribution Rules

- [ ] Gold sleeve within 10--20%? If above 20% -> add to equities instead
- [ ] New purchase avoids duplicating existing holdings (check overlap)?
- [ ] ETF TER justified by factor exposure?
- [ ] Adding to a broad position, not chasing recent outperformance?
- [ ] Is this a scheduled contribution, not a market-timing decision?

> **Never sell. Never rebalance.** Direct new cash to the most underweight sleeve. This achieves natural rebalancing without triggering taxable events.

---

## 01 -- Review Workflow

1. Show current allocation vs targets
2. Check gold sleeve -- within 10--20%?
3. Flag ETF overlap (shared top holdings across funds)
4. Calculate TER-weighted cost
5. Identify most underweight sleeve -> recommend for next contribution

---

## 02 -- DRIFT Portfolio -- Short Volatility

### Profit Mechanisms

> **Positive drift:** Equity indices have structural upward drift -- selling puts harvests the variance risk premium WITH a directional tailwind. Short naked calls on drifting assets have negative expectancy. *(Carr & Wu 2005, 2009; Mehra & Prescott 1985)*

> **Range-bound VRP:** Commodities, bonds, and precious metals trade in ranges -- no structural drift, but implied vol still overestimates realized vol. Selling premium on BOTH sides harvests the VRP without directional exposure. *(Cheng, Tang & Yan 2021)*

**No individual stocks. No sector ETFs.** Only broad indices, commodities, and bonds.

---

## 02 -- Underlyings: Equity Indices (Positive Drift)

| Underlying | Asset Class | Structure | Tier |
|-----------|------------|-----------|------|
| XSP | US Large Cap (cash-settled) | Short put + kicker / XYZ | Core |
| IWM | US Small Cap | Short put + kicker | Core |
| TQQQ | US Tech 3x leveraged | Short put + kicker | Core |
| ESTX50 | European Equity (futures) | Short put + kicker | Core |
| EEM | Emerging Markets (broad) | IC (IVP>50) | Core |
| FXI | China Large Cap | IC (IVP>50, small) | Rotational |
| EWZ | Brazil | IC (IVP>50) | Rotational |

---

## 02 -- Underlyings: Commodities & Bonds (Range-Bound)

| Underlying | Behavior | Structure | VRP | Backtest |
|-----------|----------|-----------|-----|----------|
| UNG | Extreme vol | **IC only** | +0.019 | **72.2% win, +$4,162** |
| USO | Gap risk | IC (IVP>50) | +0.017 | **58.4% win, +$571** |
| GLD | Crisis / regime | Short put only (regime) | +0.009 | **73.7% win, +$57** |
| WEAT | Seasonal range | IC or strangle (IVP>50) | +0.130 | **76.8% win, +$170** |
| PDBC | Range-bound | IC (IVP>50) | +0.243 | **79.1% win, +$44** |
| DBA | Range-bound | Strangle or IC | +0.032 | **62.7% win, +$39** |
| TLT | Range-bound | IC (IVP>50) | +0.002 | 58.8% win, +$40 |
| BNO | Range-bound | IC (IVP>50) | +0.029 | 59.8% win, +$66 |
| SLV | Whipsaw | IC (deprioritize) | +0.005 | 45.0% win, +$30 |

*Backtest: 45 DTE, 25-delta, 14-day entry interval, delta-approximation.*

---

## 02 -- Directional Block: XSP & IWM

**XSP (S&P 500 Mini, cash-settled)**
- Structure: XYZ 111 (preferred) or short put + kicker call
- Why XSP over SPY: Cash-settled -- no assignment risk. 1/10 SPX size -- good for $50k portfolios.
- IV: Moderate (15--30%). Premium is reliable, not spectacular.
- Transition: Replace with ES futures options at ~$150k+ portfolio size.

**IWM (Russell 2000)**
- Structure: Short 25-delta put + long 20-delta call (or 30/20-delta call spread)
- Why IWM: Highest IV of the US equity indices = richest premium. Small cap index leads recoveries.
- IV: 20--40%. Typically 5--10% higher than SPY.
- Note: More volatile than XSP -- wider stops needed, but premium compensates.

---

## 02 -- Directional Block: TQQQ & ESTX50

**TQQQ (ProShares UltraPro QQQ, 3x leveraged)**
- Structure: Short 25-delta put + long 20-delta call (or 30/20-delta call spread)
- Why TQQQ: Small notional (~$40/share) allows fine-grained scaling at $50k. High IV (60--80%) = very rich premium.
- Risk: 3x leverage means a 30% QQQ drop = ~70-80% TQQQ drop. Daily rebalancing creates decay drag.
- Transition: Replace with QQQ at ~$100k+.

**ESTX50 (Euro STOXX 50 futures options, Eurex)**
- Structure: Short 25-delta put + long 20-delta call
- Why ESTX50: Geographic diversification. Correlation to SPY ~0.65 -- materially lower than IWM (~0.85) or TQQQ (~0.90).
- IV: Typically lower than US (15--25%). Less premium per lot but valuable diversification.
- Note: Trades on Eurex 08:00--22:00 CET. Wider bid-ask than US options.

---

## 02 -- Directional Block: EEM (Primary EM)

**EEM (iShares MSCI Emerging Markets ETF)**
- Structure: Short 25-delta put + long 20-delta call (or 30/20-delta call spread)
- Why EEM: Broadest EM exposure -- China 25%, India, Korea, Taiwan
- Correlation to SPY ~0.55--0.70 -- meaningfully lower than US underlyings
- Driven by different economic cycles: China policy, commodity demand, dollar weakness
- IV: 20--35%. Moderate premium, good liquidity.
- Options volume: ~44,000/day -- highly liquid, penny-pilot, tight bid-ask
- Price: ~$45/share. Notional per lot: ~$4,500 -- good sizing for $50k portfolios.
- Role: **Always-on EM allocation.** Permanent 1--2 lot position.

---

## 02 -- Directional Block: FXI (Rotational EM)

**FXI (iShares China Large Cap ETF)**
- Structure: **Iron condor** (IVP > 50) -- backtest shows IC outperforms short put + kicker
- **Backtest:** IC at 51.8% win, +$71 total. VRP is **negative** (-0.019)
- Why FXI: **Lowest correlation to SPY in the entire directional block (0.30--0.55)**
- Best single-country diversifier. China economy driven by CCP policy, domestic consumption, tech regulation
- IV: 25--45%. Rich premium -- reflects event risk.
- Options volume: ~24,000/day -- liquid enough for 1-lot positions.
- Risk: Concentrated single-country bet. CCP announcements create overnight gaps.
- Role: **Rotational.** Add 1 lot when FXI IVP > 50 and China macro is constructive. Trade small.

---

## 02 -- Directional Block: EWZ (Rotational EM)

**EWZ (iShares MSCI Brazil ETF)**
- Structure: **Iron condor** (IVP > 50) -- +$155 total vs +$80 for strangle
- **Backtest:** IC at 56.7% win rate, +$155 total. VRP +0.013.
- Why EWZ: Commodity-linked EM -- driven by iron ore, soybeans, oil + domestic policy
- Correlation to SPY ~0.40--0.60. Adds LatAm cycle exposure.
- IV: 30--50%. **Richest premium of any non-leveraged country ETF.**
- Risk: Brazilian politics create event risk. BRL adds FX volatility.
- Role: **Rotational.** Add 1 lot when EWZ IVP > 50. If both FXI and EWZ qualify, pick EWZ.

> **EM Rotation Rule:** EEM is always on. FXI and EWZ are rotational -- add when IVP > 50, remove when IVP < 30. Max 2 EM underlyings active at once.

---

## 02 -- Neutral Block: GLD

> These underlyings do NOT have structural positive drift. The edge comes purely from the variance risk premium -- harvested from BOTH sides via strangles or iron condors. *(Erb & Harvey 2013; Cheng et al. 2021)*

**GLD (SPDR Gold Shares)**
- **Regime-dependent structure:**
  - Calm (VIX < 25): Strangle -- sell 25-delta put + sell 25-delta call
  - Crisis (VIX > 25): **Short put only** -- drop the call side
  - Gold spike (>10% above 20d SMA): Short call or call spread only -- mean reversion
- IV: 15--25%. Moderate premium.
- Correlation to SPY: -0.05 to +0.15. **Best crisis diversifier in the portfolio.**
- GLD strangles during calm markets act as **income-generating hedges**

---

## 02 -- Neutral Block: WEAT & PDBC

**WEAT (Teucrium Wheat Fund)**
- Structure: Iron condor or strangle (IVP > 50)
- **Backtest:** IC at 76.8% win rate, +$170 total. VRP +0.130 -- **second highest VRP tested**
- Wheat is strongly seasonal and range-bound. IV massively overestimates actual moves.
- Correlation to SPY: Near zero.
- **Liquidity warning:** ~2,000 options contracts/day -- thin. Limit orders, 1 lot only.

**PDBC (Invesco Optimum Yield Diversified Commodity, No K-1)**
- Structure: Iron condor (IVP > 50)
- **Backtest:** IC at 79.1% win rate -- **highest win rate tested.** VRP +0.243 -- **highest VRP.**
- 14-commodity basket with optimized futures roll. Roll strategy reduces realized vol below IV.
- Correlation to SPY: +0.20 to +0.40.
- **Liquidity warning:** ~1,000--2,000 options contracts/day. Limit orders only, 1 lot.

---

## 02 -- Neutral Block: DBA & BNO

**DBA (Invesco DB Agriculture Fund)**
- Structure: Strangle or iron condor
- **Backtest:** Strangle at 62.7% win rate -- **highest strangle win rate.** VRP +0.032.
- Agriculture basket (corn, soybeans, wheat, sugar, cocoa, coffee, cattle, hogs)
- Basket diversification dampens single-commodity spikes.
- Correlation to SPY: Near zero.
- **Liquidity warning:** ~1,000--3,000 options contracts/day. 1 lot only.

**BNO (United States Brent Oil Fund) -- Optional**
- Structure: Iron condor (IVP > 50)
- **Backtest:** IC at 59.8% win rate, +$66 total. VRP +0.029.
- Adds Brent alongside USO (WTI). Different supply dynamics -- Brent = OPEC+, WTI = US shale.
- Not essential -- USO covers the energy VRP adequately.

---

## 02 -- Neutral Block: SLV & TLT

**SLV (iShares Silver Trust) -- Deprioritized**
- Structure: Iron condor only
- **Backtest: 45.0% win rate, +$30 total.** Whipsaw behavior eats the premium. Stop rate 65.8%.
- Role: **Lower priority than GLD, WEAT, PDBC.** Only add if higher-priority underlyings are deployed.

**TLT (iShares 20+ Year Treasury Bond ETF)**
- Structure: Strangle (symmetric 25-delta) or iron condor
- Why TLT: **Cleanest strangle candidate.** Range-bound ($80--$110 since 2022), moderate IV.
- Correlation to SPY: -0.30 to +0.40. Negative in normal regimes, positive during Fed tightening.
- **Regime filter -- critical:** Only trade when: (1) 20d RV < 20%, AND (2) no FOMC within 14 days of expiry, AND (3) TLT between $85--$105.

---

## 02 -- Neutral Block: USO & UNG

**USO (United States Oil Fund) / CL Futures**
- Structure: Asymmetric strangle (sell 25-delta put + sell 15-20-delta call) or put-heavy IC
- Why asymmetric: Oil rallies on supply shocks are violent and gap overnight. Widen calls.
- Put side benefits from **producer hedging pressure** -- 6.4%/month before costs *(Cheng et al. 2021)*
- IV: 30--50%. **Highest IV of non-leveraged group.** Range: CL $60--$85.
- **Weekend risk:** Oil gaps on geopolitical events. Never hold naked short calls over weekends.

**UNG (United States Natural Gas Fund)**
- Structure: **Iron condor only.** No naked shorts -- ever.
- NG can move 10-20% in a single session. Naked strangles on NG are portfolio-ending trades.
- IV: 40--80%. **Richest premium of any liquid market.**
- **Backtest:** IC at 72.2% win, +$4,162 -- **strongest edge tested.** VRP +0.019.
- Correlation to SPY: Near-zero. Truly independent return stream.

---

## 02 -- Portfolio Composition: Two Blocks

| Block | Underlyings | % of DRIFT BP | Delta Profile | Purpose |
|-------|------------|--------------|--------------|---------|
| **Directional** | XSP, IWM, TQQQ, ESTX50, EEM, FXI/EWZ | 50% | Long delta | Harvest VRP + equity drift |
| **Neutral** | UNG, USO, GLD, WEAT, PDBC, DBA, TLT, BNO, SLV | 50% | Neutral delta | Harvest VRP + crisis diversification |

### Neutral Block Priority (by backtest)

| Priority | Underlying | Win% | VRP | Role |
|----------|-----------|------|-----|------|
| 1 | UNG | 72.2% | +0.019 | Core -- strongest edge |
| 2 | USO | 58.4% | +0.017 | Core -- oil VRP + hedging pressure |
| 3 | GLD | 73.7% | +0.009 | Core -- crisis diversifier |
| 4 | WEAT | 76.8% | +0.130 | Core -- highest commodity VRP |
| 5 | PDBC | 79.1% | +0.243 | Core -- highest win rate |
| 6--9 | DBA, TLT, BNO, SLV | 45--63% | +0.002--0.032 | Selective / Low priority |

Deploy from top to bottom as BP allows. $50k: 4--6 neutral underlyings. $100k+: 7--8.

---

## 02 -- Geographic Diversification & Block Split

### Geographic Diversification (Directional Block)

| Region | Underlyings | Correlation to SPY | Target BP Share |
|--------|-----------|-------------------|-----------------|
| US | XSP, IWM, TQQQ | 0.85--1.00 | 40% of directional |
| Europe | ESTX50 | 0.65 | 20% of directional |
| Emerging Markets | EEM + FXI or EWZ | 0.30--0.70 | 40% of directional |

> **Why 50/50 split:** The neutral block generates stronger risk-adjusted returns than expected. Agriculture and energy VRPs are 5--10x larger than equity indices. The neutral block earns comparable theta while providing crisis diversification.

> **Why this matters in a crash:** The directional block loses. GLD (negative equity correlation), UNG and WEAT (zero equity correlation) hold or gain. Neutral positions earn theta while acting as structural hedges.

---

## 02 -- Vol Direction by Asset Class

| Asset | Price Up = Vol... | Price Down = Vol... | Implication for Call Side |
|-------|-------------------|---------------------|--------------------------|
| Equity indices | Contracts | Expands | Short calls are safer on rallies -- vol drop helps |
| Gold, Silver, Oil, NG | **Expands** | Contracts | Short calls face vol expansion -- widen strike or spreads |
| Bonds (TLT) | Neutral | Neutral | Symmetric -- vol moves with rate uncertainty, not direction |

### DTE Range

**30--360 DTE.** Sweet spot is 45--60 DTE for theta decay. Longer-dated (90--360 DTE) valid for synthetics and PMCC structures. Management is weekly, not daily.

---

## 02 -- Pre-Trade Checklist

- [ ] IVP >= 50 or IVR >= 30 (premium selling edge exists)
- [ ] Underlying above 200d SMA (or use spreads-only if below)
- [ ] VX futures in contango (buy-the-dip environment)
- [ ] IV > HV (implied > historical vol = premium is rich)
- [ ] No FOMC / CPI within 7 days of expiry
- [ ] Adding this trade keeps portfolio BP below 50%
- [ ] Per-trade max loss <= 2% of account (calculate before entry)

---

## 02 -- Trade Structures: Short Put & XYZ

**Short Put** (simplest baseline)
- Entry: 20--30-delta, 45--60 DTE
- Profit: close at 50% of premium received
- Stop: close at 200% of premium received
- Roll: if not profitable at 21 DTE -> roll to next 45--60 DTE for a net credit
- Naked vs spread: naked OK on indices if BP allows exit at 2.5x premium

**XYZ Trade** (preferred -- better risk management)
- Buy X: put at ~30-delta
- Sell Y: put at ~25-delta (forms PDS with X, width ~2--5 points)
- Sell Z: naked put at ~20-delta, same expiry
- Ratios: 111 (conservative) / 221 (moderate) / 112 (aggressive -- caution)
- Manage X/Y and Z independently. Close Z at 50-delta (hard stop).
- Close full structure at 50% max profit or short leg < 10-delta

---

## 02 -- Trade Structures: Kicker & PMCC

**Short Put + Kicker Call** (bullish -- used on drift assets)
- Sell 20--30-delta put + buy 20-delta call (or 30/20-delta call spread) at same expiry
- Net credit or small debit. Premium income + uncapped upside if index runs.
- DTE: 45--360 days depending on conviction
- Use only when underlying clearly above 50d and 200d SMA
- Stop: close short put at 50-delta. Roll to longer DTE per drawdown framework.
- Exit kicker: let call ride -- free lottery ticket on recovery

**PMCC -- Poor Man's Covered Call**
- Buy: deep ITM call >= 120 DTE at 70--80-delta
- Sell: OTM call 7--45 DTE at 20--40-delta against it
- Roll short leg before 21 DTE; do not let long call fall below 45 DTE

### Delta Band

Entry: 20--30-delta -> Hold: 5--45-delta -> Profit close: <10-delta -> Hard stop: 50-delta

---

## 02 -- Position Management (Weekly Review)

| Situation | Action |
|-----------|--------|
| Reaches 50% profit | Close at next weekly review |
| Short leg hits 50-delta | Close immediately -- hard stop, no exceptions |
| At 21 DTE, not profitable | Roll entire structure to next 45--60 DTE for a credit |
| Underlying structurally broken | Take the loss -- do not roll a loser |
| Premise has changed | Close -- never adapt a position to justify staying in it |

### Portfolio-Level Limits

- Per trade max loss: **2% of account**
- Total BP deployed: **<= 50%** (buffer for volatility spikes)
- Single underlying: **<= 30%** of BP
- Theta target: ~0.4% of account per day

---

## 02 -- Regime Adjustments

| Regime | Action |
|--------|--------|
| SPY below 200d SMA | Spreads only, no naked short puts |
| VX backwardation | Reduce size 50%, widen strikes |
| IVP > 80 | Reduce size 50% -- tail risk elevated |
| VIX spikes >25 then "lower high" | High-priority entry signal for new 45--60 DTE puts |
| Quarter-end (final 3 days) | No new long-delta positions |
| 20d realized vol < 15% | Full size |
| 20d realized vol 15--25% | Reduce size 25--30% |
| 20d realized vol > 25% | Reduce size 50%, widen strikes |

---

## 02 -- Drawdown Scaling

> When indices fall, premium gets richer and the structural drift tailwind strengthens for the eventual recovery. Scale BP allocation from 30% up to 80% as drawdown deepens -- but only when VIX confirms fear is elevated.

Both conditions must be met to advance a tier:

| Tier | SPY Drawdown | VIX Range | DRIFT BP | Structure | DTE |
|------|-------------|-----------|----------|-----------|-----|
| Normal | 0% to -5% | < 20 | 30% | XYZ 111, short puts | 45--60 |
| Elevated | -5% to -10% | 20--30 | 40% | XYZ 111, short puts | 45--90 |
| Correction | -10% to -20% | 25--40 | 55% | XYZ 221, short puts, synthetics | 60--120 |
| Deep Correction | -20% to -30% | 35--55 | 70% | XYZ 221, synthetics, LEAPS | 90--180 |
| Bear / Capitulation | > -30% | > 50 | 80% | Spreads only, wide strikes, LEAPS | 180--360 |

---

## 02 -- Drawdown Scaling: Historical Basis

| Event | Drawdown | VIX Peak | Tier | Recovery Time |
|-------|----------|----------|------|---------------|
| 1987 Crash | -34% | 150+ | Bear | ~20 months |
| 1990 Gulf War | -20% | 36 | Deep Correction | ~4 months |
| 2000--02 Dot-com | -49% | 45 | Bear | ~56 months |
| 2007--09 GFC | -57% | 80 | Bear | ~49 months |
| 2011 Debt Ceiling | -19% | 48 | Correction | ~4 months |
| 2018 Q4 | -20% | 36 | Deep Correction | ~4 months |
| 2020 COVID | -34% | 82 | Bear | ~5 months |
| 2022 Bear | -25% | 37 | Deep Correction | ~14 months |

Corrections (-10% to -20%) recover in 3--7 months. Bear markets (-30%+) can take years -- hence LEAPS only at that tier.

---

## 02 -- Drawdown Scaling Rules

1. **Scale IN weekly, not all at once.** Deploy additional BP over 2--4 weeks. No catching falling knives on a single day.
2. **Widen DTE as you scale up.** Normal: 45--60. Correction: 60--120. Deep/Bear: 90--360.
3. **Spreads only below 200d SMA.** At Deep Correction and Bear tiers, no naked short puts.
4. **Scale DOWN faster than you scale up.** When VIX drops below tier threshold, reduce at next weekly review.
5. **Bear tier = LEAPS only.** At >-30% with VIX >50, only 180--360 DTE positions.
6. **50-delta hard stop is never suspended.** Scaling determines how much BP to deploy -- not whether to hold losers.

---

## 02 -- Portfolio Hedging: When to Hedge

> Scaling is offense -- deploying more capital into fear. Hedging is defense -- limiting the damage when fear is justified.

**Why hedge?** DRIFT is structurally short gamma. In tail events, losses are non-linear -- volatility jumps, correlations spike to 1, and the 50% BP buffer can evaporate in days.

| Regime | VIX | Hedge Action |
|--------|-----|-------------|
| Calm (< 15) | Cheapest protection | **Buy tail hedges.** Spend 0.5--1% of account per quarter. |
| Normal (15--20) | Moderate | **Maintain existing hedges.** Roll expiring protection. |
| Elevated (20--30) | Expensive | **Stop buying new hedges.** Let existing hedges work. |
| Stress (> 30) | Too late to buy | **Monetize hedges.** Sell profitable hedges into the spike. |

> **Key insight:** Buy protection when you don't need it. By the time you need it, it's too expensive.

---

## 02 -- Hedge Structures & Budget

| Structure | Cost | When to Use | Sizing |
|-----------|------|-------------|--------|
| Far-OTM SPX puts (5--10-delta) | Low | Calm regime. 90--120 DTE. | 0.25--0.5% per quarter |
| VIX call spreads (e.g., 20/35) | Low--Med | Calm regime. 60--90 DTE. | 0.25--0.5% per quarter |
| Risk-reversal overlay | Near-zero | Any regime. Built into DRIFT structure. | No extra cost |
| Reduce gross exposure | Free | Elevated+ regime. Smallest book = cheapest hedge. | Per regime table |

| Item | Annual Budget |
|------|--------------|
| Far-OTM SPX puts (4 quarterly rolls) | ~0.5--1.0% |
| VIX call spreads (4 quarterly rolls) | ~0.5--1.0% |
| **Total hedge cost** | **~1--2%** |
| DRIFT target theta | ~0.4%/day x 252 = ~100%+ of deployed BP |
| **Net: hedge cost < 2% of gross theta income** | |

---

## 02 -- Hedge Management Rules

1. **Buy in calm, monetize in stress.** Never buy VIX calls when VIX > 25.
2. **Roll quarterly.** Roll 90 DTE positions at 30 DTE remaining.
3. **Never hedge with the same structure you're selling.** Use VIX calls or far-OTM strikes that don't overlap with DRIFT positions.
4. **Gold is a passive hedge.** Your 10--20% LONG-TERM gold allocation already provides crash protection. Don't duplicate in DRIFT.
5. **Fed tightening = correlation hedge breaks.** Stocks and bonds fall together. Reduce DRIFT size, don't try to hedge with bonds.

> **The LTCM reminder:** LTCM ran a short-vol book with 25:1 leverage and no tail hedges. The 1998 Russian crisis produced losses that models said were impossible. Your 50% BP cap, 2% max loss, and hedge budget exist because of this lesson.

---

## 03 -- Range-Bound Commodities (Opportunistic)

> NG, CL (and other commodities) trade in observable multi-year ranges based on political factors, weather, and seasonality. Not a systematic strategy -- trade only when the setup is clear.

**Setup:** Enter when price reaches the top/bottom of the last reversal and shows a clear indication of reversal (wick, towers, double top/bottom).

**Structure:**
- Long premium if the underlying moves in the direction of increasing volatility (commodities: upside = vol expansion)
- Short premium if the underlying moves in the direction of vol contraction

**Research gap:** The trading ranges need to be quantified. Chart analysis suggests consistent patterns, but this is not yet backed by systematic data.

---

## 04 -- Weekly Review

Run every weekend. This is the only scheduled review cadence for the investing portfolio.

### Market Context

| Check | What to Look For |
|-------|-----------------|
| VIX / VX | Level, direction, contango vs backwardation |
| SPY vs RSP | Cap-weighted vs equal-weight -- divergence = narrowing breadth |
| Advances / Declines | With 10 & 20 EMA -- breadth trend |
| Commodities | GC, CL, NG, ZN, ZB -- range positions |
| Global | Nikkei, ESTX, EEM, FXI -- international picture |
| Regime ratios | XLY/XLP (sentiment) / XLK/XLF (growth) / QQQ/SPY (risk) |

---

## 04 -- Weekly Portfolio Actions

1. IVP scan across DRIFT underlyings -- which has elevated premium this week?
2. Review open positions: any at 50% profit? Any approaching 21 DTE? Any at 50-delta stop?
3. Commodity range check -- GC, CL, NG approaching extremes?
4. LONG-TERM: any scheduled contribution this month? -> run contribution checklist
5. Log position changes and update trade journal

---

## 05 -- Non-Negotiable Rules

1. **Never sell LONG-TERM holdings** -- redirect contributions; never realise taxable gains
2. **BP <= 50% at all times** -- the buffer is the risk management system
3. **2% max loss per DRIFT trade** -- calculate position size before every entry
4. **200% stop on short premium** -- close immediately, no exceptions
5. **50-delta hard stop** -- when short leg hits 50-delta, close it now
6. **IVP >= 50 / IVR >= 30** -- if premium is cheap, wait
7. **Never adapt a failing trade** -- close it; re-enter only on a fresh setup
8. **Stay continuously invested in DRIFT** -- theta only works when deployed
9. **Indices and commodity ETFs only** -- no individual equities in DRIFT
10. **Log every trade** -- review weekly for systematic errors and rule violations

---

## 06 -- Research-Backed Edges: LONG-TERM

Actionable findings from 161 academic papers and 114 books.

**Turn-of-month effect (McConnell & Xu, 2008):** Index returns concentrate in the last trading day through the first three days of the next month -- pension fund inflows. Schedule ETF contributions for the last day of the month.

**Quarter-end rebalancing headwind (Harvey et al., 2025):** ~17bps daily headwind in the final 3 days of each quarter. Buy in the first week of the new quarter when forced selling ends.

**Small cap value tilt (Fama-French 1993, 2015):** Size + value premium adds 1--3% expected annual return. Unreliable over 1--3 years but robust over 10+.

**Quality factor (Frazzini et al., 2018):** Buffett's alpha = quality + value + leverage. Consider adding QUAL/JQUA at 5--10% as portfolio grows.

---

## 06 -- Research-Backed Edges: LONG-TERM (cont.)

**Gold allocation discipline (Erb & Harvey, 2013):** Gold mean-reverts over long horizons. Real prices above historical average predict lower future returns. Your 10--20% band is correct.

**Passive flow distortions (Brightman & Harvey, 2025):** Index-fund dominance increases co-movement and inflates cap-weighted overvaluation. Mitigant: SCV and international tilts are underweighted by passive flows.

---

## 06 -- Research-Backed Edges: DRIFT

**Variance risk premium is structural (Carr & Wu, 2005, 2009):** IV systematically exceeds RV on indices. Premium is larger for indices than individual stocks -- confirms "indices only" rule.

**VIX-conditional entry (de Saint-Cyr, 2023):** IC and short put win rates vary dramatically with VIX at entry. Selling premium when VIX > 20 with IVP > 50 has highest probability of profit. Never sell when VIX < 15.

**Weekend theta mispricing (Jones & Shemesh, 2017):** Options overpriced over weekends -- variance allocated to calendar days, not trading days. Enter new positions Thursday/Friday.

**Dealer gamma as regime signal (SqueezeMetrics):** When GEX is positive, hedging suppresses realized vol -- ideal for selling premium. When GEX flips negative, reduce short vol exposure.

---

## 06 -- Research-Backed Edges: DRIFT (cont.)

**Skew premium = variance premium (Kozhan et al., 2013):** Selling puts and selling straddles capture the same risk factor (~0.9 correlation). Treat all short-vol positions as one risk bucket. 50% BP cap applies to the total.

**IV term structure slope (Vasquez, 2017):** When 90d IV >> 30d IV (steep), longer-dated puts are overpriced -- sell them. Flat or inverted = near-term risk elevated -- reduce or widen.

**Retail as structural counter-party (Hu et al., 2024):** 66% of retail options traders lose money on simple one-sided bets. They overpay for short-dated OTM options. You are the other side.

**Vol mean reversion after extremes (He, 2013):** After extreme moves, volatility reliably reverts. Selling premium after vol spikes is statistically highest-EV entry.

---

## 06 -- Future Profit Mechanisms (Research Phase)

| Mechanism | Source | Potential Expression | Status |
|-----------|--------|---------------------|--------|
| Risk-reversal premium | Hull & Sinclair 2021 | Sell OTM put + buy OTM call on indices | Research: validate on XSP/SPY |
| VIX futures roll yield | Cooper 2013 | Short front-month VIX in contango | Research: sizing and tail risk |
| Commodity VRP | Cheng/Tang/Yan 2021 | Short OTM puts on CL/GC; put spreads on NG/ZS/ZC | Research: backtest per commodity |
| Bond VRP | Cross-asset VRP | Short strangles/IC on ZN/ZB in stable rate regimes | Research: regime-dependency |
| Currency carry + VRP | Koijen et al. 2018 | Short strangles on 6E/6J when RV low | Research: backtest, liquidity |
| Managed vol overlay | Cooper 2010 | Scale inversely to 20d RV | Partially implemented |
| Regime-conditional timing | Harvey/Man Group 2025 | Macro regime detection for factor timing | Research: define variables |
| E/P timing signal | Shen 2002 | E/P minus T-bill rate predicts drawdowns | Research: add to weekly review |

---

## 06 -- Commodity VRP Structure Notes

| Factor | Equity Indices | Commodities (CL, NG, ZS, ZC) |
|--------|---------------|-------------------------------|
| Vol direction | Upside = vol contraction | Upside = vol expansion |
| Put pricing | Overpriced (hedging demand) | Overpriced (producer hedging) |
| Call pricing | Fairly priced | Underpriced (vol expansion on rallies) |
| Best short vol structure | Short puts, XYZ | Short puts at range bottom; strangles on CL/GC only |
| Avoid | -- | Naked strangles on NG (vol too spikey) |

**Hedging pressure (Cheng et al., 2021):** Commercial producers buy puts to protect physical inventory. Selling puts opposite to hedger flow earns 6.4%/month before costs.

**Key constraint:** Commodity futures options have lower liquidity and wider bid-ask. Use defined-risk structures on thinner markets (NG, ZS, ZC). Naked puts acceptable on CL and GC.

---

## 06 -- Bond & Currency VRP: Research Questions

**Bonds (ZN, ZB options):**
1. What is the average IV-RV spread on ZN/ZB over the last 10 years?
2. How does the VRP behave during Fed pivot points?
3. What is the optimal DTE and delta for short strangles on ZN?
4. When does selling bond vol add diversification vs. doubling the same risk?

**Currencies (6E, 6J options):**
1. What is the average IV-RV spread on 6E/6J?
2. Does carry trade combine with or offset short vol on the same pairs?
3. What is the liquidity and bid-ask on 6E/6J at 45--60 DTE and 20--30-delta?
4. Are there seasonal patterns in FX vol (year-end, JPY fiscal year-end)?

---

## Open Research / TBD

### Active Research
- **Commodity VRP backtest:** Short put strategies on CL, NG, ZS, ZC across multiple cycles
- **Bond VRP feasibility:** IV-RV spread on ZN/ZB. Short strangles during stable rate regimes.
- **Currency VRP feasibility:** IV-RV spread on 6E/6J. Short strangles at 45--60 DTE.
- **Hedge calibration:** Backtest 1--2% annual hedge budget against 2018 Q4, 2020 COVID, 2022 bear, 2024-08, 2025-04

### Validation Needed
- **Commodity range quantification:** Ranges observed but not yet measured systematically
- **Seasonality exploitation:** July = most positive month historically -- systematic trade?
- **Turn-of-month timing:** Validate pension flow effect on specific ETFs

### Integration Tasks
- **GEX integration:** Add dealer gamma to weekly review
- **Term structure monitoring:** Weekly 30d vs 90d IV slope check
- **Cross-asset VRP dashboard:** Track IV-RV across all DRIFT underlyings + commodities + bonds + currencies
