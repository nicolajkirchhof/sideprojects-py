# Investing — Personal Strategy Document

Two strategies, one shared thesis: equity markets have structural upward drift. The LONG-TERM portfolio captures it passively over decades. The DRIFT portfolio harvests it through short volatility on the same indices and commodities.

**Instruments: indices and commodity ETFs only.** No individual equities. No stock picking.

| Strategy | Horizon | Approach |
|----------|---------|----------|
| LONG-TERM | 5+ years | Buy & hold globally diversified ETFs — never sell, never rebalance |
| DRIFT | 30–360 DTE | Sell premium on index and commodity ETFs — managed weekly |
| Range-Bound Commodities | Opportunistic | Mean reversion on NG/CL at range extremes |

---

## Why I Invest

- Capture long-term proven trends through diversified ETFs
- Tax-efficient: B&H without reallocation is highly tax-positive in Germany
- Harvest shorter-term drift through short volatility on indices and commodities — especially when the market goes sideways or slightly drops
- Use structures like XYZ (111) to hedge for softer short-term swings
- Stay continuously invested — theta only works when deployed

---

## 01 — LONG-TERM Portfolio — Buy & Hold

### Allocation (Five Factor Model + Quality)

Based on the Fama-French five factor model and the Buffett's Alpha paper. All Avantis funds actively tilt toward value + profitability — embedding the quality factor into every sleeve without requiring a separate allocation.

| Sleeve | Target | Rationale | Change from v1 |
|--------|--------|-----------|-----------------|
| US Equity | 35% | Market beta + value/quality tilt | Was 40% pure market cap; reduced 5% to fund broader international |
| Global Small Cap Value | 10% | Size + Value + Profitability premium | Was US SCV only; now includes international small value |
| Developed Europe | 15% | Geographic div, value-heavy, includes UK | Merged FTSE 100 into Europe — removed 80%+ overlap |
| Developed Pacific | 10% | Japan + Australia + HK + Singapore | Was Japan-only; broader diversification per dollar |
| Emerging Markets | 10% | EM all-cap with factor tilt | Was EM small cap only; now full EM coverage |
| Gold / Precious Metals | 10–20% | Crisis hedge, real value store | No change |

### ETF Selection (all UCITS, accumulating, XETRA-listed)

| Sleeve | Primary ETF | TER | Alternative | TER | Notes |
|--------|------------|-----|-------------|-----|-------|
| **US Equity** | AVAE (Avantis America Equity UCITS) | 0.20% | VWRA (Vanguard FTSE All-World Acc) | 0.22% | AVAE: value+quality tilt on US/Canada. VWRA: global market cap, no tilt. |
| **Global SCV** | AVWS (Avantis Global Small Cap Value UCITS) | 0.39% | ZPRV (SPDR MSCI USA Small Cap Value Weighted) | 0.30% | AVWS: global SCV with profitability screen. ZPRV: US-only, cheaper, index-based. |
| **Dev. Europe** | AVEU (Avantis Europe Equity UCITS) | ~0.20% | MEUD (Lyxor Core STOXX Europe 600) | 0.07% | AVEU: factor tilt across all caps. MEUD: cheapest Europe exposure, no tilt. |
| **Dev. Pacific** | AVPE (Avantis Pacific Equity UCITS) | 0.25% | ZJPN (iShares Core MSCI Japan IMI) | 0.15% | AVPE: full Pacific with factor tilt. ZJPN: Japan-only, cheaper. |
| **Emerging Mkts** | AVEM (Avantis EM Equity UCITS) | 0.35% | EMIM (iShares Core MSCI EM IMI) | 0.18% | AVEM: factor tilt, all EM caps. EMIM: broad EM, lowest cost. |
| **Gold** | XETRA-GOLD (EWG2) | 0.36% | EGLN (iShares Physical Gold) | 0.12% | XETRA-GOLD: physical, EUR, German Börse. EGLN: cheapest, USD/London. |

**Weighted TER (primary picks):** ~0.27%. Above the 0.20% target but every dollar is factor-tilted toward value + profitability. The cost is justified by the Fama-French and Buffett's Alpha research showing 1–3% expected annual premium from these tilts.

**Simplified alternative:** Replace all regional funds with AVWC (Avantis Global Equity UCITS, 0.22%) at 60% + AVWS (10%) + AVEM (10%) + Gold (10–20%). Three equity funds, one gold, full factor tilt. Weighted TER: ~0.26%.

### Contribution Rules

- [ ] Gold sleeve within 10–20%? If above 20% → add to equities instead
- [ ] New purchase avoids duplicating existing holdings (check overlap)?
- [ ] ETF TER justified by factor exposure? (accept higher TER for Avantis factor tilt)
- [ ] Adding to a broad position, not chasing recent outperformance?
- [ ] Is this a scheduled contribution, not a market-timing decision?

> **Never sell. Never rebalance.** Direct new cash to the most underweight sleeve. This achieves natural rebalancing without triggering taxable events.

### Review Workflow

1. Show current allocation vs targets
2. Check gold sleeve — within 10–20%?
3. Flag ETF overlap (shared top holdings across funds)
4. Calculate TER-weighted cost
5. Identify most underweight sleeve → recommend for next contribution

---

## 02 — DRIFT Portfolio — Short Volatility on Indices & Commodities

### Profit Mechanisms

The DRIFT portfolio exploits two distinct mechanisms depending on the underlying:

> **Positive drift:** Equity indices have structural upward drift — selling puts harvests the variance risk premium WITH a directional tailwind. Short naked calls on drifting assets have negative expectancy. *(Carr & Wu 2005, 2009; Mehra & Prescott 1985; tastylive simulations)*

> **Range-bound VRP:** Commodities, bonds, and precious metals trade in ranges — no structural drift, but implied vol still overestimates realized vol. Selling premium on BOTH sides (strangles, iron condors) harvests the VRP without directional exposure. *(Cross-asset VRP: multiple; Hedging pressure: Cheng, Tang & Yan 2021)*

**No individual stocks. No sector ETFs.** Only broad indices, commodities, and bonds.

### Underlyings — Structure by Asset

The structure depends on the underlying's behavior. Drift assets get directional structures. Range-bound assets get neutral structures.

| Underlying | Asset Class | Behavior | Structure | VRP | Tier | Backtest |
|-----------|------------|----------|-----------|-----|------|----------|
| XSP | US Large Cap (cash-settled) | Positive drift | Short put + kicker / XYZ | — | Core | — |
| IWM | US Small Cap | Positive drift | Short put + kicker | — | Core | — |
| TQQQ | US Tech 3x leveraged | Positive drift (3x) | Short put + kicker | — | Core | — |
| ESTX50 | European Equity (futures) | Positive drift | Short put + kicker | — | Core | — |
| EEM | Emerging Markets (broad) | Positive drift (EM) | IC (IVP>50) | +0.001 | Core | 58.6% win, +$90 |
| FXI | China Large Cap | Positive drift (China) | IC (IVP>50, small) | -0.019 | Rotational | 51.8% win, +$71 |
| EWZ | Brazil | Positive drift (commodity-EM) | IC (IVP>50) | +0.013 | Rotational | 56.7% win, +$155 |
| UNG | Natural Gas ETF | Range-bound + extreme vol | **IC only** | +0.019 | **Tier 1** | **72.2% win, +$4,162** |
| USO | Crude Oil ETF | Range-bound + gap risk | IC (IVP>50) | +0.017 | **Tier 1** | **58.4% win, +$571** |
| GLD | Gold | Crisis asset / regime-dependent | Short put only (regime) | +0.009 | **Tier 1** | **73.7% win, +$57** |
| WEAT | Wheat ETF | Seasonal range-bound | IC or strangle (IVP>50) | +0.130 | **Tier 1** | **76.8% win, +$170** |
| PDBC | Broad commodity basket | Range-bound (optimized roll) | IC (IVP>50) | +0.243 | **Tier 1** | **79.1% win, +$44** |
| DBA | Agriculture basket | Range-bound | Strangle or IC | +0.032 | **Tier 1** | **62.7% win, +$39** |
| TLT | 20+ Year Treasuries | Range-bound | IC (IVP>50) | +0.002 | Tier 2 | 58.8% win, +$40 |
| BNO | Brent Crude Oil | Range-bound | IC (IVP>50) | +0.029 | Tier 2 | 59.8% win, +$66 |
| SLV | Silver | Whipsaw | IC (deprioritize) | +0.005 | Tier 3 | 45.0% win, +$30 |

*Backtest: 45 DTE, 25Δ, 14-day entry interval, delta-approximation. See `finance/core_pm/backtest_findings.md` for full methodology and caveats.*

---

### Directional Block — Equity Indices (Positive Drift)

> Equity indices drift upward structurally. Sell puts to harvest VRP with the drift working in your favor. Add a kicker call or call spread to capture upside if the index runs. Never sell naked calls on drifting assets — negative expectancy. *(Carr & Wu 2005; Who Profits From Trading Options, Hu et al. 2024)*

**XSP (S&P 500 Mini, cash-settled)**
- Structure: XYZ 111 (preferred) or short put + kicker call
- Why XSP over SPY: Cash-settled — no assignment risk. 1/10 SPX size — good for $50k portfolios.
- IV: Moderate (15–30%). Premium is reliable, not spectacular.
- Transition: Replace with ES futures options at ~$150k+ portfolio size.

**IWM (Russell 2000)**
- Structure: Short 25Δ put + long 20Δ call (or 30/20Δ call spread)
- Why IWM: Highest IV of the US equity indices = richest premium. Small cap index leads recoveries.
- IV: 20–40%. Typically 5–10% higher than SPY.
- Note: More volatile than XSP — wider stops needed, but premium compensates.

**TQQQ (ProShares UltraPro QQQ, 3x leveraged)**
- Structure: Short 25Δ put + long 20Δ call (or 30/20Δ call spread)
- Why TQQQ: Small notional (~$40/share) allows fine-grained scaling at $50k. High IV (60–80%) = very rich premium.
- Risk: 3x leverage means a 30% QQQ drop = ~70-80% TQQQ drop. Daily rebalancing creates decay drag over months — but this is priced into the higher IV you collect.
- Transition: Replace with QQQ at ~$100k+ when lot sizing is no longer a constraint.

**ESTX50 (Euro STOXX 50 futures options, Eurex)**
- Structure: Short 25Δ put + long 20Δ call
- Why ESTX50: Geographic diversification. Correlation to SPY ~0.65 — materially lower than IWM (~0.85) or TQQQ (~0.90). EUR-denominated — no FX conversion for EUR accounts.
- IV: Typically lower than US (15–25%). Less premium per lot but valuable diversification.
- Note: Trades on Eurex 08:00–22:00 CET. Wider bid-ask than US options.

**EEM (iShares MSCI Emerging Markets ETF) — Primary EM Position**
- Structure: Short 25Δ put + long 20Δ call (or 30/20Δ call spread)
- Why EEM: Broadest EM exposure in a single product — China 25%, India, Korea, Taiwan. Correlation to SPY ~0.55–0.70 — meaningfully lower than US underlyings. Driven by different economic cycles: China policy, commodity demand, dollar weakness.
- IV: 20–35%. Moderate premium, good liquidity.
- Options volume: ~44,000/day — highly liquid, penny-pilot, tight bid-ask. *(10th most liquid ETF options globally)*
- Price: ~$45/share. Notional per lot: ~$4,500 — good sizing for $50k portfolios.
- Role: **Always-on EM allocation.** Permanent 1–2 lot position.

**FXI (iShares China Large Cap ETF) — Rotational EM**
- Structure: **Iron condor** (IVP > 50) — backtest shows IC outperforms short put + kicker on FXI
- **Backtest result:** IC at 51.8% win, +$71 total. VRP is **negative** (-0.019) — the edge is structural diversification, not premium richness.
- Why FXI: **Lowest correlation to SPY in the entire directional block (0.30–0.55).** Best single-country diversifier. China's economy is driven by CCP policy, domestic consumption, and tech regulation — largely independent of US macro.
- IV: 25–45%. Rich premium — reflects event risk.
- Options volume: ~24,000/day — liquid enough for 1-lot positions.
- Risk: Concentrated single-country bet. CCP policy announcements create overnight gaps. Defined risk (IC) is essential — never naked short on FXI.
- Role: **Rotational.** Add 1 lot when FXI IVP > 50 and China macro is constructive. Trade small — this is for diversification, not income.

**EWZ (iShares MSCI Brazil ETF) — Rotational EM**
- Structure: **Iron condor** (IVP > 50) — backtest shows IC is the best structure (+$155 total vs +$80 for strangle)
- **Backtest result:** IC at 56.7% win rate, +$155 total. VRP +0.013. IVP filter adds +5% win rate — most impactful filter in the dataset.
- Why EWZ: Commodity-linked EM — driven by iron ore, soybeans, oil prices + domestic Brazilian policy. Correlation to SPY ~0.40–0.60. Adds a return stream driven by LatAm cycles.
- IV: 30–50%. **Richest premium of any non-leveraged country ETF.** Reflects political + FX risk.
- Options volume: ~27,000/day — liquid, tradeable.
- Risk: Brazilian politics create event risk (elections, fiscal policy shifts). BRL adds FX volatility. IC defined risk caps the damage.
- Role: **Rotational.** Add 1 lot when EWZ IVP > 50 and commodity backdrop is supportive. If both FXI and EWZ qualify, pick EWZ — higher VRP and better backtest performance.

**EM Rotation Rule:** EEM is always on. FXI and EWZ are rotational — add when IVP > 50, remove when IVP < 30 or regime deteriorates. Max 2 EM underlyings active at once (EEM + one rotational) to avoid over-concentrating in emerging markets.

---

### Neutral Block — Commodities, Precious Metals & Bonds (Range-Bound VRP)

> These underlyings do NOT have structural positive drift. The edge comes purely from the variance risk premium — implied vol exceeds realized vol — harvested from BOTH sides via strangles or iron condors. Directional structures (short put + kicker) are wrong here because there is no drift tailwind. *(Cross-asset VRP; Erb & Harvey 2013; Hedging pressure: Cheng et al. 2021)*

**GLD (SPDR Gold Shares)**
- Structure: **Regime-dependent.**
  - Calm market (VIX < 25): Strangle — sell 25Δ put + sell 25Δ call. Gold drifts sideways.
  - Crisis (VIX > 25): **Short put only** — drop the call side. Gold rallies during equity crashes. Being short gold calls in a crisis = unlimited loss on the one asset that's supposed to diversify your book.
  - Gold spike (GLD >10% above 20d SMA): Short call or call spread only — mean reversion play after a gold rally. *(Erb & Harvey 2024: elevated real gold prices predict poor forward returns)*
- IV: 15–25%. Moderate premium.
- Correlation to SPY: –0.05 to +0.15. **Best crisis diversifier in the portfolio.** When equities crash, GLD positions gain or hold while the directional block loses.
- Note: GLD strangles during calm markets act as **income-generating hedges** — they earn theta while providing structural diversification. More capital-efficient than buying VIX calls.

**WEAT (Teucrium Wheat Fund)**
- Structure: Iron condor or strangle (IVP > 50)
- **Backtest validated:** IC at 76.8% win rate, +$170 total. VRP +0.130 — **second highest VRP tested.**
- Why it works: Wheat is strongly seasonal and range-bound between crop cycles. IV massively overestimates actual moves. Producer hedging inflates put prices — same mechanism as oil but even stronger for agriculture.
- IV: 25–50%. Very rich relative to underlying movement.
- Correlation to SPY: Near zero. Driven by weather, crop reports, and seasonal cycles.
- **Liquidity warning:** ~2,000 options contracts/day — thin. Use limit orders, 1 lot only. If bid-ask > $0.20 on target strikes, skip.

**PDBC (Invesco Optimum Yield Diversified Commodity, No K-1)**
- Structure: Iron condor (IVP > 50)
- **Backtest validated:** IC at 79.1% win rate (filtered) — **highest win rate of any IC tested.** VRP +0.243 — **highest VRP in the entire study.**
- Why it works: Diversified 14-commodity basket (energy, metals, agriculture) with an optimized futures roll. The roll strategy reduces realized vol below what IV implies, widening the VRP artificially. You capture the gap.
- IV: 15–25%. Modest per-contract premium, but the win rate compensates.
- Correlation to SPY: +0.20 to +0.40. Partially correlated through energy component.
- Tax advantage: No K-1 filing — cleaner for tax reporting than DBC.
- **Liquidity warning:** ~1,000–2,000 options contracts/day. Limit orders only, 1 lot.

**DBA (Invesco DB Agriculture Fund)**
- Structure: Strangle or iron condor
- **Backtest validated:** Strangle at 62.7% win rate — **highest strangle win rate of all underlyings.** VRP +0.032.
- Why it works: Agriculture basket (corn, soybeans, wheat, sugar, cocoa, coffee, cattle, hogs). Seasonal + producer hedging creates the widest IV-RV gap. The basket diversification dampens single-commodity spikes that blow up strangles.
- IV: 15–30%. Moderate, but the range stability makes strangles highly reliable.
- Correlation to SPY: Near zero. Driven by weather, crop cycles, and food demand.
- **Liquidity warning:** ~1,000–3,000 options contracts/day. 1 lot only.

**BNO (United States Brent Oil Fund) — Optional**
- Structure: Iron condor (IVP > 50)
- **Backtest validated:** IC at 59.8% win rate, +$66 total. VRP +0.029.
- Why BNO: Adds Brent crude alongside USO (WTI). Different supply dynamics — Brent is driven by OPEC+, WTI by US shale. Modest diversification benefit within energy.
- Role: Only add if you want second oil exposure. Not essential — USO covers the energy VRP adequately.

**SLV (iShares Silver Trust) — Deprioritized**
- Structure: Iron condor only
- **Backtest result: 45.0% win rate, +$30 total.** Whipsaw behavior eats the premium. Stop rate 65.8%.
- IV: 25–45%. Premium is rich but silver moves too violently to keep.
- Correlation to SPY: +0.15 to +0.35.
- Role: **Lower priority than GLD, WEAT, PDBC.** Only add if all higher-priority underlyings are deployed and you have remaining BP.

**TLT (iShares 20+ Year Treasury Bond ETF)**
- Structure: Strangle (symmetric 25Δ/25Δ) or iron condor
- Why TLT: **Cleanest strangle candidate.** Range-bound ($80–$110 since 2022), moderate IV, no structural drift in either direction. Both sides collect premium reliably.
- IV: 15–25%. Lower than equities but range stability compensates.
- Correlation to SPY: –0.30 to +0.40. **Negative in normal regimes (adds diversification), positive during Fed tightening (adds risk).** *(Bridgewater 2022: stock/bond correlation flips during aggressive tightening)*
- **Regime filter — critical:** Only trade TLT strangles when: (1) 20d realized vol < 20%, AND (2) no FOMC rate decision within 14 days of expiry, AND (3) TLT between $85–$105 (mid-range). During Fed pivots (tightening → easing or vice versa), TLT makes large directional moves that break the range. Step aside.

**USO (United States Oil Fund) / CL Futures**
- Structure: Asymmetric strangle (sell 25Δ put + sell 15-20Δ call) or put-heavy iron condor
- Why asymmetric: Oil rallies on supply shocks (war, OPEC cuts) are violent and gap overnight. The call side faces blow-up risk. Widen calls to 15-20Δ or use call spreads. The put side benefits from **producer hedging pressure** — commercial producers buy puts to protect physical inventory, inflating put prices. *(Cheng, Tang & Yan 2021: liquidity-providing strategy earns 6.4%/month before costs)*
- IV: 30–50%. **Highest IV of the non-leveraged group = richest premium.** But gap risk is real.
- Range: CL $60–$85 over the last 2 years.
- At $50k: Use USO options (1–2 lots). USO ~$65/share, $6,500 notional per lot. Graduate to CL futures options at $100k+ (CL = ~$65,000 notional per lot — too large at $50k).
- **Weekend risk:** Oil gaps on geopolitical events. Never hold naked short calls on oil over weekends with active geopolitical risk.

**UNG (United States Natural Gas Fund) / NG Futures Options**
- Structure: **Iron condor only.** No naked shorts — ever.
- Why defined risk only: NG has the highest IV (40–80%) and the most extreme gap behavior of any liquid market. NG can move 10-20% in a single session on weather or storage data. Naked strangles on NG are portfolio-ending trades.
- IV: 40–80%. **Richest premium of any liquid market.** The width between implied and realized vol is massive — but so are the tails.
- **Backtest validated:** IC at 72.2% win rate, +$4,162 total P&L — **strongest edge of all underlyings tested.** The iron condor captures the massive VRP (+0.019) while the wings cap the gap risk.
- Range: NG trades in observable multi-year ranges ($1.50–$4.50 in recent years) driven by seasonality, weather, and storage.
- Correlation to SPY: –0.05 to +0.10. **Near-zero equity correlation.** Truly independent return stream.
- Seasonality: Winter demand peaks (Nov–Feb) drive IV expansion. Summer is calmer. Premium is richest in Oct–Nov (heading into uncertainty).
- At $50k: Use UNG options (1–2 lots). Graduate to NG futures options at $100k+.

---

### Portfolio Composition — Two Blocks

| Block | Underlyings | % of DRIFT BP | Delta Profile | Purpose |
|-------|------------|--------------|--------------|---------|
| **Directional** | XSP, IWM, TQQQ, ESTX50, EEM, FXI/EWZ (rotational) | 50% | Long delta | Harvest VRP + equity drift |
| **Neutral** | UNG, USO, GLD, WEAT, PDBC, DBA, TLT, BNO, SLV | 50% | Neutral delta | Harvest VRP from both sides + crisis diversification |

**Neutral block allocation priority (by backtest performance):**

| Priority | Underlying | Win% | VRP | Liquidity | Role |
|----------|-----------|------|-----|-----------|------|
| 1 | UNG | 72.2% | +0.019 | Good (ETF) | Core — strongest edge |
| 2 | USO | 58.4% | +0.017 | Excellent | Core — oil VRP + hedging pressure |
| 3 | GLD | 73.7% | +0.009 | Excellent | Core — crisis diversifier |
| 4 | WEAT | 76.8% | +0.130 | Thin (~2k/day) | Core — highest commodity VRP |
| 5 | PDBC | 79.1% | +0.243 | Thin (~1-2k/day) | Core — highest win rate, broad basket |
| 6 | DBA | 62.7% | +0.032 | Thin (~1-3k/day) | Core — best strangle candidate |
| 7 | TLT | 58.8% | +0.002 | Excellent | Selective — Fed regime filter required |
| 8 | BNO | 59.8% | +0.029 | Moderate | Optional — Brent diversification |
| 9 | SLV | 45.0% | +0.005 | Excellent | Low priority — whipsaw kills edge |

Deploy from top to bottom as BP allows. At $50k, run 4–6 neutral underlyings simultaneously. At $100k+, expand to 7–8.

**Geographic diversification within the directional block:**

| Region | Underlyings | Correlation to SPY | Target BP Share |
|--------|-----------|-------------------|-----------------|
| US | XSP, IWM, TQQQ | 0.85–1.00 | 40% of directional |
| Europe | ESTX50 | 0.65 | 20% of directional |
| Emerging Markets | EEM + FXI or EWZ | 0.30–0.70 | 40% of directional |

**Why the 50/50 split (revised from 60/40):** The backtest shows the neutral block generates stronger risk-adjusted returns than expected. Agriculture (WEAT, DBA, PDBC) and energy (UNG, USO) have VRPs 5–10x larger than equity indices. The neutral block earns comparable theta while providing crisis diversification — it should carry equal weight.

**Why this matters in a crash:** The directional block loses. The neutral block — GLD (negative equity correlation), UNG and WEAT (zero equity correlation) — holds or gains. The neutral positions earn theta income while acting as structural hedges. This is more capital-efficient than paying 1–2%/year for tail protection via VIX calls.

The EM rotation (EEM always-on + one of FXI/EWZ when IVP > 50) adds the lowest-correlation equity exposure in the directional block.

**Vol direction by asset class — affects call side sizing:**

| Asset | Price Up = Vol... | Price Down = Vol... | Implication for Call Side |
|-------|-------------------|---------------------|--------------------------|
| Equity indices | Contracts | Expands | Short calls are safer on rallies — vol drop helps |
| Gold, Silver, Oil, NG | **Expands** | Contracts | Short calls face vol expansion on rallies — widen strike or use spreads |
| Bonds (TLT) | Neutral | Neutral | Symmetric — vol moves with rate uncertainty, not direction |

### DTE Range

**30–360 DTE.** The sweet spot is 45–60 DTE for theta decay, but longer-dated positions (90–360 DTE) are valid for synthetics and PMCC structures. Management is weekly, not daily.

### Pre-Trade Checklist

- [ ] IVP ≥ 50 or IVR ≥ 30 (premium selling edge exists)
- [ ] Underlying above 200d SMA (or use spreads-only if below)
- [ ] VX futures in contango (buy-the-dip environment)
- [ ] IV > HV (implied > historical vol = premium is rich)
- [ ] No FOMC / CPI within 7 days of expiry
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

**Short Put + Kicker Call** (bullish — used on drift assets)
- Sell 20–30Δ put + buy 20Δ call (or 30/20Δ call spread) at same expiry
- Net credit or small debit. Premium income + uncapped (or spread-capped) upside if index runs.
- DTE: 45–360 days depending on conviction
- Use only when underlying clearly above 50d and 200d SMA
- Stop: close short put at 50Δ. Roll to longer DTE and scale per drawdown framework.
- Exit kicker: let call ride — free lottery ticket on recovery after a drawdown roll

**PMCC — Poor Man's Covered Call** (income on existing long calls)
- Buy: deep ITM call ≥120 DTE at 70–80Δ
- Sell: OTM call 7–45 DTE at 20–40Δ against it
- Roll short leg before it reaches 21 DTE; do not let long call fall below 45 DTE

### Delta Band

Entry: 20–30Δ → Hold: 5–45Δ → Profit close: <10Δ → Hard stop: 50Δ

### Position Management (weekly review)

| Situation | Action |
|-----------|--------|
| Reaches 50% profit | Close at next weekly review |
| Short leg hits 50Δ | Close immediately — hard stop, no exceptions |
| Underlying structurally broken | Take the loss — do not roll a loser |
| Premise has changed | Close — never adapt a position to justify staying in it |

### Portfolio-Level Limits

- Per trade max loss: **2% of account**
- Total BP deployed: **≤ 50%** (buffer for volatility spikes)
- Single underlying: **≤ 30%** of BP
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

### Drawdown Scaling — Increasing Allocation on Market Downmoves

> When indices fall, premium gets richer and the structural drift tailwind strengthens for the eventual recovery. Scale BP allocation from 30% up to 80% as drawdown deepens — but only when VIX confirms fear is elevated.

**Scaling Tiers**

Both conditions must be met to advance a tier. Drawdown alone (slow grind with low VIX) does not justify scaling. VIX alone (flash spike with shallow drawdown) does not justify scaling.

| Tier | SPY Drawdown | VIX Range | DRIFT BP | Structure | DTE |
|------|-------------|-----------|----------|-----------|-----|
| Normal | 0% to –5% | < 20 | 30% | XYZ 111, short puts | 45–60 |
| Elevated | –5% to –10% | 20–30 | 40% | XYZ 111, short puts | 45–90 |
| Correction | –10% to –20% | 25–40 | 55% | XYZ 221, short puts, synthetics | 60–120 |
| Deep Correction | –20% to –30% | 35–55 | 70% | XYZ 221, synthetics, LEAPS puts | 90–180 |
| Bear / Capitulation | > –30% | > 50 | 80% | Spreads only, wide strikes, LEAPS | 180–360 |

**Historical basis:** S&P 500 drawdowns over 50 years (1975–2025).

| Event | Drawdown | VIX Peak | Tier | Recovery Time |
|-------|----------|----------|------|---------------|
| 1987 Crash | –34% | 150+ | Bear | ~20 months |
| 1990 Gulf War | –20% | 36 | Deep Correction | ~4 months |
| 2000–02 Dot-com | –49% | 45 | Bear | ~56 months |
| 2007–09 GFC | –57% | 80 | Bear | ~49 months |
| 2011 Debt Ceiling | –19% | 48 | Correction | ~4 months |
| 2018 Q4 | –20% | 36 | Deep Correction | ~4 months |
| 2020 COVID | –34% | 82 | Bear | ~5 months |
| 2022 Bear | –25% | 37 | Deep Correction | ~14 months |

Corrections (–10% to –20%) recover in 3–7 months — well within 60–180 DTE. Bear markets (–30%+) can take years — hence LEAPS only at that tier.

**Scaling Rules**

1. **Scale IN weekly, not all at once.** When a new tier triggers, deploy additional BP over 2–4 weeks. No catching falling knives on a single day.
2. **Widen DTE as you scale up.** Normal: 45–60. Correction: 60–120. Deep/Bear: 90–360. Longer DTE gives the position time to survive extended drawdowns.
3. **Spreads only below 200d SMA.** At Deep Correction and Bear tiers, no naked short puts regardless of premium attractiveness.
4. **Scale DOWN faster than you scale up.** When VIX drops below the tier threshold, reduce allocation at the next weekly review. Premium shrinks faster than price recovers.
5. **Bear tier = LEAPS only.** At >–30% with VIX >50, only 180–360 DTE positions. Wider strikes, fewer contracts. The 2% max loss per trade still applies.
6. **50Δ hard stop is never suspended.** Scaling determines how much BP to deploy — not whether to hold losers. A position that hits 50Δ is closed at any tier.

### Portfolio Hedging — Defense for the DRIFT Book

> Scaling is offense — deploying more capital into fear. Hedging is defense — limiting the damage when fear is justified. Both use VIX as the trigger, but they serve opposite purposes.

**Why hedge a short-vol book?**
The DRIFT portfolio is structurally short gamma. In normal regimes, this earns premium. In tail events, losses are non-linear — volatility itself jumps (Eraker et al., 2003), correlations spike to 1 (Catao & Timmermann, 2007), and the 50% BP buffer can evaporate in days. The 2024-08 and 2025-04 events demonstrated this. Hedging is not optional — it's the cost of staying in the game.

**When to hedge**

| Regime | VIX | Hedge Action |
|--------|-----|-------------|
| Calm (VIX < 15) | Cheapest protection available | **Buy tail hedges.** VIX calls and far-OTM puts are at annual lows. Spend 0.5–1% of account per quarter on protection. |
| Normal (VIX 15–20) | Protection moderately priced | **Maintain existing hedges.** Roll expiring protection. No new purchases unless approaching macro events. |
| Elevated (VIX 20–30) | Protection getting expensive | **Stop buying new hedges.** Existing hedges are gaining value. Let them work. Focus on reducing position size instead. |
| Stress (VIX > 30) | Protection is expensive — too late to buy | **Monetize hedges.** Sell profitable VIX calls/puts into the spike. Use proceeds to offset DRIFT losses. |

**Key insight:** Buy protection when you don't need it. By the time you need it, it's too expensive to be effective.

**Hedge Structures**

| Structure | Cost | When to Use | Sizing |
|-----------|------|-------------|--------|
| **Far-OTM SPX puts (5–10Δ)** | Low | Calm regime. Buy 90–120 DTE. Cheap crash insurance. | 0.25–0.5% of account per quarter |
| **VIX call spreads (e.g., 20/35)** | Low–Medium | Calm regime. Buy 60–90 DTE. Profits if VIX spikes >20. | 0.25–0.5% of account per quarter |
| **Risk-reversal overlay** | Near-zero | Any regime. Sell OTM put + buy OTM call on same index. Long call leg provides upside offset. | Built into DRIFT structure — no extra cost |
| **Reduce gross exposure** | Free | Elevated+ regime. The cheapest hedge is a smaller book. | Per regime adjustment table above |

**Cost Budget**

Spend **1–2% of account per year** on tail protection. This is the insurance premium for running a short-vol book. If hedges expire worthless, that's a good year — the cost is offset many times over by DRIFT premium collected.

| Item | Annual Budget |
|------|--------------|
| Far-OTM SPX puts (4 quarterly rolls) | ~0.5–1.0% |
| VIX call spreads (4 quarterly rolls) | ~0.5–1.0% |
| **Total hedge cost** | **~1–2%** |
| DRIFT target theta income | ~0.4%/day × 252 days = ~100%+ of deployed BP |
| **Net: hedge cost is <2% of gross theta income** | |

**Hedge Management Rules**

1. **Buy in calm, monetize in stress.** Never buy VIX calls when VIX > 25 — the premium is already reflecting the fear. Sell existing hedges into spikes above 30.
2. **Roll quarterly.** Tail hedges lose value to theta. Roll 90 DTE positions at 30 DTE remaining — don't let them decay to zero.
3. **Never hedge with the same structure you're selling.** Buying puts while selling puts creates a wash. Use VIX calls or far-OTM strikes that don't overlap with your DRIFT positions.
4. **Gold is a passive hedge.** Your 10–20% LONG-TERM gold allocation already provides crash protection (Baur & Lucey, 2010: gold performs best in the 2–4 weeks following equity crashes). Don't duplicate with additional gold positions in DRIFT.
5. **Fed tightening = correlation hedge breaks.** During aggressive tightening (Bridgewater, 2022), stocks and bonds fall together. Your LONG-TERM diversification across geographies and gold is the only hedge that works in this regime. Reduce DRIFT size, don't try to hedge with bonds.

**The LTCM reminder:** Long-Term Capital Management ran a short-vol book with 25:1 leverage and no tail hedges. The 1998 Russian crisis produced losses that models said were impossible. Your 50% BP cap, 2% max loss per trade, and hedge budget exist because of this lesson. The hedge is not about expected value — it's about survival.

---

## 03 — Range-Bound Commodities (Opportunistic)

> NG, CL (and other commodities) trade in observable multi-year ranges based on political factors, weather, and seasonality. Not a systematic strategy — trade only when the setup is clear.

**Setup:** Enter when price reaches the top/bottom of the last reversal and shows a clear indication of reversal (wick, towers, double top/bottom).

**Structure:**
- Long premium if the underlying moves in the direction of increasing volatility (commodities: upside = vol expansion)
- Short premium if the underlying moves in the direction of vol contraction

**Research gap:** The trading ranges need to be quantified. Chart analysis of daily behavior over the last 10 years suggests consistent patterns, but this is not yet backed by systematic data.

---

## 04 — Weekly Review

Run every weekend. This is the only scheduled review cadence for the investing portfolio.

### Market Context

| Check | What to Look For |
|-------|-----------------|
| VIX / VX | Level, direction, contango (constructive) vs backwardation (caution) |
| SPY vs RSP | Cap-weighted vs equal-weight — divergence = narrowing breadth |
| Advances / Declines | With 10 & 20 EMA — breadth trend |
| Commodities | GC, CL, NG, ZN, ZB — range positions, uncorrelated signals |
| Global | Nikkei, ESTX, EEM, FXI — international picture |
| Regime ratios | XLY/XLP (sentiment) · XLK/XLF (growth) · QQQ/SPY (risk) |

### Portfolio Actions

1. IVP scan across DRIFT underlyings — which has elevated premium this week?
2. Review open positions: any at 50% profit? Any approaching 21 DTE? Any at 50Δ stop?
3. Commodity range check — GC, CL, NG approaching extremes?
4. LONG-TERM: any scheduled contribution this month? → run contribution checklist
5. Log position changes and update trade journal

---

## 05 — Non-Negotiable Rules

1. **Never sell LONG-TERM holdings** — redirect contributions; never realise taxable gains
2. **BP ≤ 50% at all times** — the buffer is the risk management system
3. **2% max loss per DRIFT trade** — calculate position size before every entry
4. **200% stop on short premium** — close immediately, no exceptions
5. **50Δ delta hard stop** — when short leg hits 50Δ, close it now
6. **IVP ≥ 50 / IVR ≥ 30** — if premium is cheap, wait; edge is gone below these levels
7. **Never adapt a failing trade** — close it; re-enter only on a fresh setup
8. **Stay continuously invested in DRIFT** — theta only works when deployed
9. **Indices and commodity ETFs only** — no individual equities in DRIFT
10. **Log every trade** — review weekly for systematic errors and rule violations

---

## 06 — Research-Backed Edges

Actionable findings from 161 academic papers and 114 books, filtered for the investing portfolio. Each entry is grounded in peer-reviewed research or large-sample empirical data.

### LONG-TERM Portfolio — Allocation & Timing

**Turn-of-month effect (McConnell & Xu, 2008; Ogden, 1990)**
Index returns are concentrated in the last trading day of the month through the first three days of the next month — driven by pension fund inflows. Schedule ETF contributions for the last day of the month to ride this structural flow.

**Quarter-end rebalancing headwind (Harvey et al., 2025)**
Institutional rebalancing creates a ~17bps daily headwind in the final 3 days of each quarter. Avoid buying LONG-TERM holdings in the last 3 trading days of Q1/Q2/Q3/Q4. Buy in the first week of the new quarter when forced selling ends and recovery begins.

**Small cap value tilt (Banz 1981; Fama-French 1993, 2015; Felix 2020)**
The size + value premium adds 1–3% expected annual return over market-cap weighting. Your AVUV/AVEE allocation captures this. The premium is unreliable over 1–3 years but robust over 10+. No action needed — just hold.

**Quality factor overlay (Frazzini, Kabiller & Pedersen, 2018)**
Buffett's alpha is explained by quality + value + leverage. Your portfolio captures value via SCV but lacks an explicit quality tilt. Consider adding a quality ETF (QUAL, JQUA) at 5–10% as the portfolio grows — this is a future expansion, not immediate.

**Gold allocation discipline (Erb & Harvey, 2013)**
Gold is an unreliable short-term inflation hedge but mean-reverts over long horizons. Real gold prices above historical average predict lower future returns. Your 10–20% band is correct — add to equities when gold exceeds 20% of portfolio, especially after real price spikes.

**Passive flow distortions (Brightman & Harvey, 2025)**
Index-fund dominance increases co-movement within indices and inflates cap-weighted overvaluation. Your LONG-TERM portfolio is inherently exposed to this. Mitigant: the SCV and international tilts provide some insulation since these are underweighted by passive flows.

### DRIFT Portfolio — Premium Selling Edges

**Variance risk premium is structural (Carr & Wu, 2005, 2009)**
Implied volatility systematically exceeds realized volatility on indices. This is the foundational justification for your entire DRIFT strategy. The premium is larger for indices than for individual stocks — confirming your "indices only" rule.

**VIX-conditional entry (de Saint-Cyr, 2023; tastylive research)**
Iron condor and short put win rates vary dramatically with VIX level at entry. Selling premium when VIX > 20 with IVP > 50 has the highest probability of profit. Your pre-trade checklist already captures this. The research confirms: never sell premium when VIX < 15 — the premium collected doesn't compensate for the risk.

**Weekend theta mispricing (Jones & Shemesh, 2017)**
Options are overpriced over weekends because variance is allocated to calendar days, not trading days. Short premium positions benefit from excess weekend decay. Timing: enter new short vol positions on Thursday/Friday to capture the weekend theta edge. Over a 45–60 DTE position, this accumulates 8–9 weekends of excess decay.

**Dealer gamma as regime signal (SqueezeMetrics; Pearson et al., 2007)**
When dealers are net long gamma (high GEX), their hedging suppresses realized vol — ideal for selling premium. When GEX flips negative, realized vol exceeds implied — reduce short vol exposure. Add GEX to your weekly review as a regime filter alongside VIX/VX.

**Skew premium is the same bet as variance premium (Kozhan et al., 2013)**
Selling puts and selling straddles capture the same underlying risk factor (~0.9 correlation). You cannot diversify between them. Treat all short-vol positions as one risk bucket in portfolio construction. Your 50% BP cap applies to the total, not per structure.

**IV term structure slope (Vasquez, 2017; Bennett, 2014)**
When 90-day IV >> 30-day IV (steep slope), longer-dated puts are overpriced — sell them. When the curve is flat or inverted, near-term risk is elevated — reduce exposure or widen strikes. Add term structure check to weekly review.

**Retail as structural counter-party (Hu et al., 2024; Barber & Odean)**
66% of retail options traders use simple one-sided bets and lose money. They overpay for short-dated OTM options, especially before earnings. You are the other side of this trade. The finding that "volatility trading (selling premium) earns the highest Sharpe ratio" directly validates your approach.

**Vol mean reversion after extremes (He, 2013)**
After extreme market moves, volatility reliably reverts to prior levels. This supports your drawdown scaling framework — selling premium after vol spikes is statistically the highest-EV entry. The reversion occurs in level, skew, AND kurtosis simultaneously.

### Future Profit Mechanisms (Research Phase)

These effects are documented in the research but not yet part of your active system. Evaluate gradually — one per quarter.

| Mechanism | Source | Potential Expression | Status |
|-----------|--------|---------------------|--------|
| Risk-reversal premium | Hull & Sinclair 2021 | Sell OTM put + buy OTM call on indices — captures skew mispricing | Research: validate on XSP/SPY |
| VIX futures roll yield | Cooper 2013; S&P methodology | Short front-month VIX futures during contango — harvest roll yield | Research: sizing and tail risk |
| Commodity VRP | Cheng/Tang/Yan 2021; cross-asset VRP | Short OTM puts on CL/GC at 20-30Δ; put spreads on NG/ZS/ZC — hedging pressure inflates puts | Research: backtest structures per commodity |
| Bond VRP | Cross-asset VRP (multiple) | Short strangles or iron condors on ZN/ZB during stable rate regimes | Research: regime-dependency, Fed pivot risk |
| Currency carry + VRP | Koijen et al. 2018; Asness et al. 2013 | Short strangles on 6E/6J when RV is low and term structure in contango | Research: backtest, liquidity, bid-ask |
| Managed volatility overlay | Cooper 2010 | Scale all positions inversely to 20d realized vol — improves Sharpe | Partially implemented in regime adjustments |
| Regime-conditional factor timing | Harvey/Man Group 2025 | Use macro regime detection to shift between momentum and mean-reversion | Research: define macro variables |
| E/P timing signal | Shen 2002 | S&P 500 E/P minus T-bill rate predicts drawdown probability | Research: add to weekly review as macro overlay |

### Commodity VRP — Structure Notes

Commodity options differ from equity index options in important ways:

| Factor | Equity Indices | Commodities (CL, NG, ZS, ZC) |
|--------|---------------|-------------------------------|
| Vol direction | Upside = vol contraction | Upside = vol expansion |
| Put pricing | Overpriced (hedging demand) | Overpriced (producer hedging) |
| Call pricing | Fairly priced | Underpriced (vol expansion on rallies) |
| Best short vol structure | Short puts, XYZ | Short puts at range bottom; strangles on CL/GC only |
| Avoid | — | Naked strangles on NG (vol too spikey) |

**Hedging pressure (Cheng et al., 2021):** Commercial producers buy puts to protect physical inventory. This creates structural put overpricing on commodities — the same mechanism as equity index put overpricing, but driven by hedgers not portfolio insurers. Selling puts opposite to hedger flow earns 6.4%/month before costs.

**Key constraint:** Commodity futures options have lower liquidity and wider bid-ask spreads than SPX. Use defined-risk structures (put spreads, iron condors) on thinner markets (NG, ZS, ZC). Naked puts acceptable on CL and GC where liquidity is deeper.

### Bond & Currency VRP — Research Questions

These asset classes have documented variance risk premiums but no dedicated backtest in the current research library. Answer these before deploying capital:

**Bonds (ZN, ZB options):**
1. What is the average IV-RV spread on ZN/ZB over the last 10 years?
2. How does the VRP behave during Fed pivot points (rate hikes → pauses → cuts)?
3. What is the optimal DTE and delta for short strangles on ZN?
4. Stock/bond correlation regime: when does selling bond vol add diversification vs. doubling the same risk?

**Currencies (6E, 6J options):**
1. What is the average IV-RV spread on 6E/6J?
2. Does the carry trade (long high-yield, short low-yield) combine with or offset short vol on the same pairs?
3. What is the liquidity and bid-ask on 6E/6J options at 45-60 DTE and 20-30Δ?
4. Are there seasonal patterns in FX vol (year-end, fiscal year-end for JPY)?

---

## Open Research / TBD

### Active Research
- **Commodity VRP backtest:** Backtest short put strategies on CL, NG, ZS, ZC futures options across multiple cycles. Key question: what structure per commodity, what DTE, what delta?
- **Bond VRP feasibility:** Measure IV-RV spread on ZN/ZB. Test short strangles during stable rate regimes. Identify regime filter for Fed pivot risk.
- **Currency VRP feasibility:** Measure IV-RV spread on 6E/6J. Test short strangles at 45-60 DTE. Assess liquidity and bid-ask costs.
- **Hedge calibration:** Backtest the 1–2% annual hedge budget against 2018 Q4, 2020 COVID, 2022 bear, 2024-08, 2025-04 drawdowns. Does the hedge payoff offset DRIFT losses in each event?

### Validation Needed
- **Commodity range quantification:** The RBC hypothesis needs systematic backtesting — ranges are observed but not yet measured.
- **Seasonality exploitation:** July is historically the most positive month — is there a systematic trade around this?
- **Turn-of-month contribution timing:** Validate the pension fund flow effect on your specific ETFs before formalizing.

### Integration Tasks
- **GEX integration:** Add dealer gamma exposure to weekly review. Source: SqueezeMetrics or equivalent.
- **Term structure monitoring:** Build a weekly check of 30d vs 90d IV slope across DRIFT underlyings.
- **Cross-asset VRP dashboard:** Track IV-RV spreads across all DRIFT underlyings + commodities + bonds + currencies in one view.
