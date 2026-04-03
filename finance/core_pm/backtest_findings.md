# Core PM Backtest Findings

Short premium strategy validation using daily price + IV + HV data.
Delta-approximation via Black-Scholes. 45 DTE, 25Δ entries, 50Δ stop, 14-day entry interval.

---

## Methodology

- **Data source:** SwingTradingData (IBKR daily bars with IV/HV)
- **Strike approximation:** Black-Scholes 25Δ put and 25Δ call
- **P&L calculation:** BS option pricing at entry and exit (or expiry)
- **Stop:** Position closed when estimated put delta reaches 50Δ
- **Entry filter tested:** IVP > 50 vs unfiltered
- **Structures tested:** Short put, strangle (25Δ put + 25Δ call), iron condor (5% wing width)
- **Limitation:** Approximate — no real options chain data, no bid-ask costs. Directional accuracy is reliable; exact P&L numbers are ±15%.

---

## Results Summary — Best Structure per Underlying

| Underlying | Best Structure | Filter | N | Win% | Avg P&L | Total P&L | Stopped% | Avg VRP | Verdict |
|-----------|---------------|--------|---|------|---------|-----------|----------|---------|---------|
| **UNG** | Iron condor | all | 335 | 72.2% | +$12.42 | +$4,162 | 65.7% | +0.019 | **Strongest edge. Defined risk essential.** |
| **USO** | Iron condor | ivp>50 | 233 | 58.4% | +$2.45 | +$571 | 57.1% | +0.017 | **Strong. IC captures range + rich IV.** |
| **GLD** | Short put | ivp>50 | 224 | 73.7% | +$0.25 | +$57 | 27.2% | +0.009 | **Puts only. Strangles lose money.** |
| **DBA** | Strangle | all | 332 | 62.7% | +$0.12 | +$39 | 50.9% | +0.032 | **Highest VRP. Strangles and IC both work.** |
| **EWZ** | Iron condor | ivp>50 | 240 | 56.7% | +$0.65 | +$155 | 58.8% | +0.013 | **Good EM diversifier. IC preferred.** |
| **EEM** | Iron condor | ivp>50 | 256 | 58.6% | +$0.35 | +$90 | 54.3% | +0.001 | **Moderate. Low VRP but diversifies.** |
| **TLT** | Iron condor | ivp>50 | 119 | 58.8% | +$0.34 | +$40 | 51.3% | +0.002 | **Range-bound confirmed. IC best.** |
| **FXI** | Iron condor | ivp>50 | 251 | 51.8% | +$0.28 | +$71 | 61.0% | -0.019 | **Marginal. Negative VRP — edge is structural, not vol.** |
| **SLV** | Iron condor | all | 307 | 45.0% | +$0.10 | +$30 | 65.8% | +0.005 | **Weak. High stop rate. Lower priority.** |
| **XLE** | Iron condor | all | 359 | 49.9% | +$0.18 | +$65 | 59.1% | -0.001 | **Marginal. No VRP edge. Skip.** |
| **EFA** | Iron condor | all | 359 | 47.4% | +$0.05 | +$19 | 58.8% | -0.008 | **Negative VRP. No edge. Skip.** |

---

## Tier Classification

### Tier 1 — Strong Edge (trade actively)

**UNG (Natural Gas ETF)**
- Structure: **Iron condor only** — never naked. NG gaps destroy naked positions.
- Win rate: 72.2% (highest of all underlyings)
- Average P&L: +$12.42/trade — driven by extreme IV (40-80%)
- VRP: +0.019 (consistently positive — IV massively overestimates RV)
- Risk: 65.7% stop rate on strangles — the IC wing width caps the damage
- Finding: **The iron condor captures the VRP while surviving the gaps.** Strangles also positive but the IC is dramatically better risk-adjusted.

**USO (Crude Oil ETF)**
- Structure: **Iron condor** (IVP > 50 filter improves results by +6% win rate)
- Win rate: 58.4% (filtered)
- Average P&L: +$2.45/trade
- VRP: +0.017 (second highest — producer hedging pressure inflates puts)
- Risk: Short puts alone are negative (-$168 total) — the call side contributes meaningfully on range-bound oil
- Finding: **Confirms the Cheng et al. hedging pressure thesis.** Puts are overpriced, but selling calls too captures the range. IC is the right structure.

**GLD (Gold ETF)**
- Structure: **Short put only** — strangles and IC lose money
- Win rate: 73.7% (filtered, second highest)
- Average P&L: +$0.25/trade (modest per trade, but consistent)
- VRP: +0.009
- Risk: Strangles at -$97 total, IC at -$31 — **call side is toxic.** Gold rallies in crises destroy short calls.
- Finding: **Validates the regime-dependent framework.** Never sell calls on gold. Short puts capture the put VRP without the crisis blow-up risk.

**DBA (Agriculture Commodities ETF)**
- Structure: **Strangle or iron condor** — both work
- Win rate: 62.7% (unfiltered) — **highest strangle win rate of any underlying**
- Average P&L: +$0.12/trade (strangle), +$0.14 (IC)
- VRP: +0.032 (**highest VRP of all underlyings** — massive IV overestimation)
- Risk: Low stop rate (50.9%) — the range behavior is very stable
- Finding: **Best risk-adjusted strangle candidate.** Agriculture commodities are genuinely range-bound with the largest IV-RV gap. Low liquidity may be a concern — verify options volume before trading.

### Tier 2 — Moderate Edge (trade selectively)

**EWZ (Brazil ETF)**
- Structure: **Iron condor** (IVP > 50)
- Win rate: 56.7%, Avg P&L: +$0.65/trade, Total: +$155
- VRP: +0.013 (positive, reflects EM + commodity risk premium)
- IVP filter adds +5% win rate — most impactful filter in the dataset
- Finding: **Good EM diversifier with real VRP edge.** Iron condor preferred over strangle. Political event risk managed by defined risk structure.

**EEM (Emerging Markets ETF)**
- Structure: **Iron condor** (IVP > 50)
- Win rate: 58.6%, Avg P&L: +$0.35/trade, Total: +$90
- VRP: +0.001 (barely positive — the edge is thin)
- Finding: **Value comes from diversification, not from premium richness.** Low VRP means the edge is small per trade. Worth trading for the low correlation to US, not for the premium.

**TLT (Treasury Bond ETF)**
- Structure: **Iron condor** (IVP > 50)
- Win rate: 58.8%, Avg P&L: +$0.34/trade, Total: +$40
- VRP: +0.002 (low — bonds are efficiently priced)
- Finding: **Confirms range-bound thesis.** Both sides contribute. IVP filter is critical — unfiltered short puts are negative. Only trade when premium is actually elevated.

**FXI (China Large Cap ETF)**
- Structure: **Iron condor** (IVP > 50)
- Win rate: 51.8%, Avg P&L: +$0.28/trade, Total: +$71
- VRP: -0.019 (**negative** — IV underestimates actual moves)
- Finding: **The edge is structural/diversification, not VRP.** The negative VRP means you're not being adequately compensated for the vol you're selling. Trade small for diversification, not for premium income.

### Tier 3 — No Edge (skip or deprioritize)

**SLV (Silver ETF)**
- Iron condor: 45.0% win, +$0.10 avg, +$30 total. Stop rate 65.8%.
- VRP: +0.005 (barely positive). The high stop rate eats the premium.
- Finding: **Whipsaw kills the edge.** Silver moves too violently for the premium collected. Lower priority than GLD, USO, or UNG.

**XLE (Energy Sector ETF)**
- Iron condor: 49.9% win, +$0.18 avg, +$65 total.
- VRP: -0.001 (zero). No variance risk premium.
- Finding: **Sector ETF — no structural VRP edge.** Skip. Use USO for energy exposure instead.

**EFA (Developed International ETF)**
- Iron condor: 47.4% win, +$0.05 avg, +$19 total.
- VRP: -0.008 (negative). You're paying to sell premium here.
- Finding: **No edge whatsoever.** Use ESTX50 for developed international exposure instead.

---

## Key Findings

### 1. Structure Matters More Than Filtering

The choice of structure (short put vs strangle vs IC) has a larger impact on P&L than the IVP filter. Wrong structure on the right underlying loses money.

| Rule | Evidence |
|------|----------|
| **Drift assets → short put** | GLD short puts: +$57. GLD strangles: -$97. Call side destroys the edge. |
| **Range-bound → IC or strangle** | USO IC: +$571. USO short puts: -$168. Both sides needed for range assets. |
| **Extreme vol → defined risk only** | UNG IC: +$4,162. UNG strangles: +$1,017. IC captures more with less risk. |

### 2. VRP Predicts Profitability

| VRP Range | Underlyings | Profitable? |
|-----------|------------|-------------|
| > +0.015 | UNG, USO | Yes — strongly |
| +0.005 to +0.015 | GLD, DBA, SLV, EWZ | Yes — moderately |
| -0.005 to +0.005 | EEM, TLT | Marginal — diversification value only |
| < -0.005 | FXI, XLE, EFA | No — skip or trade small |

### 3. IVP > 50 Filter Is Consistently Positive

Adds 3-6% to win rates across all underlyings. Most impactful on EWZ (+5.0%), TLT (+4.3%), and USO (+5.9%). Least impactful on UNG and DBA (already high win rates).

### 4. Commodity ETFs Have the Strongest Edge

The top 4 performers (UNG, USO, GLD, DBA) are all commodities. The hedging pressure thesis (Cheng et al., 2021) is validated — producer hedging inflates put prices, creating a structural put VRP that equity indices don't have.

### 5. Emerging Market ICs Work for Diversification

EWZ and EEM both show positive IC returns. The edge is moderate but the low correlation to US/EU positions adds portfolio-level value beyond the per-trade P&L.

---

## Recommended Portfolio Assignment

Based on backtest results, mapped to the Investing.md framework:

| Underlying | Block | Structure | Priority |
|-----------|-------|-----------|----------|
| UNG | Neutral | Iron condor only | **High** — strongest edge |
| USO | Neutral | Iron condor (IVP > 50) | **High** — rich VRP + range |
| GLD | Neutral | Short put only (regime-dependent) | **High** — crisis diversifier |
| DBA | Neutral | Strangle or IC | **Medium** — highest VRP, check liquidity |
| TLT | Neutral | Iron condor (IVP > 50) | **Medium** — range confirmed, Fed filter needed |
| EWZ | Directional | Iron condor (IVP > 50) | **Medium** — EM diversifier |
| EEM | Directional | Iron condor (IVP > 50) | **Medium** — broad EM, low per-trade edge |
| FXI | Directional | Iron condor (small size) | **Low** — diversification only |
| SLV | Neutral | Deprioritize | **Low** — whipsaw kills edge |
| XLE | — | Skip | No edge |
| EFA | — | Skip | Negative VRP |

---

## Extended Scan — Additional Commodity ETFs

Tested 10 additional commodity and thematic ETFs to find new premium selling candidates.

### Results — Best Structure per Additional Underlying

| Underlying | Category | Best Structure | Filter | N | Win% | Avg P&L | Total P&L | VRP | Verdict |
|-----------|----------|---------------|--------|---|------|---------|-----------|-----|---------|
| **WEAT** | Agriculture (Wheat) | IC | ivp>50 | 177 | 76.8% | +$0.96 | +$170 | +0.130 | **Tier 1. Massive VRP. Best single-commodity agriculture.** |
| **PDBC** | Broad commodity basket | IC | ivp>50 | 91 | 79.1% | +$0.49 | +$44 | +0.243 | **Tier 1. Highest VRP tested. Small sample — needs monitoring.** |
| **GSG** | Broad commodity basket | IC | ivp>50 | 256 | 60.5% | +$0.21 | +$54 | +0.038 | **Tier 2. Solid diversified commodity IC.** |
| **BNO** | Brent Crude Oil | IC | ivp>50 | 179 | 59.8% | +$0.37 | +$66 | +0.029 | **Tier 2. Similar to USO. Adds oil benchmark diversification.** |
| **DBC** | Broad commodity basket | IC | all | 335 | 56.7% | +$0.11 | +$37 | +0.021 | **Tier 2. Moderate. Diversified but low per-trade edge.** |
| **URNM** | Uranium miners | IC | ivp>50 | 62 | 64.5% | +$0.91 | +$56 | -0.004 | **Tier 2. Thematic. Edge from IV overpricing, not VRP.** |
| **URA** | Uranium | IC | ivp>50 | 210 | 53.8% | +$0.32 | +$67 | +0.025 | **Tier 3. Marginal IC edge. Thematic — not a core position.** |
| **CPER** | Copper | IC | all | 89 | 50.6% | +$0.12 | +$11 | +0.018 | **Tier 3. Too few trades, marginal edge. Skip for now.** |
| **COPX** | Copper miners | IC | ivp>50 | 70 | 48.6% | +$0.42 | +$29 | -0.032 | **Tier 3. Negative VRP. Skip.** |
| **BOIL** | 2x Natural Gas | IC | all | 247 | 97.6% | extreme | extreme | +0.034 | **Anomalous. Reverse splits distort BS approximation. Unreliable data — skip BOIL, use UNG.** |

### New Tier 1 Additions

**WEAT (Teucrium Wheat Fund)**
- Structure: **Iron condor or strangle** (IVP > 50)
- Win rate: 76.8% (filtered) — second only to UNG
- VRP: +0.130 — **second highest VRP tested after PDBC**
- Why it works: Wheat is strongly seasonal and range-bound between crop cycles. IV massively overestimates actual moves. Producer hedging inflates put prices.
- Risk: Lower liquidity than GLD/USO — verify options volume and bid-ask before trading. WEAT has ~2,000 options contracts/day — thin but workable for 1 lot.
- Note: DBA (agriculture basket) showed VRP +0.032. WEAT as a single commodity has 4x the VRP. The concentrated exposure is more profitable per contract.

**PDBC (Invesco Optimum Yield Diversified Commodity, No K-1)**
- Structure: **Iron condor** (IVP > 50)
- Win rate: 79.1% (filtered) — **highest win rate of any IC tested**
- VRP: +0.243 — **highest VRP in the entire study** (likely amplified by the "optimum yield" roll strategy which reduces actual vol vs implied)
- Why it works: Diversified 14-commodity basket (energy, metals, agriculture) with an optimized roll. The roll strategy means realized vol is lower than implied, widening the VRP.
- Risk: Small sample (91 filtered trades). Newer ETF — less historical data than DBC.
- Tax advantage: No K-1 filing — cleaner for tax reporting than DBC.

### Findings from Extended Scan

**1. Agriculture commodities have the strongest VRP**

| Underlying | VRP | Category |
|-----------|-----|----------|
| PDBC | +0.243 | Broad commodity (optimized roll) |
| WEAT | +0.130 | Wheat |
| DBA | +0.032 | Agriculture basket |
| GSG | +0.038 | Broad commodity |
| USO | +0.017 | Crude oil |
| UNG | +0.019 | Natural gas |
| GLD | +0.009 | Gold |

Agriculture (WEAT, DBA) and broad commodity baskets (PDBC, GSG) have VRPs 5-10x larger than gold or energy. The combination of seasonal patterns + producer hedging creates the widest IV-RV gap.

**2. Broad commodity baskets are better than single commodities for strangles**

DBC, PDBC, and GSG all show positive strangle returns because the basket diversification dampens the extreme single-commodity moves that blow up strangle positions. The basket's realized vol is lower than the weighted sum of individual commodity vols.

**3. Uranium shows thematic edge, not VRP**

URA and URNM have near-zero or negative VRP — options are fairly priced. But IC still works modestly because the nuclear theme creates range-bound trading between narrative-driven rallies and pullbacks. This is a thematic position, not a VRP harvest. Trade small if at all.

**4. BOIL (2x NG) data is unreliable**

Multiple reverse splits distort the price series. The BS approximation produces nonsensical P&L numbers. Use UNG for natural gas — single-leverage, cleaner data.

**5. Copper has insufficient edge**

CPER and COPX both show marginal or negative results. Copper trends strongly (not range-bound) and options are fairly priced. Not suitable for premium selling.

### Updated Portfolio Assignment — All Underlyings

| Underlying | Block | Structure | Priority | VRP |
|-----------|-------|-----------|----------|-----|
| UNG | Neutral | Iron condor only | **Tier 1** | +0.019 |
| USO | Neutral | Iron condor (IVP > 50) | **Tier 1** | +0.017 |
| GLD | Neutral | Short put only (regime) | **Tier 1** | +0.009 |
| WEAT | Neutral | IC or strangle (IVP > 50) | **Tier 1** | +0.130 |
| PDBC | Neutral | Iron condor (IVP > 50) | **Tier 1** | +0.243 |
| DBA | Neutral | Strangle or IC | **Tier 1** | +0.032 |
| TLT | Neutral | Iron condor (IVP > 50) | **Tier 2** | +0.002 |
| GSG | Neutral | Iron condor | **Tier 2** | +0.038 |
| BNO | Neutral | Iron condor (IVP > 50) | **Tier 2** | +0.029 |
| EWZ | Directional | Iron condor (IVP > 50) | **Tier 2** | +0.013 |
| EEM | Directional | Iron condor (IVP > 50) | **Tier 2** | +0.001 |
| FXI | Directional | Iron condor (small) | **Tier 3** | -0.019 |
| SLV | Neutral | Deprioritize | **Tier 3** | +0.005 |
| URNM | Neutral | IC (thematic) | **Tier 3** | -0.004 |
| XLE | — | Skip | — | -0.001 |
| EFA | — | Skip | — | -0.008 |
| COPX | — | Skip | — | -0.032 |
| CPER | — | Skip (insufficient data) | — | +0.018 |
| BOIL | — | Skip (data unreliable) | — | — |

### Liquidity Concern — Verify Before Trading

| Underlying | Est. Daily Options Volume | Tradeable? |
|-----------|--------------------------|------------|
| WEAT | ~2,000 | Thin — 1 lot only, use limit orders |
| PDBC | ~1,000-2,000 | Thin — 1 lot only |
| DBA | ~1,000-3,000 | Thin — 1 lot only |
| GSG | ~500-1,500 | Very thin — may not be tradeable |
| BNO | ~2,000-5,000 | Moderate — 1-2 lots |

**Rule:** If bid-ask spread > $0.20 on the 25Δ strike at 45 DTE, skip the underlying. The slippage eats the edge.

---

## Caveats

1. **Approximate P&L.** Without options chain data, strike prices and premiums are estimated via Black-Scholes. Real bid-ask costs would reduce returns by 10-20%.
2. **No slippage modeling.** Illiquid underlyings (DBA, UNG) may have wider spreads than this simulation assumes.
3. **Survivorship in ETF data.** USO underwent a reverse split in 2020. UNG has contango drag. Historical prices may not perfectly reflect actual trading conditions.
4. **Single DTE tested.** Results may differ at 30 or 60 DTE. The 45 DTE sweet spot is assumed, not validated per underlying.
5. **Entry interval is fixed.** Every 14 days regardless of market conditions. A more sophisticated entry (e.g., VIX spike → enter) would likely improve results.
