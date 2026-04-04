# Options Strategy Reference

Consolidated reference for all options structures used across the investing and trading frameworks. Each structure includes: when to use, how to construct, management rules, and research basis.

---

## Structure Selection Framework

> Options are a magnifier, not a shortcut. Read the chain for information first, structure second. Never use options to avoid defining risk. *(Trading_new.md)*

### By Market View

| View | Structures |
|------|-----------|
| **Directional (bullish)** | Short put + kicker call, XYZ, PMCC, debit spread, synthetic long |
| **Neutral (range-bound)** | Strangle, iron condor, iron butterfly, calendar |
| **Volatility (long vol)** | Long straddle, long strangle, back ratio |
| **Hedging** | Far-OTM puts, VIX call spreads, risk reversal overlay |

### By IVR Context

| IVR Range | Preferred Structure | Rationale |
|-----------|-------------------|-----------|
| < 30% | **Buy options** — long calls, debit spreads | Premium is cheap. Gamma working for you. |
| 30–50% | **Debit spreads** — reduce vega exposure | Premium is fair. Spreads reduce cost and vega drag. |
| 50–70% | **Sell premium** — short puts, strangles, XYZ, IC | Premium is rich. VRP is wide. This is your edge. |
| > 70% | **Sell premium, defined risk only** — IC, put spreads | Premium is very rich but tail risk is elevated. Define your max loss. |

*(Source: tastylive research; Trading_orig.md; confirmed by de Saint-Cyr 2023 iron condor study)*

### By Underlying Type

| Underlying | Structure | Why |
|-----------|-----------|-----|
| Drift assets (XSP, IWM, TQQQ, ESTX50, EEM) | Short put + kicker call / XYZ | Drift works for you. Short calls fight the drift. |
| Range-bound (TLT, USO, UNG, WEAT, PDBC, DBA) | Strangle or iron condor | No drift tailwind. Harvest VRP from both sides. |
| Crisis assets (GLD) | Regime-dependent: strangle (calm) / short put only (crisis) | Never short gold calls during equity crashes. |
| Individual stocks (trading only) | Long calls, debit spreads, PMCC | Never naked short on individual stocks. Defined risk only. |

---

## Structure Details

### Short Put

> The simplest premium selling structure. Harvest the put VRP — the structural overpricing of downside protection. *(Carr & Wu 2005, 2009: variance risk premium is larger for puts than calls across all indices tested)*

| Parameter | Guideline |
|-----------|----------|
| Delta at entry | 20–30Δ |
| DTE | 45–60 (sweet spot for theta decay curve) |
| Profit target | Close at 50% of premium received |
| Stop loss | Close at 200% of premium received OR when delta reaches 50Δ |
| Roll trigger | At 21 DTE if not profitable — roll to next 45–60 DTE for a net credit |
| Naked vs spread | Naked on indices if BP allows exit at 2.5× premium. Spread on everything else. |

**When to use:** Drift assets in GO regime. IVR > 50. Above 200d SMA.

**When NOT to use:** Range-bound assets (use strangle/IC instead). Below 200d SMA (use spreads only). IVR < 30 (premium too cheap).

---

### Short Put + Kicker Call ("Synthetic Long")

> Sell 20–30Δ put + buy 20Δ call (or 30/20Δ call spread). Net credit or small debit. Premium income + uncapped (or spread-capped) upside if the underlying runs.

| Parameter | Guideline |
|-----------|----------|
| Short put delta | 20–30Δ |
| Long call delta | 20Δ (or 30/20Δ call spread for defined cost) |
| DTE | 45–360 days depending on conviction |
| Profit target (put side) | Close at 50% of put credit |
| Stop (put side) | Close at 50Δ. Roll to longer DTE and scale per drawdown framework. |
| Kicker management | Let the call ride — free lottery ticket on recovery after a drawdown roll |

**When to use:** Core structure for directional block (XSP, IWM, TQQQ, ESTX50, EEM). Regime: above 50d and 200d SMA.

**Why it works:** The short put collects premium (VRP). The kicker call is funded by the put credit and provides asymmetric upside. In a recovery after drawdown, the kicker captures the bounce that your rolled puts also benefit from. Net effect: premium income in flat/up markets + upside participation in rallies.

---

### XYZ Trade

> A hedged short put structure. The PDS (put debit spread) provides a defined-risk hedge for the naked short put. Preferred over naked short puts for capital efficiency and risk management.

| Component | Construction |
|-----------|-------------|
| **X** — Long put | Buy at ~30Δ |
| **Y** — Short put | Sell at ~25Δ (forms PDS with X, width ~2–5 points) |
| **Z** — Naked short put | Sell at ~20Δ, same expiry as X/Y |

**Ratios:**

| Ratio | Risk Profile | Use Case |
|-------|-------------|----------|
| 111 | Conservative — PDS fully hedges Z | Default for new positions and cautious regimes |
| 221 | Moderate — 2 PDS hedges partially offset Z | Moderate conviction, higher premium |
| 112 | Aggressive — 2 naked puts, 1 PDS | High conviction only. Double downside exposure. Caution. |

**Management:**
- Manage X/Y (PDS) and Z (naked put) independently
- Close Z at 50Δ (hard stop, no exceptions)
- Close full structure at 50% max profit or when short legs reach <10Δ
- Roll Z at 21 DTE if not profitable

**Why XYZ over naked put:** The PDS acts as a free hedge that kicks in on slow downmoves. In a fast crash, the PDS gains value faster than Z loses, partially offsetting the damage. In a slow grind up, Z collects premium while X/Y expire worthless (small cost). *(Source: Options Strategies.md; wider spread + higher delta leads to better hedge against slow downmoves)*

---

### Strangle

> Sell premium on both sides — OTM put + OTM call. Neutral structure that profits from range-bound behavior and VRP. *(Hu et al. 2024: volatility trading via strangles earns the highest absolute returns of all options styles)*

| Parameter | Guideline |
|-----------|----------|
| Put delta | 20–30Δ |
| Call delta | 20–30Δ (symmetric) or 15–20Δ (asymmetric for commodities) |
| DTE | 45–60 |
| Profit target | Close at 50% of total credit |
| Stop loss | Close at 200% of total credit OR either side hits 50Δ |
| Roll | At 21 DTE if not profitable — roll both sides to next 45–60 DTE |

**Asymmetric strangle for commodities:** Commodity vol expands on upside (opposite of equities). Widen the call side to 15–20Δ to reduce blow-up risk on rallies. The put side benefits from producer hedging pressure (Cheng et al., 2021). *(Vol direction: commodities upside = expansion; equities upside = contraction)*

**SPY-specific (from research):** Best backtested setup is 16Δ put / 10Δ call. The asymmetry reflects the equity vol skew — puts are overpriced, calls are fairly priced.

**When to use:** Range-bound assets (TLT, DBA, PDBC). GLD in calm regimes only. Never on drift assets where the call side fights structural drift.

---

### Iron Condor

> A strangle with protective wings. Defined max loss = wing width minus credit received. *(de Saint-Cyr 2023: IC win rates on SPX vary from 50–85% depending on VIX level and DTE)*

| Parameter | Guideline |
|-----------|----------|
| Short put delta | 20–30Δ |
| Short call delta | 20–30Δ (or 15–20Δ for commodities) |
| Wing width | ~5% of underlying price (or $5 for ETFs) |
| DTE | 30–60 (sweet spot: 45 DTE) |
| Profit target | Close at 50% of credit |
| Stop loss | Close at 200% of credit OR either short leg hits 50Δ |
| Max loss | Wing width × 100 − credit received (per lot) |

**When to use:**
- Range-bound assets where you want defined risk (USO, UNG, WEAT, PDBC)
- Any underlying when portfolio BP is limited
- Bear regime (spreads only — no naked shorts)

**IC vs Strangle:** The IC caps your max loss at the wing width. This matters most on gap-prone underlyings (UNG, USO, FXI). The backtest shows IC outperforms strangles on UNG ($4,162 vs $1,017), USO ($571 vs $142), and most commodities — the wings prevent catastrophic gap losses.

**VIX-conditional entry:** Avoid deploying ICs when VIX < 15 — the premium collected doesn't compensate for the risk. Best performance when IVP > 50. *(de Saint-Cyr 2023; confirmed by backtest)*

---

### PMCC — Poor Man's Covered Call

> A leveraged covered call substitute. Buy a deep ITM long call as the "stock replacement" and sell short-term OTM calls against it for income.

| Component | Construction |
|-----------|-------------|
| Long call | ≥120 DTE, 70–80Δ (deep ITM) |
| Short call | 7–45 DTE, 20–40Δ (OTM) |

**Management:**
- Roll short leg before it reaches 21 DTE
- Do not let long call fall below 45 DTE — roll the long leg before that
- If short call goes ITM: roll up and out for a credit, or close the spread
- Total cost = long call debit − short call credits accumulated

**When to use:**
- Extended trend (>20 days running) on Kell/EMA reclaim setups (trading)
- Bullish regime on individual stocks where you want income + upside participation
- **Not for index/commodity DRIFT trades** — the short put + kicker is more capital-efficient for indices

---

### Debit Spread (Bull Call / Bear Put)

> Defined-risk directional trade. Cap both the cost and the risk.

| Parameter | Bull Call | Bear Put |
|-----------|----------|----------|
| Long leg | Buy ATM or slightly OTM call | Buy ATM or slightly OTM put |
| Short leg | Sell further OTM call | Sell further OTM put |
| Width | $3–10 depending on underlying | $3–10 |
| DTE | 30–60 for swing trades | 30–60 |
| Max profit | Width − debit paid | Width − debit paid |
| Max loss | Debit paid | Debit paid |

**When to use:**
- IVR 40–70% (too expensive for naked long calls, too cheap for premium selling)
- High-conviction directional trade with defined risk
- EP (Type A) setups when IV spiked on the gap — the short leg offsets elevated vega

---

### Calendar Spread

> Buy a longer-dated option, sell a shorter-dated option at the same strike. Profits from theta differential and IV crush on the front month.

| Parameter | Guideline |
|-----------|----------|
| Long leg | 60–120 DTE |
| Short leg | 7–30 DTE |
| Strike | ATM or slightly OTM |
| Profit zone | Narrow — underlying must stay near strike |
| Management | Close when front month expires or approaches expiry |

**When to use:**
- Pre-earnings plays where you expect IV expansion into the event *(IVR < 30% at entry, IV expected to expand — Richardson & Veenstra 2006)*
- The back month hedges the front month. Close the entire spread day before earnings — never hold through.

**Caution:** Calendar spreads have narrow profit zones and require the underlying to stay near the strike. Not suitable for trending markets or assets with large expected moves.

---

## Universal Management Rules

These apply across all structures. Non-negotiable.

| Rule | Detail | Source |
|------|--------|--------|
| **50Δ hard stop** | When any short leg reaches 50Δ, close immediately. No exceptions. | Your system |
| **50% profit target** | Close at 50% of max credit. Frees capital, avoids gamma risk. | tastylive research |
| **200% max loss on credit** | Close when loss = 2× credit received. | tastylive / Options Strategies.md |
| **21 DTE roll** | If not profitable at 21 DTE, roll to next 45–60 DTE for a net credit. | tastylive research |
| **IVP ≥ 50 for selling** | Do not sell premium when IVP < 50 or IVR < 30. The edge is gone. | Your system; confirmed by backtest |
| **No naked short on individual stocks** | Defined risk only (spreads, IC). Naked ok on indices only. | Trading_new.md |
| **Never adapt a failing trade** | Close it. Re-enter only on a clean, fresh setup. | Your system |
| **Skew ≠ diversification** | Selling puts and selling strangles capture the same risk factor (~0.9 correlation). They're one bucket. | Kozhan et al., 2013 |

---

## Liquidity Check — Before Any Trade

| Check | Threshold | Action if Failed |
|-------|-----------|-----------------|
| Bid-ask spread | < $0.10 on target strikes | Skip the underlying or move to a more liquid expiry |
| Open interest | > 500 on target strike and expiry | Skip — phantom OI doesn't count |
| Daily options volume | > 2,000 contracts/day on the ETF | Skip — fills will be poor. Use limit orders only on thin names. |
| Underlying price range | $3–$1,000 | Below $3: penny stock, skip. Above $1,000: notional too large per lot. |

---

## Structure Decision Tree

```
Is the underlying drifting upward?
├─ YES → Is IVR > 50?
│        ├─ YES → Short put + kicker call (or XYZ for hedged version)
│        └─ NO  → Long call or debit spread (buy cheap premium)
│
└─ NO (range-bound / neutral) → Is IVR > 50?
         ├─ YES → Strangle (liquid, wide bid-ask < $0.10)
         │        Iron condor (less liquid OR gap-prone underlying)
         └─ NO  → No trade. Wait for premium to fatten.
```

---

## Research Basis

| Finding | Source | Implication |
|---------|--------|-------------|
| Implied vol systematically exceeds realized vol on indices | Carr & Wu 2005, 2009 | Selling premium has a structural edge |
| Iron condor win rate 65–85% at 15-20Δ, 30-45 DTE | de Saint-Cyr 2023; ApexVol | IC is the highest-probability defined-risk structure |
| Volatility selling = highest Sharpe ratio options style | Hu et al., 2024 | Short vol beats directional, long vol, and delta-neutral |
| Retail loses 5-14% per trade buying options around earnings | Losing is Optional (2021) | Be the seller, not the buyer — especially pre-earnings |
| Weekend theta is overpriced | Jones & Shemesh 2017 | Enter short premium positions Thursday/Friday to capture weekend decay |
| Skew premium and variance premium are the same risk factor | Kozhan et al. 2013 | Don't diversify between puts and strangles — they're one bet |
| Commodity puts overpriced by producer hedging | Cheng, Tang & Yan 2021 | Selling puts on commodities captures hedging pressure premium |
| VIX mean-reverts strongly | Bailey et al. 2014 | Sell premium after VIX spikes — reversion is the core mechanism |
| Agriculture commodities have 5-10× the VRP of equities | Core PM backtest 2026 | WEAT, DBA, PDBC are the highest-VRP premium selling targets |

Sources:
- [Iron Condor Strategy Guide 2026 - Trasignal](https://trasignal.com/blog/learn/iron-condor-strategy/)
- [Iron Condor Win Rate Guide - ApexVol](https://apexvol.com/strategies/iron-condor)
- [Short Put Explained - tastytrade](https://tastytrade.com/learn/trading-products/options/short-put/)
- [Strangle vs Iron Condor - OptionsTrading.org](https://www.optionstrading.org/blog/strangle-vs-iron-condor/)
