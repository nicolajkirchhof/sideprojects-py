# Academic Foundations — Investor Skill

The strategies in this skill are grounded in peer-reviewed research. This file summarises
the key findings and their direct implications for portfolio construction and DRIFT trades.

---

## 1. The Equity Risk Premium (Long-Term Foundation)

**Core finding:** Stocks have outperformed risk-free bonds by approximately 5–7% annually
over 100+ years across all major markets. This "risk premium" is the compensation investors
receive for bearing equity risk.

**Why it persists:** The premium is structural — it compensates for real economic risk
(recessions, drawdowns, uncertainty). Investors who cannot stomach drawdowns exit,
permanently transferring wealth to those who hold through volatility.

**Implication for LONG-TERM:** Hold equities for the full risk premium. Selling during
downturns captures the worst of the risk without the eventual reward. Time in market >
timing the market.

**Implication for DRIFT:** Selling puts on index ETFs harvests the equity risk premium
in a different form — you collect premium because you are taking on downside risk that
other market participants want to offload.

---

## 2. The Five Factor Model (Fama & French, 1993–2015)

**Core finding:** Asset returns are explained by five systematic factors:
1. **Market (Mkt-RF)**: excess return of the market over risk-free rate
2. **Size (SMB)**: small cap stocks outperform large cap over time
3. **Value (HML)**: cheap stocks (high book-to-market) outperform expensive ones
4. **Profitability (RMW)**: profitable firms outperform unprofitable ones
5. **Investment (CMA)**: conservative investors outperform aggressive ones

**The size premium:** Small cap stocks outperform large cap by ~2–3% annually over long
periods. Holding period matters — the premium is unreliable over 1–3 years but historically
robust over 10+ years.

**The value premium:** Cheap stocks (low P/B, P/E) outperform expensive ones by ~3–5%
annually. Most concentrated in small cap value (combining size + value).

**Implication:** A portfolio tilted toward small cap value (AVUV, ZPRV) earns a higher
expected return than a pure market-cap index, at the cost of higher short-term volatility
and longer periods of underperformance. Commitment and a long horizon are required.

---

## 3. Warren's Alpha (Frazzini, Kabiller & Pedersen, 2013)

**Core finding:** Buffett's outperformance is explained by three things:
1. **Leverage**: Berkshire uses 1.7× leverage via insurance float
2. **Quality bias**: preference for safe, profitable, low-beta stocks
3. **Value bias**: buying cheap relative to fundamentals

**The BAB (Betting Against Beta) factor:** Low-beta stocks earn higher risk-adjusted returns
than high-beta stocks — the opposite of what CAPM predicts. This is because leverage-
constrained investors bid up high-beta stocks and undervalue low-beta ones.

**Implication for LONG-TERM:** Quality factor (profitable, low-debt companies) and value
factor exposure adds expected return. The Five Factor Model captures this systematically.

---

## 4. Positive Drift — Index ETF Simulation Findings

**Core finding (from personal research + Tasty Trade data):** Over 10+ years of simulation:
- Selling puts on SPY, QQQ, IWM at various expirations (0DTE to 180DTE) is profitable
  across nearly all timeframes
- Short naked calls have negative expectancy (underlying drift works against them)
- Short naked puts have positive expectancy (underlying drift works for them)
- 45–60 DTE offers the best balance of theta decay rate and time for management

**Why 45–60 DTE is optimal:**
- Theta decay is non-linear — accelerates in the final 30 days
- Entering at 45–60 DTE captures the "knee" of the decay curve
- Enough time to be wrong and still manage the position before it becomes critical
- The 20–30Δ strike at 45 DTE has won historically the large majority of occurrences

**Seasonality note:** July is historically the most positive month for indices. No strong
seasonality signal exists for the 45–60 DTE strategy specifically, but be more cautious in
September/October (historically weakest months).

---

## 5. Century Momentum (Geczy & Samonov, 2016)

**Core finding:** The "momentum" anomaly — past 12-month winners continue to outperform
for 1–3 months — has persisted for nearly 100 years across all asset classes tested.

**The signal:** An asset's 12-month cumulative return (excluding the most recent month)
predicts its next 1–3 month performance.

**Implication for DRIFT regime filter:**
- Only sell puts on index ETFs that have a positive 12-month return (above 12M SMA)
- This simple filter eliminates the worst environments for selling premium
- When indices are below their 12-month SMA, switch to spreads-only or reduce exposure

---

## 6. VIX Mean Reversion (Bailey et al., "What Makes the VIX Tick?")

**Core finding:** VIX has a strong structural tendency to mean-revert. It overshoots
realized volatility during macro stress events, creating a persistent premium for those
who sell vol after spikes.

**The mechanism:** VIX spikes on macroeconomic news and uncertainty, but often rises
above the actual subsequent realized volatility. This overshoot is driven by panic hedging
demand from institutions.

**The signal:** VIX spike above 25 → "lower high" in VIX on subsequent days = reversion
underway = optimal time to sell premium.

**Practical implication:** After a VIX spike, entering 45–60 DTE short puts captures:
1. Elevated premium (IVP is high)
2. Mean-reversion tailwind (VIX heading back down)
3. Potential equity recovery (underlying drift resumes)

---

## 7. Volatility Targeting (Harvey et al., "Alpha Generation and Risk Smoothing")

**Core finding:** Scaling position size inversely to recent realized volatility improves
risk-adjusted returns by reducing exposure during high-volatility clusters.

**Why volatility clusters:** Large price moves tend to be followed by more large moves
(the GARCH effect). Maintaining constant size through volatility clusters results in
disproportionate drawdowns relative to expected return.

**Practical implementation for DRIFT:**
- When 20-day realized vol is low (< 15% annualized): sell at normal size
- When 20-day realized vol is elevated (15–25%): reduce size by 25–30%
- When 20-day realized vol is very high (> 25%): reduce size by 50%, widen strikes
- Review every Friday at close — adjust next week's sizing accordingly

---

## 8. VTS Slope Alpha (Vasquez, implied vol term structure research)

**Core finding:** Stocks where long-term implied volatility (6-month IV) significantly
exceeds short-term IV (1-month IV) — a "steep slope" — generate higher subsequent returns.
The market overestimates long-term uncertainty relative to near-term realized risk.

**Implication for DRIFT:** When an index ETF's IV term structure is steep (long-dated IV
>> short-dated IV), it is an especially favourable environment for selling longer-dated
puts — you are collecting a premium that is structurally inflated.

**How to check:** Compare 30-day IV to 90-day IV on the options chain. A 90-day IV that
is 20%+ above 30-day IV = steep slope = priority for new 45–60 DTE short puts.

---

## 9. Options Complexity Alpha (Hu et al., "Who Profits From Trading Options?")

**Core finding:** Retail investors using simple single-leg options (buying calls/puts)
systematically lose money. Those using multi-leg "volatility strategies" and "risk neutral"
combinations earn significantly higher returns and Sharpe ratios.

**Why complexity wins:**
- Single directional bets fail because bid/ask spreads and theta exceed the directional move
- Multi-leg strategies (XYZ, spreads, strangles) isolate specific effects (IV crush, drift,
  pinning) while neutralizing the "retail drag" of simple options
- The ability to manage delta, gamma, and vega simultaneously is a persistent skill edge

**Implication:** The XYZ trade structure is preferred over naked short puts precisely
because its complexity provides better risk management and a cleaner exposure profile.
Never let simplicity be an excuse for inferior risk management.

---

## 10. Quarter-End Rebalancing Headwind (Harvey et al., 2025)

**Core finding:** Large institutions (pensions, sovereign wealth) engage in regular
rebalancing that creates predictable price patterns. When equities outperform bonds over
a period, funds are forced to sell stocks to return to mandate targets — creating a
17bps daily headwind in the final days of quarters.

**Implication:**
- Avoid opening new long-delta positions in the final 3 trading days of each quarter
- Use this period to close existing profitable positions and take profits
- The first week of the new quarter often sees a reversal as forced selling ends →
  a good entry point for new 45–60 DTE short puts at elevated IVP
