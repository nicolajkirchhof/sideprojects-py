# Swing Plot Dashboard — Profit Mechanism Map

This document maps every widget on the refactored `swing_plot` dashboard to the
profit mechanisms (PMs) defined in `TradingPlaybook.md` and `InvestingPlaybook.md`,
so each plot can be traced back to a concrete decision in a real strategy.

Roles:
- **T** — Swing Trader (`TradingPlaybook.md`, PM-01 … PM-11)
- **I-LT** — Long-term investor (Investing §01, factor ETFs, buy-and-hold)
- **I-DR** — DRIFT investor (Investing §02, directional + neutral VRP harvesting)

---

## Tab: Seasonality (Daily/Monthly Stats)

| Widget | Role | PM / Decision |
|---|---|---|
| Daily-return violins by weekday | T | PM-02 / PM-03 entry timing — avoid stat-weak weekdays for new swing entries |
| Daily-return violins by month | I-DR | Calendar-driven VRP harvest: NG/UNG (Oct–Nov), WEAT (crop cycle), GLD (May weakness) |
| Daily-return violins by month | T | PM-03 pre-earnings drift window alignment |

**Primary role:** I-DR (seasonal commodities drive the top-tier ICs — UNG 72% win, WEAT 77%, PDBC 79%).

---

## Tab: Volatility & VRP

Purpose: decide whether short premium is attractive *today*, and whether the
symbol passes the IVP>50 gate shared by the entire DRIFT rotational block.

| Widget | Role | PM / Decision |
|---|---|---|
| **IVP timeline (1y)** with 50% DRIFT-eligible band | I-DR | Entry filter for every rotational underlying (EEM, FXI, EWZ, UNG, USO, WEAT, PDBC, TLT, BNO). IVP ≥ 50 → eligible. |
| **IVP timeline** | T | PM-04 OTM Informed Flow — low IVP = cheap long options; high IVP = favour spreads |
| **VRP rolling (IV − HV20) + percentile** | I-DR | Neutral block (commodities/bonds) — core harvest edge. High positive VRP at high percentile = size up. |
| IV − HV premium violins | I-DR | Distribution of the edge across different HV windows; sanity-check VRP stability |
| IV vs HV time series | I-DR | Regime view: calm vs crisis markets for GLD structure selection (strangle vs short-put-only above VIX 25) |
| IV vs HV14 / HV20 scatters | I-DR | Persistence check — is IV consistently above realised, or only conditionally? |
| **Rolling 60d correlation to SPY** | I-LT | Diversification check for factor ETFs (MSCI EM Small Cap, Euro Stoxx 600, FTSE 100, Japan). |
| **Rolling 60d correlation to SPY** | I-DR | Directional block sizing — FXI/EWZ/ESTX50 justification rests on low correlation (FXI 0.30–0.55, EWZ 0.40–0.60, ESTX50 ~0.65). |

**Primary role:** I-DR. Every widget here feeds a DRIFT entry or sizing decision.

---

## Tab: Drawdown

Purpose: bound the worst-case path for a held position, whether an ETF or a
short-put assigned and held.

| Widget | Role | PM / Decision |
|---|---|---|
| Severity × duration scatter (IV-coloured) | I-LT | LT hold tolerance on factor ETFs — how bad can it get before recovery? |
| Severity × duration scatter | I-DR | Short-put assignment window planning — if assigned, how long typically underwater? |
| Short / Med / Long DD paths (overlaid histories) | I-LT | Psychological prep for holding through 30d / 30–65d / 65d+ drawdowns |
| Short / Med / Long DD paths | I-DR | 45 DTE cycle horizon comparison: most DD clusters inside or outside a cycle? |
| **Time-underwater histogram (log scale)** | I-LT | Core LT stat — median and 90th-percentile days to next ATH on factor ETFs |
| **Time-underwater histogram** | I-DR | Risk-of-stuck-assignment for GLD, UNG, WEAT — how long might a short strike be underwater? |
| **Time-underwater histogram** | T | PM-01 max-adverse-excursion reference — contextualises swing stop distance |

**Primary role:** I-LT. Secondary: I-DR (assignment planning).

---

## Tab: Trend & Pullback

Purpose: support pullback entries on uptrending, RS-positive names. This is
the trader's home tab.

| Widget | Role | PM / Decision |
|---|---|---|
| MA20 Regime episode counts + recovery rate | T | PM-01 Breakout Momentum — quantifies "how reliable is uptrend persistence" on this symbol |
| MA20 Regime median durations | T | PM-01 holding-period target; PM-09 mean-reversion window |
| Episode duration distribution (violins) | T | PM-09 Mean Reversion to Trend — pullback duration gives timeout for dip entries |
| Max depth below MA20 (violins) | T | PM-09 stop-loss sizing for pullback entries; PM-01 stop-slack below MA20 |
| Forward returns by regime (5d / 10d / 20d) | T | PM-01 — validates "buy uptrend, avoid breakdown". Now correctly measured from episode start. |
| Intra-trend retracement depth (violins + scatter) | T | PM-09 pullback entry sizing — is 0.5×, 1×, 1.5× ATR the right buy level? |
| **RS line vs SPY + 63d RS percentile** | T | **PM-05 RS/RW Divergence — the core filter.** Every PM-01/PM-02/PM-09 entry requires RS positive. |
| **RS line vs SPY + 63d percentile** | T | PM-02 PEAD — post-earnings drift only held in RS-positive names |

**Primary role:** T (PM-01, PM-05, PM-09).

---

## Tab: Move & OTM Options

Purpose: decide whether a swing thesis can be expressed via **long OTM options**
(fat tails required) and size stops/targets. Bridges trader and long-premium
trades.

| Widget | Role | PM / Decision |
|---|---|---|
| ATR20-normalised move histogram + EXPLOSIVE/GRADUAL flag | T | PM-01 stop sizing in ATR units; distinguishes "fire" names from "plodders" |
| Tail frequencies by direction (>1.0× / 1.5× / 2.0× / 3.0× ATR) | T | PM-04 OTM Informed Flow — asymmetry of upside vs downside tails |
| **OTM Long-Option Viability (reframed QQ)** — VIABLE / MARGINAL / POOR flag | T | PM-04 — direct answer to "can I express this swing with a long OTM call/put?" Kurtosis + upper tail freq vs Gaussian. |
| **IVP timeline** (1y) | T | PM-04 — combined with OTM viability: VIABLE + low IVP = long OTM; VIABLE + high IVP = debit spread |
| HV regime timeline (Low/Med/High with shading) | T | PM-01 holding horizon — extend holds in Low HV, tighten in High HV |
| HV episode duration violins | T | Regime persistence — how long do vol states last on this name? |
| HV transition heatmap | T | Probability that High HV mean-reverts vs persists — sizing signal for long gamma |
| Impulse forward returns (±1.75× ATR impulse sessions) | T | **PM-01 exit rule validation** — after a big move, does it continue (hold) or fade (exit)? |

**Primary role:** T (PM-01, PM-04). Secondary: long-premium expression of any PM.

---

## Tab: Price Chart + Probability Trees

(Unchanged by this refactor; included for completeness.)

| Widget | Role | PM / Decision |
|---|---|---|
| PyQtGraph multi-pane chart | T, I-DR | Universal context — price, MAs, Bollinger, volume, IV/HV, SPY overlay |
| Probability trees (daily/weekly/monthly) | T | Expected path distribution for PM-01 target setting |
| Probability trees | I-DR | Expected range for IC wing selection (25Δ strike realism check) |

---

## Summary — Which role each tab serves

| Tab | Trader (T) | LT Investor (I-LT) | DRIFT (I-DR) |
|---|:---:|:---:|:---:|
| Seasonality | ○ | — | ● |
| Volatility & VRP | ○ | ○ | ● |
| Drawdown | ○ | ● | ○ |
| Trend & Pullback | ● | — | — |
| Move & OTM Options | ● | — | ○ |
| Price Chart | ● | ○ | ● |
| Probability Trees | ● | — | ○ |

● primary · ○ secondary · — not relevant

---

## Deliberate omissions

These were dropped in the refactor because they did not map cleanly to any PM:

- **Gap vs intraday decomposition** — no PM uses gap-fill as an entry edge (swing holds span many sessions; DPM-04 is a day-trade PM).
- **IV-change vs price-change scatter** — noisy, no PM uses delta-IV dynamics directly.
- **IV Rank (IVR)** — replaced by **IV Percentile (IVP)** across the dashboard; IVP is what the InvestingPlaybook references (`IVP > 50` is the DRIFT rotational filter).

## Known gaps (future work)

Still missing for full PM coverage:

- **PM-02 PEAD / PM-03 Pre-Earnings Anticipation** — no earnings-reaction clustering view yet (consider a per-earnings ATR magnitude + fwd-return tab)
- **VCP tightness / range-contraction detector** — PM-01 entry refinement
- **DPM-06 0DTE VRP by day-of-week / VIX regime** — separate dashboard, out of scope for swing_plot
- **Investing §01 factor-tilt drift** — cross-ETF allocation drift vs target weights (better served by a dedicated portfolio dashboard)
