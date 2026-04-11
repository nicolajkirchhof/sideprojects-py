# Swing Plot Dashboard — Profit Mechanism Map

This document maps every widget on the `swing_plot` dashboard to the
profit mechanisms (PMs) defined in `TradingPlaybook.md` and `InvestingPlaybook.md`,
so each plot can be traced back to a concrete decision in a real strategy.

Roles:
- **T** — Swing Trader (`TradingPlaybook.md`, PM-01 … PM-11)
- **I-LT** — Long-term investor (Investing §01, factor ETFs, buy-and-hold)
- **I-DR** — DRIFT investor (Investing §02, directional + neutral VRP harvesting)

---

## Tab: Price Chart

6-pane PyQtGraph chart with on-demand IBKR refresh.

| Widget | Role | PM / Decision |
|---|---|---|
| OHLC candles + MA overlay (5/10/20/50/100/200) + BB | T, I-DR | Universal context — price structure, trend, Bollinger envelope |
| Legend overlay (symbol, OHLC, MA values + distance %) | T | PM-01 / PM-05 — instant read on trend alignment and MA distance |
| SPY normalised overlay | T | PM-05 RS/RW — visual relative performance |
| Volume bars (coloured by day change) + V20 MA | T | PM-01 — volume confirmation on breakouts and VCP dry-up |
| ATR% (ATRP9 + ATRP20) + COTR (right axis) | T | PM-01 stop sizing (ATR%) + impulse detection (COTR ±1.75 threshold) |
| IV/HV (right axis) + IVP (left axis, 50% ref line) | T, I-DR | PM-04 option structure choice (IVP); DRIFT IVP>50 gate; VRP regime |
| Comparative RS vs SPY (RS + RSMA20) | T | **PM-05 RS/RW Divergence** — the core filter for every swing entry |
| TTM Squeeze (momentum + squeeze dots) | T | PM-01 — breakout timing signal, VCP confirmation |
| IBKR connection indicator + freshness badge | T, I-DR | Data recency awareness; manual refresh sync |

**Primary role:** T + I-DR. The chart is the universal starting point.

---

## Tab: Probability Trees

| Widget | Role | PM / Decision |
|---|---|---|
| Probability trees (daily/weekly/monthly) | T | Expected path distribution for PM-01 target setting |
| Probability trees | I-DR | Expected range for IC wing selection (25Δ strike realism check) |

---

## Tab: Seasonality

| Widget | Role | PM / Decision |
|---|---|---|
| Daily-return violins by weekday | T | PM-02 / PM-03 entry timing — avoid stat-weak weekdays for new swing entries |
| Daily-return violins by month | I-DR | Calendar-driven VRP harvest: NG/UNG (Oct–Nov), WEAT (crop cycle), GLD (May weakness) |
| Daily-return violins by month | T | PM-03 pre-earnings drift window alignment |

**Primary role:** I-DR (seasonal commodities drive the top-tier ICs — UNG 72% win, WEAT 77%, PDBC 79%).

---

## Tab: Volatility

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

## Tab: Trend Regime

Purpose: quantify MA20 regime behaviour — uptrend persistence, pullback depth and
recovery, breakdown duration. The trader's conviction tab.

| Widget | Role | PM / Decision |
|---|---|---|
| MA20 Regime episode counts + recovery rate | T | PM-01 Breakout Momentum — quantifies "how reliable is uptrend persistence" on this symbol |
| MA20 Regime median durations | T | PM-01 holding-period target; PM-09 mean-reversion window |
| Episode duration distribution (violins) | T | PM-09 Mean Reversion to Trend — pullback duration gives timeout for dip entries |
| Max depth below MA20 (violins) | T | PM-09 stop-loss sizing for pullback entries; PM-01 stop-slack below MA20 |
| Forward returns by regime (5d / 10d / 20d) | T | PM-01 — validates "buy uptrend, avoid breakdown". Measured from episode start. |

**Primary role:** T (PM-01, PM-09).

---

## Tab: Pullback and VCP

Purpose: support pullback entries on uptrending names and detect Volatility
Contraction Patterns (VCP) for breakout timing.

| Widget | Role | PM / Decision |
|---|---|---|
| Intra-trend retracement depth (violins + scatter) | T | PM-09 pullback entry sizing — is 0.5×, 1×, 1.5× ATR the right buy level? |
| **RS line vs SPY + 63d RS percentile** | T | **PM-05 RS/RW Divergence — the core filter.** Every PM-01/PM-02/PM-09 entry requires RS positive. |
| **RS line vs SPY + 63d percentile** | T | PM-02 PEAD — post-earnings drift only held in RS-positive names |
| **VCP tightness timeline** (10d range/close, tight/loose bands) | T | **PM-01 Breakout Momentum** — is the name in a tight consolidation right now? |
| VCP tightness histogram + current marker | T | PM-01 — how tight is "tight" for this specific symbol (relative to its own history)? |
| **Conditional fwd returns by tightness tercile** (5d/10d/20d) | T | PM-01 — does this name actually follow through after tight bases? |

**Primary role:** T (PM-01, PM-05, PM-09).

---

## Tab: Move and Options

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
| **Overnight reversal** — distributions, hit rate, rolling edge | T | **PM-08 Overnight Reversal** — does this name show a reversal premium after large intraday drops? |

**Primary role:** T (PM-01, PM-04, PM-08). Secondary: long-premium expression of any PM.

---

## Tab: Earnings Drift

Purpose: visualise PM-02 (PEAD) and PM-03 (Pre-Earnings Anticipation) on a
per-symbol basis using the momentum_earnings per-ticker parquet dataset.

| Widget | Role | PM / Decision |
|---|---|---|
| All earnings events cluster (t-20 to t+24, median + IQR) | T | PM-02 — is PEAD real on this name? Direction and magnitude of post-earnings drift. |
| Beats cluster + Misses cluster | T | PM-02 — surprise-direction split; beat drift vs miss drift asymmetry |
| Median drift comparison (All / Beats / Misses) | T | PM-02 / PM-03 — combined view of pre- and post-earnings drift patterns |
| Hit rate at +5d / +10d / +20d by group | T | PM-02 — statistical reliability of the drift at each horizon |
| **Pre-drift vs post-drift scatter** (t-20→t-1 vs t-1→t+5, coloured beat/miss) | T | **PM-03 Pre-Earnings Anticipation** — does pre-earnings drift predict the surprise direction? |
| **Conditional hit rate** (P(beat\|pre>0), P(beat\|pre<0), P(+5d>0\|pre>0/pre<0)) | T | **PM-03** — quantifies the predictive value of pre-earnings drift for entry timing |

**Primary role:** T (PM-02, PM-03).

---

## Summary — Which role each tab serves

| Tab | Trader (T) | LT Investor (I-LT) | DRIFT (I-DR) |
|---|:---:|:---:|:---:|
| Price Chart | ● | ○ | ● |
| Probability Trees | ● | — | ○ |
| Seasonality | ○ | — | ● |
| Volatility | ○ | ○ | ● |
| Drawdown | ○ | ● | ○ |
| Trend Regime | ● | — | — |
| Pullback and VCP | ● | — | — |
| Move and Options | ● | — | ○ |
| Earnings Drift | ● | — | — |

● primary · ○ secondary · — not relevant

---

## Deliberate omissions

These were dropped in the refactor because they did not map cleanly to any PM:

- **Gap vs intraday decomposition** — no PM uses gap-fill as an entry edge (swing holds span many sessions; DPM-04 is a day-trade PM).
- **IV-change vs price-change scatter** — noisy, no PM uses delta-IV dynamics directly.
- **IV Rank (IVR)** — replaced by **IV Percentile (IVP)** across the dashboard; IVP is what the InvestingPlaybook references (`IVP > 50` is the DRIFT rotational filter).
- **EMA Distance pane** — MA distances are now shown inline in the chart legend overlay.

## Known gaps (future work)

- **DPM-06 0DTE VRP by day-of-week / VIX regime** — separate dashboard, out of scope for swing_plot
- **Investing §01 factor-tilt drift** — cross-ETF allocation drift vs target weights (better served by a dedicated portfolio dashboard)
