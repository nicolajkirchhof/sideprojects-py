# Intraday PM — Research Summary

Generated: 2026-04-21
Period covered: 2020-01-01 to 2026-04-01
Instruments tested: IBDE40, IBGB100, IBUS500, IBUST100, IBUS30, IBJP225, IBES35, IBAU200, IBEU50, IBFR40, IBCH20, IBNL25, USGOLD
Automation target instruments: FDXS (IBDE40), MNQ (IBUST100), MES (IBUS500), MYM (IBUS30), FN225M (IBJP225)

---

## Master Verdict Table

All strategies backtested, with overall Go/No-go per instrument.
Signal bars marked † have been upgraded from the original Hougaard convention based on candle scan.

| Strategy | IBDE40 | IBGB100 | IBUS500 | IBUST100 | IBUS30 | IBJP225 | IBES35 | IBAU200 | IBEU50/IBFR40/IBCH20/IBNL25/USGOLD |
|----------|--------|---------|---------|----------|--------|---------|--------|---------|--------------------------------------|
| **OCO candle scan** | **Go** | No-go | No-go | **Go** | **Go** | **Go** | Pilot | Pilot | No-go |
| **Following Range Break** | Go | Pilot | Go | Go | Go | Go | No-go | — | — |
| **Hougaard ASRS** (4th 5m) | No-go | — | No-go | **Go** | — | — | — | — | — |
| **Hougaard SRS** (2nd 15m) † | **Go** | No-go | No-go | **Go** | — | — | — | — | — |
| **OCO Opening Bar 30m** (scan) | — | No-go | No-go | **Go** | — | — | — | — | — |
| **Hougaard FOMC Rule of 4** | — | — | **Go** (event-only) | — | — | — | — | — | — |
| **ORB 15m** | **Go** | No-go | No-go | **Go** | — | — | — | — | — |
| **ORB 30m** | **Go** | No-go | No-go | **Go** | — | — | — | — | — |
| **VWAP Extrema** | **Go** | **Go** | **Go** | **Go** | — | — | — | — | — |
| **Noon Iron Butterfly** | No-go | — | **Go** | — | — | — | — | — | — |
| **Micro/Macro Trend** | Do not pursue | — | — | — | — | — | — | — | — |
| **0DTE Iron Condor** | — | — | No-go | — | — | — | — | — | — |
| **Dealer Gamma Regime** | No filter value | — | No filter value | — | — | — | — | — | — |

---

## Candle Scan Findings

Two scans run. Scan 1 (original, 4 instruments, 384 combinations): 95 positive EV.
Scan 2 (expanded, 13 instruments, 1248 combinations): 282 positive EV.

**ATR stop consistently matches or beats bar-range stop** across the top combinations —
use ATR stop as default for SRS-style strategies. Bar-range stop remains valid for ASRS.

### Scan 1 upgrades (IBDE40, IBUST100)

**IBDE40:** The conventional SRS signal bar (09:15, 2nd 15-min bar) is suboptimal.
The **3rd 15-min bar at 09:30** is the peak signal: EV +3.04, Sharpe +0.054 vs
EV +1.03, Sharpe +0.018 for the original. The 09:15 bar is too early — the DAX
open finds its direction in the first 30 minutes, not the first 15.

**IBUST100:** The SRS baseline (09:45, 2nd 15-min bar) sits near the frontier (Sharpe +0.053)
but the **1st 30-min bar at 09:30** is the outright best: EV +5.81, Sharpe +0.067.

| Instrument | Original signal | Original Sharpe | Upgraded signal | Upgraded Sharpe | Delta |
|------------|----------------|-----------------|-----------------|-----------------|-------|
| IBDE40 | 15min/bar1/atr (09:15) | +0.018 | 15min/bar2/atr (09:30) | +0.054 | +0.036 |
| IBUST100 | 15min/bar1/atr (09:45) | +0.052 | 30min/bar0/atr (09:30) | +0.067 | +0.015 |

### Scan 2 new discoveries (9 additional instruments)

**IBUS30 (Dow Jones):** Highest Sharpe in the entire expanded scan. Multiple early-session
5-min bars show strong edge, peaking at **5min/bar2/bar_range (09:40 ET)** with
EV +8.67, Sharpe +0.098. The 30min/bar2 (10:30 ET) adds EV +11.19, Sharpe +0.073
with larger absolute moves. Edge is broad across bars 1–5 (09:35–09:55 ET).
Automation instrument: **MYM** (Micro Dow Jones, CBOT).

**IBJP225 (Nikkei 225):** Broad, consistent edge across many bars and timeframes.
Best single bar: **5min/bar13/bar_range (10:05 Tokyo)** at Sharpe +0.088.
Recommended signal for automation: **30min/bar1/atr (09:30 Tokyo, closes 10:00)**
at EV +10.60, Sharpe +0.075 — simpler and avoids the need to count 13 5-min bars.
Note: automation requires FN225M (OSE.JPN, JPY-denominated) — currency and
exchange access implications; evaluate separately before live deployment.

**IBES35 (IBEX 35):** Weak positive edge. Best: 10min/bar4/bar_range (09:40 Frankfurt)
at Sharpe +0.037. Not strong enough to automate; pilot-only status.

**IBAU200 (ASX 200):** Weak positive edge. Best: 30min/bar1/atr (10:30 Sydney)
at Sharpe +0.032. Pilot-only; Australian session adds operational complexity.

**IBFR40, IBCH20:** Near-zero best Sharpe (+0.021, +0.020). No viable edge.

**IBEU50:** No-go across all bars (best -0.074). EuroStoxx 50 lacks OCO bracket edge
despite sharing a session with the viable IBDE40.

**IBNL25 (AEX):** Severe negative Sharpe (-0.652), win rate only 17–18%. The AEX has
a much smaller point range than other EU indices — the fixed 2-pt entry offset is
proportionally too large, making entries uneconomic. This is a parameterisation issue,
not a structural edge question; do not retest without instrument-specific offset calibration.

**USGOLD:** No OCO bracket edge (best Sharpe -0.268). Gold's intraday behaviour does
not produce the directional opening breakouts that drive the other instruments.

### Full cross-instrument Sharpe ranking (best combo per instrument)

| Instrument | Best combo | Best Sharpe | Verdict |
|------------|-----------|-------------|---------|
| IBUS30 | 5min/bar2/bar_range (09:40 ET) | +0.098 | **Go** |
| IBJP225 | 5min/bar13/bar_range (10:05 Tokyo) | +0.088 | **Go** |
| IBUST100 | 30min/bar0/atr (09:30 ET) | +0.067 | **Go** |
| IBDE40 | 15min/bar2/atr (09:30 Frankfurt) | +0.054 | **Go** |
| IBES35 | 10min/bar4/bar_range (09:40 Frankfurt) | +0.037 | Pilot |
| IBAU200 | 30min/bar1/atr (10:30 Sydney) | +0.032 | Pilot |
| IBFR40 | 30min/bar1/atr (09:30 Frankfurt) | +0.021 | No-go |
| IBCH20 | 30min/bar2/atr (10:00 Zurich) | +0.020 | No-go |
| IBGB100 | best across 96 combos | -0.020 | No-go |
| IBUS500 | best across 96 combos | -0.019 | No-go |
| IBEU50 | best across 144 combos | -0.074 | No-go |
| USGOLD | best across 144 combos | -0.268 | No-go |
| IBNL25 | best across 144 combos | -0.652 | No-go (offset issue) |

---

## Go Strategies — Key Parameters

### 1. VWAP Extrema (BT-6-S2)
**Instruments:** IBDE40, IBGB100, IBUS500, IBUST100
**Edge:** VWAP trend continuation — bracket placed in trend direction at extrema
**Filter:** High-edge time slots (success rate ≥ 60%, EV > 2 pts)

| Instrument | N trades/yr | EV (pts) | Sharpe | Key slots |
|------------|-------------|----------|--------|-----------|
| IBDE40 | ~15,000 | +49.77 | +0.524 | 08:00-09:30 Frankfurt, 14:30-17:30 |
| IBGB100 | ~17,500 | +17.58 | +0.495 | 07:00-08:30 London |
| IBUS500 | ~13,500 | +11.37 | +0.444 | 08:30-09:05 ET |
| IBUST100 | ~11,000 | +57.61 | +0.520 | 08:30-09:05 ET |

> Note: EV figures are in CFD points and depend on lot size for absolute P&L.
> High signal count — requires automated execution; manual monitoring not viable.

---

### 2. OCO Early Session Break — IBUS30 (scan-discovered)
**Signal:** 3rd 5-min bar at 09:40 ET *(strongest single bar in entire expanded scan)*
**Entry:** OCO bracket ±2 pts of bar high/low
**Stop:** bar range + 4 pts (bar-range trailing stop)
**Exit:** IBKR native trailing stop or 16:00 ET
**Automation instrument:** MYM (Micro Dow Jones, CBOT)

| N | Win% | EV (pts) | Sharpe |
|---|------|----------|--------|
| 1546 | 44.4% | +8.67 | +0.098 |

> Highest Sharpe across all 13 instruments and 1248 combinations.
> Edge is broad: bars 1–5 (09:35–09:55 ET) all show Sharpe +0.070–+0.098.
> Recommend piloting with bar2 (09:40) as primary; bar5 (09:55) as alternative.

---

### 3. OCO Opening Bar 30m — IBJP225 (scan-discovered)
**Signal:** 2nd 30-min bar at 09:30 Tokyo *(closes 10:00; broad edge across many bars)*
**Entry:** OCO bracket ±2 pts of bar high/low
**Stop:** 20% of 14-day daily ATR
**Exit:** IBKR native trailing stop or 15:30 Tokyo
**Automation instrument:** FN225M (Micro Nikkei, OSE.JPN — JPY-denominated; evaluate exchange access before deploying)

| N | Win% | EV (pts) | Sharpe |
|---|------|----------|--------|
| 1481 | 44.6% | +10.60 | +0.075 |

> Edge is broad — at least 10 bar/timeframe combinations above Sharpe +0.070.
> Peak bar is 5min/bar13 (10:05 Tokyo, Sharpe +0.088) but 30min/bar1 is the
> simplest operationally and avoids over-fitting a single bar.
> Automation requires Japan futures access; assess separately before live deployment.

---

### 4. Hougaard SRS — IBDE40 † (upgraded signal bar, renamed from §2)
**Signal:** 3rd 15-min bar at 09:30 Frankfurt *(upgraded from 09:15 via candle scan)*
**Entry:** OCO bracket ±2 pts of signal bar high/low
**Stop:** 20% of 14-day daily ATR
**Exit:** 2-bar trailing stop on 15-min bars or 17:30 Frankfurt
**Automation instrument:** FDXS

| N | Win% | EV (pts) | Sharpe |
|---|------|----------|--------|
| 1567 | 43.7% | +3.04 | +0.054 |

> Original (09:15 bar): EV +1.03, Sharpe +0.018 (Long +1.46/+0.031, Short +0.61/+0.010). Upgrade adds +0.036 Sharpe.
> Per-direction data not available for the upgraded bar from candle scan output.

---

### 5. OCO Opening Bar 30m — IBUST100 (scan-discovered)
**Signal:** 1st 30-min bar at 09:30 ET *(top US performer in original candle scan)*
**Entry:** OCO bracket ±2 pts of bar high/low
**Stop:** 20% of 14-day daily ATR
**Exit:** 2-bar trailing stop on 30-min bars or 16:00 ET
**Automation instrument:** MNQ

| N | Win% | EV (pts) | Sharpe |
|---|------|----------|--------|
| 1539 | 42.2% | +5.81 | +0.067 |

> Highest OCO bracket Sharpe on MNQ. Across all instruments, IBUS30/MYM (strategy §2) is highest (+0.098).
> Fires at the same time as the ORB window — ensure no double-entry on same bar.

---

### 6. Hougaard ASRS — IBUST100 (BT-4-S5)
**Signal:** 4th 5-min bar at 09:45 ET (fallback to 5th bar if range < 5 pts)
**Entry:** OCO bracket ±2 pts
**Stop:** bar range + 4 pts (symmetric)
**Exit:** 2-bar trailing stop on 5-min bars or 16:00 ET
**Automation instrument:** MNQ

| Direction | N | Win% | EV (pts) | Sharpe |
|-----------|---|------|----------|--------|
| Long | 761 | 46.4% | +3.51 | +0.071 |
| Short | 783 | 40.2% | +1.73 | +0.031 |
| Overall | 1544 | 43.3% | +2.61 | +0.049 |

Best sub-segments: Tuesday (+7.48), wide bars >25 pts (+3.05).

---

### 7. Hougaard SRS — IBUST100 (BT-4-S6)
**Signal:** 2nd 15-min bar at 09:45 ET *(confirmed near-optimal by candle scan)*
**Entry:** OCO bracket ±2 pts
**Stop:** 20% of 14-day daily ATR
**Exit:** 2-bar trailing stop on 15-min bars or 16:00 ET
**Automation instrument:** MNQ

| Direction | N | Win% | EV (pts) | Sharpe |
|-----------|---|------|----------|--------|
| Long | 767 | 48.8% | +4.60 | +0.071 |
| Short | 775 | 39.7% | +2.72 | +0.036 |
| Overall | 1542 | 44.2% | +3.66 | +0.052 |

Best sub-segments: Friday (+9.22), Tuesday (+6.74), wide bars.

---

### 8. ORB — IBDE40 (BT-5-S2)
**Session:** 09:00–17:30 Frankfurt
**Entry offset:** 1 pt beyond ORB high/low
**Stop:** opposite ORB side
**Target:** 2R
**Automation instrument:** FDXS

| Window | Direction | N | EV (pts) | Sharpe |
|--------|-----------|---|----------|--------|
| 15m | Long | 1241 | +0.12 | +0.001 |
| 15m | Short | 1214 | +2.12 | +0.024 |
| 30m | Long | 1143 | +2.31 | +0.025 |
| 30m | Short | 1123 | +0.58 | +0.006 |

30m long and 15m short are the strongest variants.

---

### 9. ORB — IBUST100 (BT-5-S1)
**Session:** 09:30–16:00 ET
**Entry offset:** 0.25 pts
**Stop:** opposite ORB side
**Target:** 2R
**Automation instrument:** MNQ

| Window | Direction | N | EV (pts) | Sharpe |
|--------|-----------|---|----------|--------|
| 15m | Long | 1249 | +3.16 | +0.037 |
| 15m | Short | 1188 | +1.14 | +0.012 |
| 30m | Long | 1140 | +2.39 | +0.025 |
| 30m | Short | 1063 | +0.89 | +0.008 |

---

### 10. Hougaard FOMC Rule of 4 — IBUS500
**Signal:** 4th 10-min bar after 14:00 ET FOMC announcement
**Entry:** OCO bracket ±2 pts
**Exit:** 2-bar trailing stop or 17:00 ET
**Frequency:** ~10 events/year
**Automation instrument:** MES

| Group | N | Win% | EV (pts) | Sharpe |
|-------|---|------|----------|--------|
| FOMC | 48 | 43.8% | +3.76 | +0.110 |
| Non-FOMC control | 1487 | 29.5% | -2.08 | -0.191 |

Edge is event-specific. Calendar-driven — low frequency, requires FOMC date feed.

---

### 11. Following Range Break — Long (BT-1)
**Best variant:** `02_pct` (0.2% confirmation above range) on 5m/10m bars
**Direction:** Long only (short has negative net EV across all instruments)
**Instruments:** IBDE40, IBUS500, IBUST100, IBJP225 (strongest), IBUS30

| Symbol | TF | Net EV% | Go/No-go |
|--------|----|---------|----------|
| IBJP225 | 5m | +0.045% | Go |
| IBUST100 | 5m | +0.029% | Go |
| IBUS500 | 5m | +0.020% | Go |
| IBDE40 | 5m | +0.018% | Go |
| IBUS30 | 5m | +0.020% | Go |
| IBGB100 | 5m | +0.013% | Pilot |

---

### 12. Noon Iron Butterfly — SPX/IBUS500
**Entry:** 12:00 ET; wings = 5% ATM
**Exit:** 16:00 ET (session close)
**Automation instrument:** MES (options not available in micro — requires SPX options or SPY)

| IV bucket | N | Win% | EV (pts) | Sharpe |
|-----------|---|------|----------|--------|
| Low IV | 355 | 67.9% | +2.40 | +0.183 |
| Normal IV | 364 | 60.4% | +1.92 | +0.115 |
| High IV | 355 | 71.8% | +6.69 | +0.278 |

> Caveat: IV proxy is HV20. True SPX options require separate IV data and a brokerage
> that supports 0-1 DTE options. Cannot automate via IBKR micro futures.

---

## No-Go Strategies

| Strategy | Instrument | EV (pts) | Sharpe | Primary reason |
|----------|------------|----------|--------|----------------|
| Hougaard 1BP/1BN | IBGB100 | -2.06 to -3.72 | -0.084 to -0.192 | No edge at first bar |
| Hougaard ASRS | IBDE40 | -0.27 | -0.007 | Marginal; short-only edge too small |
| Hougaard ASRS | IBUS500 | -1.39 | -0.116 | No edge on S&P at this time |
| Hougaard SRS | IBGB100 | -0.53 | -0.024 | No edge on FTSE; confirmed by scan (best: Sharpe -0.020) |
| Hougaard SRS | IBUS500 | -1.34 | -0.086 | No edge on S&P; confirmed by scan (best: Sharpe -0.019) |
| OCO bracket (any bar) | IBGB100 | best: -0.45 | -0.020 | Scan confirms no viable bar exists (96 combos) |
| OCO bracket (any bar) | IBUS500 | best: -0.36 | -0.019 | Scan confirms no viable bar exists (96 combos) |
| OCO bracket (any bar) | IBEU50 | best: -1.01 | -0.074 | EuroStoxx 50 lacks opening direction despite shared session with IBDE40 |
| OCO bracket (any bar) | IBFR40 | best: +0.56 | +0.021 | Edge too weak to trade; not reproducible at meaningful Sharpe |
| OCO bracket (any bar) | IBCH20 | best: +0.71 | +0.020 | Edge too weak; SMI lacks sufficient open momentum |
| OCO bracket (any bar) | IBNL25 | best: -1.94 | -0.652 | AEX point range too small; 2-pt offset is disproportionate |
| OCO bracket (any bar) | USGOLD | best: -2.29 | -0.268 | No directional opening breakout edge on gold |
| ORB 15m/30m | IBGB100 | -2.88 to -3.73 | -0.074 to -0.108 | Structurally no edge |
| ORB 15m/30m | IBUS500 | -1.18 to -2.25 | -0.054 to -0.115 | Fill rate high, edge absent |
| Micro/Macro Trend | IBDE40 | +3.07 | +0.512 | Low Sharpe; full-session monitoring; costs erode edge |
| 0DTE Iron Condor | IBUS500 | -0.93 | -0.859 | HV20 understates IV; credits too small; negative overall |
| Noon Iron Butterfly | IBDE40/ESTX50 | +4.18 / +1.22 | +0.054 / +0.032 | Tail risk too large (-832 pts P5 for DAX) |

> IBGB100 has no viable intraday edge across any tested strategy family.
> IBUS500 has no viable OCO bracket edge at any bar or timeframe.
> IBEU50, IBNL25, USGOLD: no OCO bracket edge confirmed across expanded scan.

---

## Unfit — Moved to `backtests/unfit/`

| File | Strategy | Reason |
|------|----------|--------|
| `hougaard_1bp.py` | Hougaard 1BP/1BN on IBGB100 | All variants No-go |
| `micro_macro_trend_bt.py` | Micro/Macro Trend PoC | Researcher score 10/20; costs erode edge |

---

## Filter Signals (from FILTER_SIGNALS.md)

### Weekday directional bias
No actionable edge. hc/lc probabilities are uniformly high (80–97%) across all
weekday × structure combinations. **Do not apply a weekday directional filter.**

### PDC proximity
PDC is most reliably tested in the **first 30-min window** of each session.
Median distance at open is 0.06–0.18% vs 0.30–0.60% at mid-session.

**Rule:** Entry signals in the first 30 min are higher confidence. After 60 min,
require PDC to have already been crossed before taking a BT-4/BT-5 signal.

### Post-extreme-day drift
After a down extreme day (|ret| > 2%): positive mean 2-4 week forward drift across
all 7 instruments.

**DRIFT rule:** Enter short-puts within 2 days of a down extreme day.
Avoid adding DRIFT positions within 1 week of an up extreme day.

### Dealer Gamma Regime (DPM-02)
HV20 percentile is too noisy as a GEX proxy. Direction match is near-random across
low/mid/high HV regimes for both IBUS500 and IBDE40.
**No filter value — do not apply.**

---

## Automation Target List

Strategies cleared for automation via IBKR Gateway (micro futures, bracket orders).
Signal bars marked † reflect candle scan upgrades.

| Strategy | Instrument | Micro future | Signal time | Sharpe | Frequency |
|----------|------------|-------------|-------------|--------|-----------|
| VWAP Extrema (high-edge slots) | IBDE40 | FDXS | Frankfurt open + close | +0.524 | ~15,000/yr |
| VWAP Extrema (high-edge slots) | IBUST100 | MNQ | Pre-market + open | +0.520 | ~11,000/yr |
| VWAP Extrema (high-edge slots) | IBUS500 | MES | Pre-market + open | +0.444 | ~13,500/yr |
| OCO Early Session Break | IBUS30 | MYM | 09:45 ET (bar closes; 3rd 5m bar) | +0.098 | ~250/yr |
| OCO Opening Bar 30m | IBJP225 | FN225M† | 10:00 Tokyo (2nd 30m bar closes) | +0.075 | ~250/yr |
| Hougaard SRS ‡ | IBDE40 | FDXS | 09:45 Frankfurt (3rd 15m bar closes) | +0.054 | ~250/yr |
| OCO Opening Bar 30m | IBUST100 | MNQ | 10:00 ET (1st 30m bar closes) | +0.067 | ~250/yr |
| Hougaard ASRS | IBUST100 | MNQ | 09:50 ET (4th 5m bar closes) | +0.049 | ~250/yr |
| Hougaard SRS | IBUST100 | MNQ | 10:00 ET (2nd 15m bar closes) | +0.052 | ~250/yr |
| ORB 30m long | IBDE40 | FDXS | 09:30 Frankfurt | +0.025 | ~550/yr |
| ORB 15m/30m | IBUST100 | MNQ | 09:45/10:00 ET | +0.037 | ~1,100/yr |
| Hougaard FOMC | IBUS500 | MES | FOMC announcement days | +0.110 | ~10/yr |

> † FN225M (Micro Nikkei) is listed on OSE.JPN and is JPY-denominated. Requires
> evaluation of Japan futures exchange access before deployment. Alternatively,
> pilot on the full Nikkei 225 CFD (IBJP225) via IBKR CFD if micro not accessible.
>
> ‡ Signal times corrected from SUMMARY v1: scheduler fires at bar *close* + 30s,
> not at bar *open*. FDXS SRS bar opens 09:30, closes 09:45 → fires 09:45:30 Frankfurt.
> MNQ strategies: OCO Opening Bar 30m closes 10:00 ET, ASRS closes 09:50 ET.
>
> IBUS30 conflict: the 3rd 5m bar (09:40–09:45 ET) overlaps with the ASRS bar on MNQ.
> Both fire at 09:45:30 ET but on different instruments — no conflict.
>
> MNQ 10:00 ET conflict: OCO Opening Bar 30m and SRS both fire simultaneously.
> Conflict resolver handles same-instrument same-time signals (see engine design).
>
> All micro futures: IBKR native trailing stop (orderType=TRAIL) survives disconnects.
> Position sizing: single contract per instrument.

---

## Research Gaps

All instrument × strategy combinations have now been tested. No known gaps remain
before beginning execution engine implementation.

| Gap | Status |
|-----|--------|
| IBDE40 ORB | Filled 2026-04-21 |
| IBGB100 Hougaard SRS | Filled 2026-04-21 |
| IBUS500 + IBUST100 Hougaard ASRS/SRS | Filled 2026-04-21 |
| IBGB100 ORB | Filled 2026-04-21 |
| Candle scan (optimal bar search) | Completed 2026-04-21 |

---

## Next Steps

1. **Execution engine** — `finance/execution/` module
   - IBKR Gateway connection (ib_insync, port 4001 live / 4002 paper)
   - APScheduler for timezone-aware strategy scheduling
   - Bracket order placement with hard stops (transmit=True)
   - Risk module: single contract, max daily loss per instrument
   - IBKR Flex Query integration for automated P&L sync to tradelog

2. **Strategy priority for live deployment**
   - Start with Hougaard SRS on FDXS — one signal/day, 09:45 Frankfurt, defined stop
   - Add OCO Opening Bar 30m on MNQ — 10:00 ET, same infrastructure
   - Add OCO Early Session Break on MYM — 09:45 ET, highest scan Sharpe (+0.098)
   - Add IBUST100 ASRS/SRS (same engine, 09:50 ET / 10:00 ET)
   - Add OCO Opening Bar 30m on FN225M — 10:00 Tokyo; requires Japan futures access assessment first
   - VWAP Extrema last (highest complexity — requires real-time VWAP calculation)
