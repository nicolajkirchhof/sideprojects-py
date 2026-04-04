---
marp: true
theme: default
paginate: true
header: 'Daytrading Playbook'
footer: 'Automation Research'
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

# Daytrading Playbook

Intraday strategies on index futures and liquid stocks — automation candidates.

> This playbook is a **research and automation** document. These strategies are NOT part of the active discretionary trading system. They are documented here as profit mechanisms to evaluate, backtest, and potentially automate. Manual daytrading has a **97% failure rate** (Chague et al., 2020). The only path to profitable daytrading is **systematic execution with zero discretion**.

| SPEC | |
|------|---|
| Timeframe | Intraday (minutes to hours) |
| Instruments | Index futures (ES, NQ, DAX, FTSE), SPY, QQQ |
| Execution | **Automated only** — no manual discretion during the session |
| Risk per trade | Max 0.5% of account |
| Status | Research / Automation development |

---

## The Case Against Manual Daytrading

The research is unambiguous:

| Finding | Source |
|---------|--------|
| 97% of individuals who day traded >300 days lost money | Chague, De-Losso, Giovannetti 2020 (Brazil futures) |
| Only 0.5% earned more than a bank teller's salary | Same study |
| HFT firms dominate through microsecond speed advantages | Baron, Brogaard, Hagstromer & Kirilenko 2017 |
| Retail loses $358,000/day on 0DTE options alone | Beckmeyer, Branger & Gayda 2023 |
| Top 1% of day traders have genuine persistent skill | Cross-Section of Speculator Skill (Taiwan data) |

**Implication:** Manual daytrading is negative expected value for retail. The only viable approach is to identify specific, quantifiable intraday profit mechanisms and **automate their execution** — removing the human from the loop entirely.

---

## DPM-01 — Intraday Momentum (ORB + News) *(Grade B)*

> Stocks reacting to fundamental news exhibit persistent intraday momentum after the opening range is established.

| Attribute | Detail |
|-----------|--------|
| Market effect | News-driven stocks trend intraday after ORB breakout |
| Why it works | Information diffusion takes hours, not minutes. Institutional algos react systematically, creating sustained directional flow. |
| Academic basis | ORB methodology (Kullamagi); News-driven intraday momentum (SSRN 2024); Boudoukh et al. 2013 |
| Timeframe | 30 min to 6 hours (open to close) |
| Signals | Stock gaps >3% on news catalyst · ORB above 15/30min candle high · RVOL > 2x by 9:45 |
| Structure | Long/short futures or stock. Entry: break of ORB candle. Stop: below ORB low. Target: 2x risk or EOD. |
| Automation | Fully automatable — gap screen + ORB trigger + hard stop + time-based exit |
| Research gap | Backtest ORB breakout win rate by gap size, RVOL, and news presence on ES/NQ |

---

## DPM-02 — Dealer Gamma Regime (GEX-Based) *(Grade B)*

> When dealers are short gamma, intraday moves are amplified. When long gamma, moves are dampened. GEX predicts intraday realized vol better than VIX.

| Attribute | Detail |
|-----------|--------|
| Market effect | Dealer delta-hedging amplifies (short gamma) or suppresses (long gamma) intraday moves |
| Why it works | Dealers must hedge their options book. Short gamma = buy rallies, sell drops (amplify). Long gamma = reverse (dampen). |
| Academic basis | SqueezeMetrics; Pearson et al. 2007; Zarattini et al. 2024 |
| Timeframe | Intraday — signal is daily GEX reading, execution is intraday |
| Signals | GEX negative → trade momentum. GEX positive → fade moves, mean reversion. |
| Structure | GEX negative: ORB momentum on ES/NQ. GEX positive: fade extremes, sell premium (0DTE strangles). |
| Automation | Automatable — GEX data available pre-market, conditions binary |
| Research gap | Backtest: momentum returns on negative vs positive GEX days on ES |

---

## DPM-03 — S/R Swing (Price Action at Key Levels) *(Grade B)*

> Price reverses at key structural levels — PDH/PDL/PDC, ONH/ONL, weekly levels. These are institutional reference points.

| Attribute | Detail |
|-----------|--------|
| Market effect | Price reacts at structurally significant levels where institutional orders cluster |
| Why it works | Institutional algos anchor to prior session reference points. Retail stops cluster beyond these levels, providing liquidity for reversals. |
| Academic basis | Pivot point trading (Person 2006); Al Brooks price action methodology |
| Signals | Price reaches key level · reversal pattern (Elephant Bar, doji, engulfing) · volume spike · VWAP supports direction |
| Entry | 2+ medium bars close fully through prior candle in counter-trend direction. Wicks larger on counter-trend side. Wait for bars to fully form. |
| Stop | Below last S/R level. Move with each new S/R area — not just to break-even. |
| Exit | Doji with large wick · body/ATR shrinking · counter-trend candles · VWAP top/bottom · 20 EMA flattens · 2R reached |
| Automation | Partially automatable — level identification mechanical, pattern recognition requires judgment |

---

## DPM-04 — Overnight Reversal *(Grade B)*

> Sell-offs create robust positive overnight returns. The largest positive equity returns accrue between 2-3 AM ET (European market open).

| Attribute | Detail |
|-----------|--------|
| Market effect | End-of-day order imbalances from forced selling resolve overnight |
| Why it works | Forced sellers (margin calls, risk limits) create EOD excess supply. Overnight, absorbed at a discount. *(Boyarchenko, Larsen, Whelan 2023: 3.6% annualized overnight drift)* |
| Timeframe | Close-to-open (16:00 to 9:30 next day) |
| Signals | SPY/ES down >1% intraday on above-avg volume · NOT driven by fundamental shift · VIX spike is intraday only |
| Structure | Buy ES futures MOC. Exit at next-day open or 9:45 if gap is positive. |
| Automation | Fully automatable — conditions are quantitative, execution is mechanical |
| Research gap | Backtest: overnight returns on ES after >1% down days, filtered by VIX regime and news presence |

---

## DPM-05 — VWAP Mean Reversion *(Grade C)*

> Price oscillates around VWAP. Extreme deviations revert. The further from VWAP, the stronger the reversion pull.

| Attribute | Detail |
|-----------|--------|
| Market effect | VWAP acts as a gravitational center. Institutional execution benchmarks to VWAP, creating a self-reinforcing anchor. |
| Why it works | Institutional algos target VWAP fills. When price deviates significantly, algos shift order flow to pull price back. |
| Academic basis | VWAP as institutional benchmark; empirical observation |
| Timeframe | Minutes to hours |
| Signals | Price > 2 SD from VWAP · declining momentum (smaller candles, wicks growing) · volume fading at extreme |
| Structure | Fade the deviation. Entry: counter-trend at +/-2 SD from VWAP. Stop: beyond the extreme. Target: VWAP or +/-1 SD. |
| Automation | Automatable — VWAP calculation is mechanical, entry/exit rules are quantitative |
| Research gap | Backtest: mean reversion from +/-2 SD VWAP on ES/NQ. Win rate, optimal SD threshold, time-of-day filter. |

---

## DPM-06 — 0DTE Variance Risk Premium *(Grade B)*

> 0DTE options carry an elevated variance risk premium. Retail systematically overpays. Selling 0DTE premium is structurally profitable but requires rigorous risk management.

| Attribute | Detail |
|-----------|--------|
| Market effect | 0DTE implied vol exceeds realized vol even more than longer-dated options |
| Why it works | Retail demand (>75% of SPX option volume is 0DTE) inflates prices. Sellers capture VRP + spread. *(Goldman Sachs 2023; Beckmeyer et al. 2023; Vilkov 2024)* |
| Timeframe | Same-day (0DTE — open to close) |
| Signals | IVP elevated on SPX 0DTE vs historical. GEX positive. No FOMC/CPI today. |
| Structure | Sell 0DTE SPX iron condor/strangle at 10:00. Strikes at +/-1 SD expected move. Close at 50% profit or 15:30. |
| Risk | **Extreme tail risk.** Single gap event can wipe months of premium. Hard stop at 200% of credit. Max 2% account per trade. |
| Automation | Fully automatable — quantitative entry, mechanical management |
| Research gap | Backtest 0DTE IC on SPX: win rate by GEX regime, VIX level, day-of-week |

---

## DPM-07 — Scalping Range-Bound Consolidation *(Grade C)*

> When price trades in a defined intraday range, buy the low and sell the high. Works in the dead zone (10:30-14:30) when trending strategies fail.

| Attribute | Detail |
|-----------|--------|
| Market effect | Intraday consolidation after morning trend creates a defined range. Institutional TWAP/VWAP execution absorbs supply at boundaries. |
| Why it works | After the morning impulse, institutions spread remaining orders across the day. Price oscillates in a narrow range until the last-hour impulse. |
| Timeframe | Minutes (10:30-14:30 "dead zone") |
| Signals | Defined trading range with overlapping candles (25-50% overlap). At least three bounces between range boundaries. |
| Entry | At range boundary in trend direction. SL: just below range. Target: opposite boundary. Move SL to BE at 50% of range. |
| Automation | Automatable — range detection is quantitative, entry/exit is mechanical |
| Research gap | Backtest: intraday range detection on ES/NQ. Win rate by range width, time of day, prior morning trend strength. |

---

## Global Rules for Automated Daytrading

These rules apply across all intraday strategies. **Non-negotiable.**

| Rule | Detail | Source |
|------|--------|--------|
| **No manual overrides during session** | If the system is running, you don't touch it. Overrides destroy edge. | 97% failure rate |
| **Entries only after candle close** | No developing-bar entries. Same rule as swing trading. | Original rules |
| **Same idea max 2 executions** | If a setup triggers twice and fails twice, stop for the day. | Original rules |
| **Trades against trend only at key S/R** | Counter-trend only at PDH/PDL/PDC/ONH/ONL with immediate BE stop. | Original rules |
| **Max 0.5% risk per trade** | Hard stop. No exceptions. No "letting it breathe." | Portfolio mgmt |
| **Max 3 trades per day** | Overtrading is the primary killer of daytrading accounts. | Research consensus |
| **No trading around FOMC/CPI/NFP** | Binary events destroy intraday edge. Flat before announcement. | Original rules |

---

## Automation Development Roadmap

| Phase | Focus | Deliverable |
|-------|-------|-------------|
| 1 | **Backtest DPM-01 (ORB) and DPM-04 (Overnight Reversal)** | Win rate, Sharpe, max DD on ES/NQ. If positive → proceed. |
| 2 | **Backtest DPM-02 (GEX Regime)** | Does GEX filtering improve DPM-01 and DPM-06 returns? |
| 3 | **Build execution engine** | Bracket order automation on IBKR TWS API. Paper trade for 30 days. |
| 4 | **Live test DPM-01 + DPM-04** | Minimum capital. Log everything. 60 trades before scaling. |
| 5 | **Add DPM-06 (0DTE) if Phase 1-4 profitable** | Highest premium but highest tail risk. Only after proven execution infrastructure. |

### Key Reference — Original Notes Preserved

**S/R Swing:** 2+ bars close fully through prior candle, wicks larger on counter-trend side, reversal at ONH/ONL/PDH/PDL/PDC/PWH/PWL, wait for bars to fully form.
**Stops:** Move SL with every major S/R area. Use S/R levels for stops, not arbitrary distances.
**Exits:** Doji with large wick, body/ATR shrinking, counter-trend candles, VWAP top/bottom, 20 EMA neutral, 2R reached.
**Scalping:** Defined range with overlapping candles, entry at boundary, SL to BE at 50%, target opposite boundary.
**Mean Reversion on News:** Wait for initial push to fade, enter with loose SL, apply trailing SL on reversal.
