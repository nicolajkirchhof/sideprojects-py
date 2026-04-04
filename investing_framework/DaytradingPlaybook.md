# Daytrading Playbook

Intraday strategies on index futures and liquid stocks — automation candidates.

> This playbook is a **research and automation** document. These strategies are NOT part of the active discretionary trading system. They are documented here as profit mechanisms to evaluate, backtest, and potentially automate. Manual daytrading has a 97% failure rate (Chague et al., 2020). The only path to profitable daytrading is systematic execution with zero discretion.

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

**Implication:** Manual daytrading is negative expected value for retail. The only viable approach is to identify specific, quantifiable intraday profit mechanisms and automate their execution — removing the human from the loop entirely.

---

## Profit Mechanisms — Intraday

### DPM-01 — Intraday Momentum (ORB + News) *(Grade B · RESEARCH)*

> Stocks reacting to fundamental news exhibit persistent intraday momentum after the opening range is established. The first 30 minutes set the direction for the day.

| Attribute | Detail |
|-----------|--------|
| Market effect | News-driven stocks trend intraday after the opening range breakout |
| Why it works | Information diffusion takes hours, not minutes. Institutional algorithms react to news systematically, creating a sustained directional flow. |
| Academic basis | ORB methodology (Kullamägi); News-driven intraday momentum (SSRN 2024); Boudoukh et al. 2013 (news vs no-news continuation) |
| Timeframe | 30 min to 6 hours (open to close) |
| Signals | Stock gaps >3% on news catalyst · ORB above 15/30min candle high · RVOL > 2× by 9:45 |
| Structure | Long/short futures or stock. Entry: break of ORB candle. Stop: below ORB low. Target: 2× risk or EOD. |
| Automation | Fully automatable — gap screen + ORB trigger + hard stop + time-based exit |
| Research gap | Backtest ORB breakout win rate by gap size, RVOL, and news presence on ES/NQ |

### DPM-02 — Dealer Gamma Regime (GEX-Based) *(Grade B · RESEARCH)*

> When dealers are short gamma, intraday directional moves are amplified. When long gamma, moves are dampened. GEX predicts intraday realized vol better than VIX.

| Attribute | Detail |
|-----------|--------|
| Market effect | Dealer delta-hedging amplifies (short gamma) or suppresses (long gamma) intraday moves |
| Why it works | Dealers must hedge their options book. When short gamma, they buy into rallies and sell into drops — amplifying the move. When long gamma, the reverse — they dampen moves. *(SqueezeMetrics; Pearson et al. 2007; Zarattini et al. 2024)* |
| Timeframe | Intraday — signal is daily GEX reading, execution is intraday |
| Signals | GEX negative → expect amplified moves, trade momentum. GEX positive → expect mean reversion, fade moves. |
| Structure | GEX negative days: ORB momentum on ES/NQ. GEX positive days: fade extremes, sell premium (0DTE strangles). |
| Automation | Automatable — GEX data available pre-market, conditions binary |
| Research gap | Backtest: momentum strategy returns on negative GEX days vs positive GEX days on ES |

### DPM-03 — S/R Swing (Price Action at Key Levels) *(Grade B · RESEARCH)*

> Price reverses at key structural levels — prior day high/low/close, overnight high/low, weekly levels. These are institutional reference points.

| Attribute | Detail |
|-----------|--------|
| Market effect | Price reacts at structurally significant levels where institutional orders cluster |
| Why it works | Institutional algorithms anchor to prior session reference points (PDH, PDL, PDC, ONH, ONL, PWH, PWL). Retail stops cluster just beyond these levels, providing liquidity for reversals. |
| Academic basis | Empirical observation (original daytrading notes); pivot point trading (Person 2006); Al Brooks price action methodology |
| Timeframe | Minutes to hours |
| Signals | Price reaches key level (PDH/PDL/PDC/ONH/ONL) · forms reversal pattern (Elephant Bar, doji, engulfing) · volume spike at level · VWAP supports direction |
| Entry rules | *(From your original playbook)*: At least two medium-sized bars close fully through the prior candle in counter-trend direction. Wicks larger on counter-trend side. Wait for bars to fully form. |
| Stop | Below the last S/R level. Move with each new S/R area — not just to break-even. |
| Exit signals | Doji with large wick in trend direction · body/ATR shrinking · counter-trend candles forming · VWAP forms a top/bottom · 20 EMA flattens · 2R reached |
| Automation | Partially automatable — level identification is mechanical, but pattern recognition at levels requires judgment. Best suited for semi-automated alerts + manual confirmation. |

### DPM-04 — Overnight Reversal *(Grade B · RESEARCH)*

> Sell-offs create robust positive overnight returns. The largest positive equity returns accrue between 2–3 AM ET (European market open).

| Attribute | Detail |
|-----------|--------|
| Market effect | End-of-day order imbalances from forced selling resolve overnight. European institutional buying absorbs the discount. |
| Why it works | Forced sellers (margin calls, risk limits) create end-of-day excess supply. Overnight, this supply is absorbed at a discount. *(Boyarchenko, Larsen, Whelan 2023: 3.6% annualized overnight drift)* |
| Timeframe | Close-to-open (16:00 to 9:30 next day) |
| Signals | SPY/ES down >1% intraday on above-avg volume · NOT driven by fundamental shift · VIX spike is intraday only (not multi-day trend) |
| Structure | Buy ES futures MOC. Exit at next-day open or 9:45 if gap is positive. |
| Automation | Fully automatable — conditions are quantitative, execution is mechanical |
| Research gap | Backtest: overnight returns on ES after >1% down days, filtered by VIX regime and news presence |

### DPM-05 — VWAP Mean Reversion *(Grade C · RESEARCH)*

> Price oscillates around VWAP. Extreme deviations revert. The further from VWAP, the stronger the reversion pull.

| Attribute | Detail |
|-----------|--------|
| Market effect | VWAP acts as a gravitational center. Institutional execution benchmarks to VWAP, creating a self-reinforcing anchor. |
| Why it works | Institutional algorithms target VWAP fills. When price deviates significantly, algos shift order flow to pull price back. |
| Academic basis | Empirical (VWAP as institutional benchmark); your original daytrading notes (sentiment trade based on VWAP) |
| Timeframe | Minutes to hours |
| Signals | Price > 2 standard deviations from VWAP · declining momentum (smaller candles, wicks growing) · volume fading at extreme |
| Structure | Fade the deviation. Entry: counter-trend at ±2 SD from VWAP. Stop: beyond the extreme. Target: VWAP or ±1 SD. |
| Automation | Automatable — VWAP calculation is mechanical, entry/exit rules are quantitative |
| Research gap | Backtest: mean reversion from ±2 SD VWAP on ES/NQ. Win rate, optimal SD threshold, time-of-day filter. |

### DPM-06 — 0DTE Variance Risk Premium *(Grade B · RESEARCH)*

> 0DTE options carry an elevated variance risk premium. Retail systematically overpays. Selling 0DTE premium is structurally profitable but requires rigorous risk management.

| Attribute | Detail |
|-----------|--------|
| Market effect | 0DTE implied vol exceeds realized vol even more than longer-dated options |
| Why it works | Retail demand (>75% of SPX option volume is now 0DTE) inflates prices. Bid-ask spreads are wide. Sellers capture the VRP + spread. *(Goldman Sachs 0DTE data 2023; Beckmeyer et al. 2023; Vilkov 2024)* |
| Timeframe | Same-day (0DTE — open to close) |
| Signals | IVP elevated on SPX 0DTE vs historical. GEX positive (dealer gamma dampens moves = low realized vol for sellers). No FOMC/CPI today. |
| Structure | Sell 0DTE SPX iron condor or strangle at 10:00 after morning vol settles. Strikes at ±1 SD expected move. Close at 50% profit or 15:30 whichever first. |
| Risk | **Extreme tail risk.** A single gap event can wipe out months of premium. Hard stop at 200% of credit. Never more than 2% of account per trade. |
| Automation | Fully automatable — quantitative entry, mechanical management |
| Research gap | Backtest 0DTE IC on SPX: win rate by GEX regime, VIX level, day-of-week |

### DPM-07 — Scalping Range-Bound Consolidation *(Grade C · RESEARCH)*

> When price trades in a defined intraday range, buy the low and sell the high. Works in the dead zone (10:30–14:30) when trending strategies fail.

| Attribute | Detail |
|-----------|--------|
| Market effect | Intraday consolidation after morning trend creates a defined range. Institutional TWAP/VWAP execution absorbs supply at range boundaries. |
| Why it works | After the morning impulse, institutions spread remaining orders across the day. Price oscillates in a narrow range until the last-hour impulse. |
| Timeframe | Minutes (10:30–14:30 "dead zone") |
| Signals | *(From your original playbook)*: Defined trading range with overlapping candles (≥25-50% overlap). At least three bounces between range boundaries. |
| Entry | At range boundary in trend direction. SL: just below range. Target: opposite boundary. Move SL to BE at 50% of range. |
| Automation | Automatable — range detection is quantitative, entry/exit is mechanical |
| Research gap | Backtest: intraday range detection on ES/NQ. Win rate by range width, time of day, and prior morning trend strength. |

---

## Global Rules for Automated Daytrading

These rules apply across all intraday strategies. Non-negotiable.

| Rule | Detail | Source |
|------|--------|--------|
| **No manual overrides during session** | If the system is running, you don't touch it. Overrides destroy edge. | 97% day trader failure rate |
| **Entries only after candle close** | No developing-bar entries. Same rule as swing trading, applied to intraday bars. | Your original rules |
| **Same idea max 2 executions** | If a setup triggers twice and fails twice, stop for the day. | Your original rules |
| **Trades against trend only at key S/R** | Counter-trend only at PDH/PDL/PDC/ONH/ONL with immediate BE stop. | Your original rules |
| **Max 0.5% risk per trade** | Hard stop. No exceptions. No "letting it breathe." | Portfolio management |
| **Max 3 trades per day** | Overtrading is the primary killer of daytrading accounts. | Research consensus |
| **No trading around FOMC/CPI/NFP** | Binary events destroy intraday edge. Flat before announcement. | Your original rules |

---

## Automation Development Roadmap

| Phase | Focus | Deliverable |
|-------|-------|-------------|
| 1 | **Backtest DPM-01 (ORB) and DPM-04 (Overnight Reversal)** | Win rate, Sharpe, max DD on ES/NQ. If positive → proceed. |
| 2 | **Backtest DPM-02 (GEX Regime)** | Does GEX filtering improve DPM-01 and DPM-06 returns? |
| 3 | **Build execution engine** | Bracket order automation on IBKR TWS API. Paper trade for 30 days. |
| 4 | **Live test DPM-01 + DPM-04** | Minimum capital. Log everything. 60 trades before scaling. |
| 5 | **Add DPM-06 (0DTE) if Phase 1–4 profitable** | Highest premium but highest tail risk. Only after proven execution infrastructure. |

---

## Key Reference — From Your Original Notes

### Preserved from `Daytrading (Automated)_orig.md`

**S/R Swing Entry Criteria:**
- At least two medium-sized bars (or one large) close fully above last candle in opposite direction
- Two bars have almost no overlay and only small wicks
- Wicks larger on counter-trend direction than trend direction
- Reversal at key level: ONH, ONL, PDH, PDL, PDC, PWH, PWL
- Overlaying trend (gap, overnight range, prior days) supports direction
- Wait for bars to fully form — never enter on a developing bar

**S/R Swing Stop Placement:**
- Move SL with every major S/R area — not just to break-even
- Below last low in bull trend, above last high in bear trend
- Use S/R levels for stops, not arbitrary distances

**S/R Swing Exit Signals:**
- Doji with large wig in trend direction
- Consecutive candles with large wigs in trend direction
- Body/ATR shrinking
- Counter-trend candles forming
- VWAP forms top/bottom
- 20 EMA goes neutral
- 2R reached

**Scalping Rules:**
- Defined trading range with overlapping candles
- Entry at range boundary in trend direction
- SL to BE at 50% of candle height in your favor
- Target: opposite boundary of range

**Mean Reversion on News:**
- Wait for initial push to fade
- Enter with loose SL
- Apply trailing SL on reversal
