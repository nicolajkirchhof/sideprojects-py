# Breakout Strategy

**A rules-based system synthesizing Minervini · Kullamägi · Kell · Velez · Bruzzese**

| SPEC | |
|---|---|
| HOLD PERIOD | 5–50 days |
| MAX RISK / TRADE | 0.5% portfolio |
| ENTRY WINDOW | ORB 15/30 min |

## 01 — Market Context First

Before scanning a single stock — know the regime you're trading in

### Market Regime — The Non-Negotiable Filter

> "The wind must be at your back. A great stock in a bad market is a losing trade." — Bruzzese

**✓ GO — CONDITIONS MET**

- SPY / QQQ above 50d & 200d SMA — Bruzzese: wind at your back
- 200d SMA is sloping upward — Minervini Trend Template Stage 2
- VIX below 20 or falling after spike — Low fear = momentum-friendly
- Advancing stocks > Declining stocks — Breadth confirms the move
- Sector of candidate shows RS vs SPY — Bruzzese: sector confirmation
- QQQ / SPY ratio stable or rising — Tech leadership intact

**✗ NO-GO — REDUCE / STOP**

- SPY below 200d SMA — Bear market — longs will fail
- VIX > 30 or spiking sharply — Volatility kills breakouts
- Declining > Advancing by 2:1+ — Broad market distribution
- 3+ failed breakouts in your list — Market rejecting setups
- Major macro event in 48h — FOMC / CPI / NFP = binary risk
- Your last 3 trades stopped out — Slow down — regime has shifted

## 02 — Stock Scanning & Selection

Barchart-based screener → quality filter → watchlist of max 20

### Barchart Scan — Universe Filters & Sort Columns

**BASE FILTERS (always on)**

| Filter | Value | Reason |
|---|---|---|
| Avg Volume | > 1,000,000 | Liquidity — options must be tradeable |
| Price | > $3.00 | Avoid penny stocks |
| No earnings | within 5 days | No binary events on entries |

**SCAN SORT COLUMNS — Run each separately, build watchlist**

| Column | Rationale |
|---|---|
| 5d Chg % | Momentum leaders this week — who is already moving |
| 1M Chg % | Sustained strength — not a 1-day wonder |
| 52W High | Near all-time high = Stage 2 uptrend, no overhead supply |
| RVOL Leaders | Today's relative volume spike — something is happening now |
| High Call Vol | OTM call buying = informed money anticipating upside |
| High Put Ratio | Unusual put activity = potential short squeeze fuel |
| OI vs Vol OTM | Option vol >> open interest in next 4W = directional bet |

**SCANNING PROCESS**

1. Run each scan — 5d, 1M, RVOL, Call vol, Put ratio
2. Apply base filters — Vol >1M, Price >$3, No earnings <5d
3. Trend template check — Price > 50d > 200d, 200d sloping up
4. RS check vs SPY — Is it outperforming or holding on dips?
5. Chart quality check — Tight base? VCP? Clean structure?
6. Add to watchlist — Max 20 names, Prioritize top 5

### Stock Selection Checklist — All 5 boxes must be checked

A stock only goes on the shortlist if it passes every filter. One fail = skip it.

**01 — TREND TEMPLATE (Minervini)**

- Price > 50d SMA > 200d SMA
- 200d SMA trending upward (not flat)
- Price within 25–30% of 52-week high
- 52W high scan: near or at multi-year high = no overhead supply

**02 — RELATIVE STRENGTH (Bruzzese)**

- RS line at or near new highs vs SPY
- Stock holds up / rises when SPY dips
- Outperforming sector peers in last 1M
- Higher lows while SPY makes lower lows = institutional support

**03 — BASE QUALITY (Minervini + Kell + Kullamägi)**

- Consolidation of at least 1–3 weeks (tight range)
- Volume contracting during the base (VDU — dry-up)
- ATR 0–6 range (not extended / parabolic)
- ADR > 3% — stock has the daily range to deliver 2R+ in your timeframe
- 10 EMA > 20 EMA > 50 SMA — stacked and sloping up

**04 — CATALYST / NARRATIVE (Camillo + PEAD)**

- Is there a reason for this move? Earnings beat, news, sector theme
- PEAD: positive earnings surprise + stock closed top 25% of range
- Social / sector narrative gaining traction (not already mainstream)
- No binary event within 5 days of planned entry

**05 — RISK PARAMETERS (Universal)**

- Stop loss defined before entry — below base low / ORB low
- Stop ≤ 7% from entry (Minervini hard limit)
- Position size: 0.5% max loss = (0.5% portfolio) ÷ (entry − stop)
- R:R ≥ 2:1 minimum — if target doesn't justify risk, skip

**GOLDEN RULE:** Never trade without a defined SL. Rules. Not feelings.

### Setup Quality — ATR Extension & Entry Types

**ATR EXTENSION TABLE**

| ATR Range | Stage | Action |
|---|---|---|
| 0 – 3 ATR | Early / Stable Trend | Sweet spot. Sustainable entry. |
| 3 – 6 ATR | Established Trend | Strong momentum. Diminishing R:R. |
| 7 – 10 ATR | Extended Trend | Climax risk. Only partial entries. |
| > 10 ATR | Parabolic / Exhausted | High reversion risk. Avoid entries. |

**TYPE A — Episodic Pivot (EP)** — Kullamägi

> Why it works: Institutions can't deploy capital in one day. The gap-up is day 1 of 40–60 days of forced accumulation. PEAD research confirms the drift duration.

- Gap up ≥ 10% (prefer 15%+) on 5–10× avg volume
- Fundamental catalyst: earnings beat, major news
- Stock closes top 25% of day's range — institutions buying
- Entry: ORB 15/30 min high next day, or gap-day close
- Highest velocity — biggest winners come from EPs

**TYPE B — VCP Breakout** — Minervini

> Why it works: Each pullback shakes out weak holders. When supply is fully absorbed, even modest buying drives price explosively — the line of least resistance is higher.

- 2–6 weeks of tightening range, 3+ contraction points
- Volume dries up in final days (VDU = no sellers left)
- Breakout above pivot on 40–50%+ above avg volume
- Entry: ORB above 15/30 min candle on breakout day
- Lower velocity than EP but higher reliability

**TYPE C — EMA Reclaim / Wedge Pop** — Kell + Velez

> Why it works: Institutions add to winning positions on pullbacks to moving averages. The reclaim candle signals their bid has re-appeared — you're entering where Big Money is buying.

- Stock in uptrend pulls back to 10d or 20d EMA
- EMA crossback: price tags EMA then snaps back above
- Elephant Bar off EMA = institutional buying signal
- Entry: ORB above 15/30 min candle on reclaim day
- Lowest risk entry — tightest stop possible

> Prefer TYPE A > TYPE B > TYPE C in trending markets. In choppy markets, stick to TYPE B and C only.

## 03 — Entry Execution

ORB-based entries — first 30 min or last 30 min only

### ORB Entry Rules — Opening Range Breakout

You trade max 30–45 min of the first hour and last hour. The ORB is your only entry mechanism.

**TRADING WINDOW**

| Time | Event |
|---|---|
| 9:30 | Open — ORB starts |
| 9:45 | 15min candle — Type A/EP entry window |
| 10:00 | 30min candle — Preferred ORB level |
| 10:15 | Trading OFF — No new entries |
| 14:30 | Pre-close scan — Review watchlist |
| 15:15 | Last hour opens — Late-day entries |
| 15:45 | Entry deadline — Last valid ORB |
| 16:00 | Close — Log & review |

**MORNING ORB (9:30 – 10:15)**

- Wait for the 15min OR 30min candle to fully form — NEVER enter on a developing candle
- Entry trigger: buy stop limit just above the high of the 15min or 30min candle
- 15min candle → use for Type A EP setups (explosive gap + volume)
- 30min candle → use for Type B VCP and Type C EMA reclaim setups
- Volume on the breakout candle must be above the developing average for that time
- If SPY gaps down hard at open — skip morning entries, reassess at last hour

**LAST HOUR ORB (15:15 – 15:45)**

- Only enter if stock is in top 25% of day's range heading into last hour (Velez / Kullamägi)
- Late-day strength = institutional buying into close, not retail momentum
- Entry: above high of first 15min candle of the last hour
- Valid only if stock held near highs through the mid-day — no recovery plays
- Stop: below the last-hour opening candle low
- Strong preference: stock showing RS vs SPY in the final 2 hours

### Stop Loss Placement & Position Sizing

**STOP LOSS RULES**

| Rule | Detail |
|---|---|
| Primary SL | Below the ORB low (low of the 15/30min entry candle) |
| Maximum SL | 7% below entry — Minervini hard limit, non-negotiable |
| Base SL | Below the base/consolidation low (for VCP setups) |
| EMA SL | Below the 10d EMA (for EMA reclaim entries) |
| Intraday rule | Never move SL further away once placed. Move only to BE+ |
| Break-even rule | Move to break-even after first partial profit-take (1.5–2R) |
| Hard rule | NEVER add to a losing position. One SL hit = close. |

**POSITION SIZING FORMULA**

- Max $ risk per trade = 0.5% × Total Portfolio
- Shares / Contracts = Max $ risk ÷ (Entry − Stop)

**EXAMPLE**

| | |
|---|---|
| Portfolio value | $100,000 |
| Max risk (0.5%) | $500 |
| Entry price | $45.00 |
| Stop price (ORB low) | $42.50 |
| Risk per share | $2.50 |
| Shares to buy | 500 ÷ 2.50 = 200 shares |

## 04 — Trade Management & Exits

Scale out at 1.5–2R · Run the rest · Exit on MA/ATR signals

### Profit-Taking Rules & Exit Signals

**PROFIT-TAKING STAIRCASE**

| Stage | Action | Detail |
|---|---|---|
| ENTRY | Full size | Buy stop above ORB high. SL placed immediately. |
| 1.5 – 2R | −30–50% | Take 30–50% off. Move stop to break-even. Lock in income (Kullamägi 3–5 day rule). |
| > 4R | −30% more | Take another 30% off. Trail remaining stop using 5 EMA. |
| RUNNER | Last portion | Hold until an exit signal fires. Never let a 2R winner go to a loss. |

**EXIT SIGNALS — ANY ONE FIRES = CLOSE REMAINDER**

| Signal | Condition | Rationale |
|---|---|---|
| 5 MA Signal | Second consecutive daily close below the 5 EMA | Short-term momentum broken — Kullamägi / your rule |
| ATR Candle | Single candle > 1.5× ATR(14) against your position | Abnormal selling pressure — institutional exit signal |
| 10 MA Signal | Daily close below the 10 EMA | Trend cycle weakening — Kell exit rule |
| 20 MA Signal | Daily close below the 20 EMA on above-avg volume | Swing cycle over — Kell / Kullamägi / Bruzzese |
| RS Breakdown | Stock starts underperforming SPY on down days | Institutional support withdrawn — Bruzzese signal |
| Narrative Break | Catalyst is negated — bad news, guidance cut | Camillo: once the story breaks, the trade is over |
| Extension Rule | Stock > 20% above 10d EMA (Kell exhaustion) | Begin scaling — don't chase. Start taking profits. |
| Time Stop | 50 days elapsed + thesis not playing out | Capital tied up = opportunity cost. Release and redeploy. |

### Daily Execution Checklist

**PRE-MARKET ~9:00**

- Check market regime: SPY/QQQ above key MAs?
- VIX level and direction
- Review watchlist: any gap-ups with volume catalyst?
- Check earnings calendar — remove any stocks reporting today/tomorrow
- Define ORB levels: yesterday's high/low and prior range
- Set buy stop limit orders above 15/30min ORB levels (don't enter manually)

**FIRST HOUR 9:30–10:15**

- Wait for 15min candle to form — DO NOT enter on developing candle
- Confirm volume: is RVOL building? Is breakout candle above avg vol?
- Check SPY behavior: is market supporting the move?
- Enter via pre-set buy stop limit if trigger hits
- At 10:15 — stop taking new positions. Monitor existing only.

**MID-DAY 10:15–15:15**

- No new entries. Manage open positions only.
- Check if open trades hit profit targets → execute partial exits
- Note stocks holding well / showing RS for last-hour review
- Update watchlist: remove failed setups, add new candidates

**LAST HOUR 15:15–15:45**

- Scan watchlist: which stocks are in top 25% of day's range?
- Confirm RS: is stock green while SPY is flat/down?
- Enter via ORB of first 15min candle of the last hour
- Hard deadline: no entries after 15:45

**AFTER CLOSE**

- Log all trades: entry, SL, target, result, notes
- Review exits: were signals followed? Any emotion-based decisions?
- Update watchlist for next day
- Check daily MAs on open positions: any exit signals forming?

## Golden Rules — Non-Negotiable

1. **MANAGE BY RULES, NOT FEELINGS** — Every decision — entry, exit, sizing — must be pre-defined. If it's not in the rulebook, don't do it.
2. **NEVER TRADE A DEVELOPING CANDLE** — The close counts. Wait for the full 15/30min candle. Entries on forming bars are forbidden.
3. **NEVER ADD TO A LOSING TRADE** — One stop-out closes the position entirely. Averaging down is the fastest way to blow an account.
4. **EXIT WHEN PREMISE IS BROKEN** — Don't adapt a failing trade. If your reason for being in is gone — get out. No exceptions.
5. **RESPECT YOUR TIMEFRAME** — You are a 5–50 day swing trader. No intraday management. No watching every tick. Set stops. Walk away.
6. **NEVER FIGHT THE TREND** — Only fade a trend with a proven strategy. Otherwise — never short a strong trend, never long a broken chart.
7. **THE MARKET IS RIGHT, YOU ARE NOT** — When 3+ consecutive setups fail — pause. The regime has shifted. Reassess before the next trade.
8. **PATIENCE IS THE EDGE** — All five traders wait for tight setups. High-probability trades require patience. Forcing trades destroys edge.

## Trader DNA — Who Contributed What

The rules in this system are not invented — they are distilled from proven practitioners.

**Mark Minervini** — SEPA® / VCP

- Trend Template (Price > 50d > 200d, 200d rising)
- VCP — Volatility Contraction Pattern entry
- Hard stop ≤ 7–8% from entry
- Pivot point breakout on 40–50%+ volume

**Kristjan Kullamägi** — Breakout / EP

- Episodic Pivot — 10%+ gap on 5–10× volume
- ORB entry (1/5/60 min opening range)
- 3–5 day rule: sell 1/3–1/2 early, let rest run
- 10/20 EMA trailing stop for the runner

**Oliver Kell** — Price Cycle

- EMA stack: 10 > 20 > 50 = perfect trend
- Wedge Pop entry signal
- Exhaustion at >20% above 10d EMA → take profits
- Exit when price closes below 20d EMA on volume

**Oliver Velez** — Institutional Footprint

- Elephant Bar = institutional ignition signal
- Tail Bar — wick stings MA and recovers
- 40/40/20: entry location, management, mindset
- Late-day validation: top 25% of range = Big Money

**Vincent Bruzzese** — Market-Relative

- RS/RW vs SPY — only long if stock holds on SPY dips
- Sector confirmation required for valid breakout
- Market regime filter: SPY above 50d & 200d
- Higher lows while SPY makes lower lows = buy
