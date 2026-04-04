---
marp: true
theme: default
paginate: true
header: 'Breakout Strategy'
footer: 'Minervini · Kullamägi · Kell · Velez · Bruzzese'
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

# Breakout Strategy

**A rules-based system synthesizing Minervini · Kullamägi · Kell · Velez · Bruzzese**

| SPEC | |
|---|---|
| HOLD PERIOD | 5-50 days |
| MAX RISK / TRADE | 0.5% portfolio |
| ENTRY WINDOW | ORB 15/30 min |

---

## 01 - Market Regime: GO / NO-GO

> "The wind must be at your back. A great stock in a bad market is a losing trade." - Bruzzese

**GO - CONDITIONS MET** | **NO-GO - REDUCE / STOP**
---|---
SPY / QQQ above 50d & 200d SMA | SPY below 200d SMA - Bear market
200d SMA sloping upward (Stage 2) | VIX > 30 or spiking sharply
VIX below 20 or falling after spike | Declining > Advancing by 2:1+
Advancing stocks > Declining stocks | 3+ failed breakouts in your list
Sector of candidate shows RS vs SPY | Major macro event in 48h (FOMC/CPI/NFP)
QQQ / SPY ratio stable or rising | Your last 3 trades stopped out

---

## 02 - Barchart Scan: Base Filters

Barchart-based screener -> quality filter -> watchlist of max 20

**BASE FILTERS (always on)**

| Filter | Value | Reason |
|---|---|---|
| Avg Volume | > 1,000,000 | Liquidity - options must be tradeable |
| Price | > $3.00 | Avoid penny stocks |
| No earnings | within 5 days | No binary events on entries |

---

## 02 - Barchart Scan: Sort Columns & Process

**SCAN SORT COLUMNS - Run each separately, build watchlist**

| Column | Rationale |
|---|---|
| 5d Chg % | Momentum leaders this week - who is already moving |
| 1M Chg % | Sustained strength - not a 1-day wonder |
| 52W High | Near ATH = Stage 2 uptrend, no overhead supply |
| RVOL Leaders | Today's relative volume spike - something is happening now |
| High Call Vol | OTM call buying = informed money anticipating upside |
| High Put Ratio | Unusual put activity = potential short squeeze fuel |
| OI vs Vol OTM | Option vol >> open interest in next 4W = directional bet |

**SCANNING PROCESS:** Run each scan -> Apply base filters -> Trend template check (Price > 50d > 200d) -> RS check vs SPY -> Chart quality check (tight base? VCP?) -> Add to watchlist (max 20, prioritize top 5)

---

## Checklist Box 1: Trend Template (Minervini)

A stock only goes on the shortlist if it passes **every** filter. One fail = skip it.

- Price > 50d SMA > 200d SMA
- 200d SMA trending upward (not flat)
- Price within 25-30% of 52-week high
- 52W high scan: near or at multi-year high = no overhead supply

---

## Checklist Box 2: Relative Strength (Bruzzese)

- RS line at or near new highs vs SPY
- Stock holds up / rises when SPY dips
- Outperforming sector peers in last 1M
- Higher lows while SPY makes lower lows = institutional support

---

## Checklist Box 3: Base Quality (Minervini + Kell + Kullamägi)

- Consolidation of at least 1-3 weeks (tight range)
- Volume contracting during the base (VDU - dry-up)
- ATR 0-6 range (not extended / parabolic)
- ADR > 3% - stock has the daily range to deliver 2R+ in your timeframe
- 10 EMA > 20 EMA > 50 SMA - stacked and sloping up

---

## Checklist Box 4: Catalyst / Narrative (Camillo + PEAD)

- Is there a reason for this move? Earnings beat, news, sector theme
- PEAD: positive earnings surprise + stock closed top 25% of range
- Social / sector narrative gaining traction (not already mainstream)
- No binary event within 5 days of planned entry

---

## Checklist Box 5: Risk Parameters (Universal)

- Stop loss defined before entry - below base low / ORB low
- Stop <= 7% from entry (Minervini hard limit)
- Position size: 0.5% max loss = (0.5% portfolio) / (entry - stop)
- R:R >= 2:1 minimum - if target doesn't justify risk, skip

**GOLDEN RULE:** Never trade without a defined SL. Rules. Not feelings.

---

## ATR Extension Table

| ATR Range | Stage | Action |
|---|---|---|
| 0 - 3 ATR | Early / Stable Trend | Sweet spot. Sustainable entry. |
| 3 - 6 ATR | Established Trend | Strong momentum. Diminishing R:R. |
| 7 - 10 ATR | Extended Trend | Climax risk. Only partial entries. |
| > 10 ATR | Parabolic / Exhausted | High reversion risk. Avoid entries. |

---

## Setup Type A: Episodic Pivot (Kullamägi)

> Why it works: Institutions can't deploy capital in one day. The gap-up is day 1 of 40-60 days of forced accumulation. PEAD research confirms the drift duration.

- Gap up >= 10% (prefer 15%+) on 5-10x avg volume
- Fundamental catalyst: earnings beat, major news
- Stock closes top 25% of day's range - institutions buying
- Entry: ORB 15/30 min high next day, or gap-day close
- **Highest velocity - biggest winners come from EPs**

> Prefer TYPE A > TYPE B > TYPE C in trending markets. In choppy markets, stick to TYPE B and C only.

---

## Setup Type B: VCP Breakout (Minervini)

> Why it works: Each pullback shakes out weak holders. When supply is fully absorbed, even modest buying drives price explosively - the line of least resistance is higher.

- 2-6 weeks of tightening range, 3+ contraction points
- Volume dries up in final days (VDU = no sellers left)
- Breakout above pivot on 40-50%+ above avg volume
- Entry: ORB above 15/30 min candle on breakout day
- **Lower velocity than EP but higher reliability**

---

## Setup Type C: EMA Reclaim / Wedge Pop (Kell + Velez)

> Why it works: Institutions add to winning positions on pullbacks to moving averages. The reclaim candle signals their bid has re-appeared - you're entering where Big Money is buying.

- Stock in uptrend pulls back to 10d or 20d EMA
- EMA crossback: price tags EMA then snaps back above
- Elephant Bar off EMA = institutional buying signal
- Entry: ORB above 15/30 min candle on reclaim day
- **Lowest risk entry - tightest stop possible**

---

## ORB Entry Rules: Morning Session

You trade max 30-45 min of the first hour and last hour. The ORB is your only entry mechanism.

| Time | Event |
|---|---|
| 9:30 | Open - ORB starts |
| 9:45 | 15min candle - Type A/EP entry window |
| 10:00 | 30min candle - Preferred ORB level |
| 10:15 | Trading OFF - No new entries |

- Wait for the 15min OR 30min candle to fully form - **NEVER enter on a developing candle**
- Entry trigger: buy stop limit just above the high of the 15/30min candle
- 15min candle -> Type A EP setups; 30min candle -> Type B VCP / Type C EMA reclaim
- Volume on the breakout candle must be above the developing average
- If SPY gaps down hard at open - skip morning entries, reassess at last hour

---

## ORB Entry Rules: Last Hour Session

| Time | Event |
|---|---|
| 14:30 | Pre-close scan - Review watchlist |
| 15:15 | Last hour opens - Late-day entries |
| 15:45 | Entry deadline - Last valid ORB |
| 16:00 | Close - Log & review |

- Only enter if stock is in top 25% of day's range heading into last hour (Velez / Kullamägi)
- Late-day strength = institutional buying into close, not retail momentum
- Entry: above high of first 15min candle of the last hour
- Valid only if stock held near highs through mid-day - no recovery plays
- Stop: below the last-hour opening candle low
- Strong preference: stock showing RS vs SPY in the final 2 hours

---

## Stop Loss Placement & Position Sizing

| Rule | Detail |
|---|---|
| Primary SL | Below the ORB low (low of the 15/30min entry candle) |
| Maximum SL | 7% below entry - Minervini hard limit, non-negotiable |
| Base SL | Below the base/consolidation low (for VCP setups) |
| EMA SL | Below the 10d EMA (for EMA reclaim entries) |
| Intraday rule | Never move SL further away. Move only to BE+ |
| Break-even rule | Move to BE after first partial profit-take (1.5-2R) |
| Hard rule | NEVER add to a losing position. One SL hit = close. |

**Sizing:** Max $ risk = 0.5% x Portfolio. Shares = Max $ risk / (Entry - Stop)
**Example:** $100K portfolio, $500 max risk, entry $45, stop $42.50 -> 200 shares

---

## Profit-Taking Staircase

| Stage | Action | Detail |
|---|---|---|
| ENTRY | Full size | Buy stop above ORB high. SL placed immediately. |
| 1.5 - 2R | -30-50% | Take 30-50% off. Move stop to break-even. Lock in income (Kullamägi 3-5 day rule). |
| > 4R | -30% more | Take another 30% off. Trail remaining stop using 5 EMA. |
| RUNNER | Last portion | Hold until an exit signal fires. Never let a 2R winner go to a loss. |

---

## Exit Signals - Any One Fires = Close Remainder

| Signal | Condition | Rationale |
|---|---|---|
| 5 MA Signal | 2nd consecutive close below 5 EMA | Short-term momentum broken |
| ATR Candle | Single candle > 1.5x ATR(14) against you | Abnormal selling pressure |
| 10 MA Signal | Daily close below the 10 EMA | Trend cycle weakening |
| 20 MA Signal | Close below 20 EMA on above-avg volume | Swing cycle over |
| RS Breakdown | Stock underperforms SPY on down days | Institutional support withdrawn |
| Narrative Break | Catalyst negated - bad news, guidance cut | Story breaks = trade is over |
| Extension Rule | Stock > 20% above 10d EMA | Begin scaling out |
| Time Stop | 50 days elapsed + thesis not playing out | Release and redeploy capital |

---

## Golden Rules - Non-Negotiable

1. **MANAGE BY RULES, NOT FEELINGS** - Every decision must be pre-defined. Not in the rulebook = don't do it.
2. **NEVER TRADE A DEVELOPING CANDLE** - The close counts. Wait for the full 15/30min candle.
3. **NEVER ADD TO A LOSING TRADE** - One stop-out closes the position entirely.
4. **EXIT WHEN PREMISE IS BROKEN** - If your reason for being in is gone - get out. No exceptions.
5. **RESPECT YOUR TIMEFRAME** - You are a 5-50 day swing trader. Set stops. Walk away.
6. **NEVER FIGHT THE TREND** - Never short a strong trend, never long a broken chart.
7. **THE MARKET IS RIGHT, YOU ARE NOT** - When 3+ setups fail - pause. Regime has shifted.
8. **PATIENCE IS THE EDGE** - High-probability trades require patience. Forcing trades destroys edge.

---

## Trader DNA - Who Contributed What

**Mark Minervini** - SEPA / VCP
- Trend Template (Price > 50d > 200d, 200d rising) | VCP entry | Hard stop <= 7-8% | Pivot breakout on 40-50%+ volume

**Kristjan Kullamägi** - Breakout / EP
- Episodic Pivot (10%+ gap on 5-10x volume) | ORB entry | 3-5 day rule: sell 1/3-1/2 early | 10/20 EMA trailing stop

**Oliver Kell** - Price Cycle
- EMA stack: 10 > 20 > 50 | Wedge Pop entry | Exhaustion at >20% above 10d EMA | Exit below 20d EMA on volume

**Oliver Velez** - Institutional Footprint
- Elephant Bar = ignition signal | Tail Bar recovery | 40/40/20 rule | Late-day validation: top 25% of range

**Vincent Bruzzese** - Market-Relative
- RS/RW vs SPY | Sector confirmation | Market regime filter: SPY above 50d & 200d | Higher lows while SPY lower lows = buy
