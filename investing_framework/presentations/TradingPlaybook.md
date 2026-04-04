---
marp: true
theme: default
paginate: true
header: 'Trading Playbook'
footer: 'Swing Trading System'
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

# Swing Trading Playbook

**Relative strength momentum trading -- 5-50 days -- stocks and options.**

| SPEC | |
|------|---|
| Timeframe | 5-50 days |
| Risk per trade | 0.5% of portfolio |
| Direction | Long AND short -- RS/RW vs SPY in both directions |
| Execution windows | First hour (9:30-10:15) and last hour (15:15-15:45) only |
| Charting | 5, 10, 20, 50 SMA - Bollinger Bands - Put/Call volume ratio |
| Instruments | Stocks and options on liquid names (Vol > 1M, Price > $3) |

---

## Core Thesis

> Stocks with relative strength against SPY are being accumulated by institutions. Stocks with relative weakness are being distributed. These divergences persist for weeks -- not days -- because institutional capital moves slowly. Enter on the breakout, ride the follow-through, exit when relative strength breaks.

This thesis synthesises four edges:

- **Minervini:** Supply exhaustion (VCP) creates low-risk breakout entries
- **Bruzzese:** RS/RW vs SPY identifies institutional accumulation before the move
- **Kullamagi:** Episodic pivots and ORB entries capture the ignition moment
- **Kell:** SMA stack and price cycles define trend health and exhaustion

Adapted to your style: SMA-only charting, Bollinger Bands for volatility context, first/last hour execution only, options for leveraged expression with hard stops.

---

## Profit Mechanisms -- Overview

Each trade must exploit an identified profit mechanism -- a specific, repeatable market effect with documented evidence.

**Active Mechanisms (Grade A):**

| # | Mechanism | Role |
|---|-----------|------|
| PM-01 | Breakout Momentum | Primary -- drives Type A, B, C setups |
| PM-02 | Post-Earnings Announcement Drift | Amplifier inside EP entries |
| PM-03 | Pre-Earnings Anticipation | Standalone -- T-14 to T-1 trade |
| PM-04 | OTM Informed Flow | Confirmation signal, not standalone |
| PM-05 | RS/RW Divergence | Core selection filter for every trade |

---

## PM-01 -- Breakout Momentum *(Grade A - ACTIVE)*

> Momentum persists because institutions can't deploy capital instantly and information diffuses slowly. The ignition event forces large funds to start building. They buy for weeks -- not days.

| Attribute | Detail |
|-----------|--------|
| **Market effect** | Price drift following supply exhaustion (VCP) or information shock (EP) |
| **Why it works** | Institutional liquidity constraints + behavioural under-reaction to new information |
| **Academic basis** | Century momentum (Geczy & Samonov 2017); VCP confirmed in Minervini's empirical track record |
| **Timeframe** | 5-50 days (primary leg of the move) |
| **Signals** | VCP tightening + VDU, SMA stack intact, RS leading, volume-confirmed breakout |
| **Setups** | Type A (EP), Type B (VCP), Type C (SMA Reclaim) |
| **Structure** | Long call 45-60 DTE (IVR < 40%) - debit spread (IVR 40-70%) - stock + stop |
| **Invalidation** | Double close back inside the breakout range. RS breakdown vs SPY. |

---

## PM-02 -- Post-Earnings Announcement Drift *(Grade A - ACTIVE)*

> A strong earnings surprise creates 40-60 days of forced institutional accumulation. The gap is day 1, not the full trade.

| Attribute | Detail |
|-----------|--------|
| **Market effect** | Price drift for 40-60 days after an earnings surprise |
| **Why it works** | Institutional under-reaction + forced accumulation by funds tracking quality/momentum screens |
| **Academic basis** | Ball & Brown 1968; Bernard & Thomas 1989, 1993; confirmed across 100+ years and 40+ countries |
| **Timeframe** | 40-60 days from announcement |
| **Signals** | Gap >= 10% on 5-10x volume - closes top 25% of range - EPS beat top decile |
| **Amplifiers** | Larger EPS surprise = longer drift - Small/mid cap > large cap - Strong sector confirms |
| **Setups** | Type A (EP) -- PEAD is not a separate setup; it is a momentum amplifier inside EP entries |
| **Structure** | Long ATM call 45-60 DTE - bull call vertical if IV spiked on gap |
| **Invalidation** | Gap fills within 3 days. RS flips negative. |

---

## PM-03 -- Pre-Earnings Anticipation *(Grade A - ACTIVE)*

> Markets begin incorporating earnings expectations 10-20 days before the report. IV expansion adds a vega tailwind to directional bets.

| Attribute | Detail |
|-----------|--------|
| **Market effect** | Price drift + IV expansion 10-20 days before earnings announcement |
| **Why it works** | Institutional positioning ahead of expected beat (Richardson & Veenstra 2006) |
| **Timeframe** | T-14 to T-1 (exit day before earnings -- never hold through binary event) |
| **Signals** | RS stock in Stage 2 - beat >= 3 of last 4 quarters - holding 20 SMA at T-14 - IVR < 30% |
| **Structure** | ATM call or call diagonal. Buy at T-14, exit at T-1 regardless of P&L. |
| **Stop** | Close below 20 SMA or 50% premium loss -- whichever first |
| **Invalidation** | Stock gapped unpredictably on any of last 4 reports. IVR > 40% at entry (premium too rich). |

---

## PM-04 -- OTM Informed Flow *(Grade A - ACTIVE)*

> Informed investors prefer OTM options for leverage. Their activity in OTM puts vs calls predicts future stock returns.

| Attribute | Detail |
|-----------|--------|
| **Market effect** | Directional signal from unusual OTM option volume |
| **Why it works** | Informed traders use OTM options to maximize leverage on private information (Pan & Poteshman 2006) |
| **Timeframe** | 5-20 days (signal precedes the move by days to weeks) |
| **Signals** | OTM call vol > 3x OI in next 4 weeks - block prints in multi-leg call spreads in final 2h - put/call ratio on RS stock > 0.7 |
| **Role** | **Confirmation signal, not standalone entry.** Adds to watchlist priority when combined with a Type A/B/C setup. |

---

## PM-05 -- RS/RW Divergence *(Grade A - ACTIVE)*

> Stocks that hold up when SPY drops are being accumulated by institutions. Stocks that drop more are being distributed. The divergence persists for weeks.

| Attribute | Detail |
|-----------|--------|
| **Market effect** | Price divergence from broad market signals institutional accumulation or distribution |
| **Why it works** | Institutions build/exit positions over weeks. RS/RW is their footprint. (Bruzzese) |
| **Timeframe** | 5-50 days (the divergence is the holding period) |
| **Signals** | RS line at new highs vs SPY (longs) - new lows (shorts) - higher lows while SPY makes lower lows |
| **Role** | **Core selection filter.** Every trade in this playbook requires RS/RW confirmation. |
| **Setups** | Type A/B/C (longs via RS), Type D (shorts via RW) |
| **Invalidation** | RS flips -- stock starts underperforming SPY on down days (longs) or outperforming on up days (shorts) |

---

## Embedded Mechanisms (Context Filters)

These are not standalone trades but are embedded in scanning, selection, or regime decisions.

| # | Mechanism | Role | Status |
|---|-----------|------|--------|
| 04 | Century Momentum | 12-month return filter in trend template (box 01) | Embedded in scan |
| 06 | Positive Drift on Indices | Cross-over to DRIFT portfolio when indices are in Stage 2 | Embedded in regime |
| 10 | VIX Mean Reversion | After VIX spike >25 + "lower high" -> SPY/QQQ ORB entry (index trade only) | Active -- index ETFs |
| 15 | Rebalancing Tailwind | Avoid new longs in final 3 days of quarter. Enter first week of new quarter. | Embedded in timing |
| 21 | S/R Reversal | Bollinger Band lower touch in uptrend as pullback entry signal (Type C) | Embedded in setup |
| 30 | Options Complexity Alpha | Use multi-leg structures over simple calls/puts for better risk-adjusted returns | Embedded in structure |
| -- | Sentiment-Conditional TA | Technical signals work better in high-sentiment regimes. Use as GO/NO-GO overlay. | Embedded in regime |
| -- | Option Momentum | Options with high recent returns continue outperforming. Favour positive recent option returns. | Embedded in structure |

---

## PM-06 -- News-Driven Drift *(Grade B - RESEARCH)*

> News-driven price moves show strong continuation. No-news extreme moves tend to reverse.

| Attribute | Detail |
|-----------|--------|
| Market effect | Price continuation for days-weeks after identified news events; reversal on no-news moves |
| Why it works | Information diffuses slowly; complex or nuanced news takes longer to price (Boudoukh et al., 2013) |
| Academic basis | Boudoukh, Feldman, Kogan, Richardson 2013: variance ratios 120% higher on news days |
| Timeframe | 5-20 days post-news |
| Signals | Large move (>3%) WITH identified catalyst = continuation. WITHOUT catalyst = fade candidate. |
| Structure | Long call / stock on news-driven RS breakouts. Fade on no-news spikes in RW stocks. |
| Invalidation | Move reverses within 2 days despite clear catalyst. |
| Research gap | Build a systematic news-vs-no-news classifier. Backtest on watchlist. |

---

## PM-07 -- Retail Attention Contrarian *(Grade B - RESEARCH)*

> Stocks with extreme retail buying frenzies are systematically overpriced. Fade the herding.

| Attribute | Detail |
|-----------|--------|
| Market effect | Intense retail buying (attention-driven) predicts -4.7% abnormal returns over 20 days |
| Why it works | Retail buys on attention/salience, not fundamentals. Price overshoots, then mean-reverts. (Barber et al. 2021) |
| Academic basis | Robinhood herding study; Boehmer et al. 2021 retail flow tracking; Kaniel et al. 2008 |
| Timeframe | 5-20 days after the attention spike |
| Signals | Extreme RVOL + social media spike + retail herding WITHOUT fundamental catalyst |
| Structure | Short stock or bear put spread 30-45 DTE after initial frenzy fades (day 2-3) |
| Invalidation | Stock holds gains for 5+ days with institutional follow-through |
| Research gap | Define "extreme retail attention" quantitatively. Backtest 20-day returns. |

---

## PM-08 -- Overnight Reversal *(Grade B - RESEARCH)*

> Sell-offs create robust positive overnight returns. Enter at the close after a sell-off, exit at the open.

| Attribute | Detail |
|-----------|--------|
| Market effect | Largest positive equity returns accrue between 2-3 AM ET (European open). Post-sell-off overnight returns average 3.6% annualized. |
| Why it works | End-of-day order imbalances from forced selling resolve overnight. European institutional buying absorbs the discount. (Boyarchenko et al. 2023) |
| Timeframe | Overnight (close-to-open). Can be used as entry timing for swing trades. |
| Signals | Stock down >2% intraday on above-average volume. RS intact (not a fundamental breakdown). |
| Structure | Buy stock at MOC. Evaluate at next morning open -- hold as swing if setup confirms. |
| Invalidation | Overnight gap-down despite prior sell-off = fundamental shift. Exit immediately. |
| Research gap | Backtest overnight returns on RS stocks after >2% intraday drops. |

---

## PM-09 -- Mean Reversion to Trend *(Grade B - RESEARCH)*

> Stocks oscillate around their trend in a quasi-periodic pattern. Oversold pullbacks within an uptrend revert to the moving average.

| Attribute | Detail |
|-----------|--------|
| Market effect | De-trended price residuals behave as mean-reverting Ornstein-Uhlenbeck process |
| Why it works | Institutional rebalancing creates oscillations. Overreaction to short-term news + institutional buying on dips. (Nassar & Ephrem 2020) |
| Timeframe | 5-15 days (half the oscillation cycle) |
| Signals | Price touches lower BB while 20 SMA is rising. Volume declining on pullback. RS intact vs SPY. |
| Structure | This IS Type C (SMA Reclaim) -- with a quantitative mean-reversion framework underneath. |
| Invalidation | Close below 50 SMA. RS flips negative. Volume expands on pullback (distribution). |
| Research gap | Quantify oscillation periodicity for top 20 watchlist names. |

---

## PM-10 -- Insider & Corporate Action Drift *(Grade B - RESEARCH)*

> Insider buying and share buybacks predict positive returns. These are the strongest fundamental confirmation signals.

| Attribute | Detail |
|-----------|--------|
| Market effect | Insider purchases predict positive returns over 3-12 months. Buybacks show 3-5% drift over 1-2 months. |
| Why it works | Insiders have information advantage. Buybacks signal undervaluation. Both reduce float. (Ikenberry et al. 1995) |
| Timeframe | 20-50 days post-announcement or post-filing |
| Signals | SEC Form 4 insider purchase > $100k by C-suite. Buyback > 5% of float. Stock in Stage 2 / RS positive. |
| Structure | Stock + hard stop or long call 45-60 DTE. Debit spreads work well for steady drift. |
| Invalidation | Insider buy is routine auto-purchase. Buyback not followed by actual repurchase activity. |
| Research gap | Build SEC Form 4 screen. Backtest insider buy + VCP combo. |

---

## PM-11 -- Short Squeeze Setup *(Grade B - RESEARCH)*

> Extreme short interest + catalyst + dealer gamma exposure = forced covering cascade.

| Attribute | Detail |
|-----------|--------|
| Market effect | Short covering amplified by dealer gamma hedging creates violent multi-day rallies |
| Why it works | Shorts forced to buy to cover. Dealers hedging short gamma must also buy. Both amplify the move. |
| Timeframe | 3-10 days (squeeze is fast and violent) |
| Signals | SI > 20% of float - days-to-cover > 5 - RS turning positive - catalyst - dealer GEX negative |
| Structure | Long stock or long call 30 DTE. Take profits aggressively -- squeezes reverse fast. |
| Invalidation | SI declining before entry (shorts already covering). No catalyst -- squeeze without trigger is hope. |
| Research gap | Quantify SI% + DTC combination for reliable squeeze outcomes. |

---

## Research Pipeline

All mechanisms below are being evaluated. Do not trade until all 4 steps of the Outlier framework are complete.

| # | Mechanism | Cluster | Grade | Priority |
|---|-----------|---------|-------|----------|
| 06 | News-Driven Drift | Drift | B | High -- directly actionable |
| 07 | Retail Attention Contrarian | Institution | B | High -- contrarian edge |
| 08 | Earnings Volatility Crush | Fear | A | High -- already graded A |
| 09 | Overnight Reversal | Drift | B | Medium -- entry timing |
| 10 | Mean Reversion to Trend | Drift | B | Medium -- extends Type C |
| 11 | Insider & Corporate Action | Drift | B | Medium -- confirmation |
| 12 | Short Squeeze | Institution | B | Medium -- episodic |
| 14 | Gamma-Induced Pinning | Institution | B | Low |
| 20 | End of Month / Turn of Month | Regime | B | Low |
| 29 | VTS Slope Alpha | Fear | B | Low |

---

## How to Add a New Profit Mechanism

> Use the 4-step Outlier framework. Do not trade a mechanism until all four steps are validated.

**Step 1 -- Define the Profit Mechanism**
Identify the SPECIFIC market effect you're trying to capture.
- Price movement: drift, momentum, breakout/down, mean reversion
- Volatility: expansion, contraction, crush, mean reversion
- Structural: hedging pressure, rebalancing flows, informed flow

**Step 2 -- PM Behavior & Signals**
Build tools to identify, measure, and time the mechanism.
- Document behavior and conditions
- Build a signal list -- test each signal against the PM
- Match the best signals; trim to the most predictive set

---

## How to Add a New Profit Mechanism (cont.)

**Step 3 -- Outline Fitting Structures**
How to monetise it via the right trade structure.
- Create candidate structures (stock, options, spread, combination)
- Overlay portfolio risk rules and BP constraints
- Identify the best-fit structure for this mechanism

**Step 4 -- Build the Strategy**
Turn the best structure into a fully defined, testable strategy card.
- Isolate key inputs: instrument, structure, greeks, sizing, entry/exit rules
- Backtest -> paper trade -> live test (minimum 30 trades)
- Add to mechanism deck only after live validation

**Feedback loop:** After 30+ live trades, review. Update the mechanism card or archive it.

---

## Mechanism Card Template

```
PM-XX -- [Name] *(Grade [A/B/C] - [ACTIVE/RESEARCH/EMBEDDED])*

> [One-sentence thesis]

| Attribute | Detail |
|-----------|--------|
| Market effect | [what you're capturing] |
| Why it works | [structural reason + academic basis] |
| Timeframe | [how long the effect lasts] |
| Signals | [what triggers entry] |
| Structure | [how to express it] |
| Invalidation | [what kills the thesis] |
| Status | [research stage or active] |
```

---

## 01 -- Market Regime

> 75% of stocks follow the general market. Context always comes first. *(Bruzzese)*

### GO -- All Must Be Present

- SPY above 20 SMA and 50 SMA -- both sloping upward
- VIX below 20 or falling after a spike
- Advancing > Declining stocks (breadth expanding)
- At least one sector showing clear RS vs SPY

### NO-GO -- Any One Triggers Pause

- SPY below 50 SMA
- VIX > 30 or spiking sharply
- Declining > Advancing 2:1+
- 3+ consecutive stopped-out trades -- regime has shifted
- Major macro event within 48h (FOMC / CPI / NFP)

---

## 01 -- Direction Bias

| SPY Regime | Long Setups | Short Setups |
|-----------|------------|--------------|
| Above 20 + 50 SMA, both rising | Full deployment | Avoid -- drift works against you |
| Above 50 SMA, 20 SMA flattening | Normal size | Selective -- weak stocks in weak sectors only |
| Below 50 SMA | Avoid -- bear market | Full deployment on RW stocks |

---

## 02 -- Scanning & Stock Selection

### Weekly Scan (Weekend)

Run every weekend. Sets the watchlist for the week.

1. **Sector rotation check** -- Which sectors gained RS vs SPY this week? Which are building bases?
2. **RS leaders** -- Stocks holding near highs while SPY dipped. These are being accumulated.
3. **RW laggards** -- Stocks making new lows while SPY holds. These are being distributed. *(Short candidates)*
4. **Put/call volume scan** -- Unusual OTM call volume > 3x OI = informed upside anticipation. Unusual put volume on RS stock = squeeze fuel.
5. **Update watchlist** -- Cap at 20 names. Rank by setup quality. Top 5 for live trading.

---

## 02 -- Daily Scan (Pre-Market)

| Check | What to Look For |
|-------|-----------------|
| Gap-ups/downs | > 5% move with volume > 2x average -- look for catalyst |
| RS divergence | Stock green while SPY red (or vice versa) -- accumulation/distribution |
| SMA reclaim | Stock reclaiming 20 SMA after a pullback -- potential Type C entry |
| Volume spike | RVOL > 1.5x before 10am -- institutional activity starting |
| Bollinger squeeze | BB width contracting to multi-week low -- breakout imminent |
| Earnings proximity | Remove any stocks reporting within 5 days -- no binary risk |

---

## 02 -- Stock Selection: 5-Box Checklist

All 5 must pass. One fail = skip.

**01 -- Trend Template** *(Minervini, adapted to SMAs)*
- Price > 20 SMA > 50 SMA (longs) - Price < 20 SMA < 50 SMA (shorts)
- 50 SMA sloping in trade direction
- Price within 25% of 52-week high (longs) or low (shorts)
- Positive 12-month return for longs - negative for shorts *(century momentum filter)*

**02 -- Relative Strength / Weakness** *(Bruzzese)*
- RS line at/near new highs vs SPY (longs) - new lows (shorts)
- Stock holds up when SPY dips (longs) - drops more when SPY dips (shorts)
- Outperforming/underperforming sector peers over 1M
- Higher lows while SPY makes lower lows = institutional support (longs)

---

## 02 -- 5-Box Checklist (cont.)

**03 -- Base Quality** *(Minervini + Kell, adapted)*
- Consolidation of at least 1-3 weeks (tight range)
- Volume contracting during the base (VDU -- dry-up)
- ATR 0-6x from base (not extended / parabolic). ADR > 3%.
- SMA stack: 5 > 10 > 20 > 50, all sloping in trade direction
- **Bollinger Band squeeze:** BB width < 20-day average BB width = volatility contraction

**04 -- Catalyst** *(Camillo + PEAD)*
- Clear reason for the move: earnings beat, news, sector theme, social narrative
- PEAD (if earnings): gap >= 10% AND closed top 25% of day's range
- Put/call ratio: unusual call volume on RS stock = informed bid
- No binary event within 5 days

**05 -- Risk Parameters**
- Stop defined before entry (base low / ORB low / 20 SMA)
- Stop <= 7% from entry (Minervini hard limit)
- Size = (0.5% x portfolio) / (entry - stop). R:R >= 2:1 minimum

---

## 03 -- Setup Type A: Episodic Pivot *(Kullamagi + PEAD)*

> The gap-up is day 1 of 40-60 days of institutional accumulation.

- Gap >= 10% (prefer 15%+) on 5-10x avg volume
- Fundamental catalyst: earnings beat, major news
- Closes top 25% of day's range -- institutions buying into the close
- **Entry:** ORB above 15min candle high (morning window)
- Highest velocity -- biggest winners. Also highest risk.

---

## 03 -- Setup Type B: VCP Breakout *(Minervini)*

> Supply exhaustion -- each pullback shakes out weak holders. When the last seller is gone, any buying drives price explosively.

- 2-6 weeks of tightening range, 3+ contraction points
- Volume dries up in final days (VDU)
- **Bollinger squeeze visible:** BB width at multi-week low inside the base
- Breakout above pivot on 40-50%+ above avg volume
- **Entry:** ORB above 30min candle on breakout day

---

## 03 -- Setup Type C: SMA Reclaim *(adapted from Kell + Velez)*

> Institutions add to winning positions on pullbacks to moving averages.

- SMA stack intact: 5 > 10 > 20 > 50, all sloping up (longs)
- Pullback to 10 or 20 SMA on contracting volume
- **Elephant Bar:** Candle >= 2x average bar size, opens near/below SMA, closes above it near the high. Institutional bid confirmed.
- **Tail Bar:** Long lower wick stings the SMA and recovers -- institutions stepped in.
- **Entry:** ORB above 30min candle on reclaim day. Last-hour entry valid if stock in top 25% of range at 15:15.

---

## 03 -- Setup Type D: Breakdown / Short *(Bruzzese RW, adapted)*

> Stocks with relative weakness against SPY in a weak regime are being distributed by institutions. Mirror the long playbook.

- Price < 20 SMA < 50 SMA. 50 SMA sloping down.
- RW line at new lows vs SPY. Drops more on SPY down days.
- Base forming below declining SMAs -- supply being created
- **Entry:** ORB below 30min candle low on breakdown day
- **Stop:** Above the breakdown candle high or 20 SMA

---

## 03 -- Setup Priority

| Regime | Priority |
|--------|----------|
| Strong bull (SPY > 20 + 50, breadth expanding) | A > B > C. No shorts. |
| Moderate bull (SPY > 50, 20 flattening) | B > C > A. Selective shorts (Type D) on extreme RW. |
| Correction / bear (SPY < 50) | D only. No longs. Spreads only for options. |

---

## 04 -- Entry Execution

> You trade the first hour and the last hour. Nothing in between. Set orders, walk away.

### Trading Windows

| Window | Time | Purpose |
|--------|------|---------|
| Pre-market | 9:00-9:30 | Review regime, scan, set ORB alerts |
| **Morning ORB** | **9:30-10:15** | Primary entry window. 15min (Type A) or 30min (Type B/C/D) candle. |
| Dead zone | 10:15-15:15 | No entries. Manage stops only. |
| **Last hour ORB** | **15:15-15:45** | Secondary entry. Top/bottom 25% of range required. |
| Post-close | 16:00-16:15 | Log trades, review exits, update watchlist. |

---

## 04 -- Entry Rules

- **Wait for the candle to fully close.** Never enter on a developing candle. The close counts.
- **Pre-set buy/sell stop limit orders** above (longs) or below (shorts) the ORB candle high/low. Not a manual click.
- 15min candle for Type A (EP) -- moves fast. 30min candle for Type B/C/D.
- RVOL > 1.5x by 9:45 = strong signal. < 1.5x = reduce size or skip.
- If SPY gaps hard against your direction at open -- skip morning entries, reassess at last hour.

### Last-Hour Entry (Velez validation)

- **Longs:** Stock in top 25% of day's range at 15:15. Held highs through mid-day.
- **Shorts:** Stock in bottom 25% of day's range at 15:15. Held lows through mid-day.
- Enter above/below high/low of first 15min candle of the last hour.
- RS/RW confirmation: stock diverging from SPY in the final 2 hours.

---

## 05 -- Stop Loss & Sizing

### Hard Stops -- Non-Negotiable

> Due to limited intraday availability, all stops must be hard (bracket orders). No mental stops.

| Entry Type | Stop Placement |
|-----------|---------------|
| ORB entry | Below/above the entry candle low/high |
| VCP breakout | Below the base / consolidation low |
| SMA reclaim | Below the 20 SMA |
| EP (Type A) | Below the gap-open level |
| Short (Type D) | Above the breakdown candle high or 20 SMA |
| **Hard maximum** | **7% from entry -- never exceeded** |

### Position Sizing

```
Max $ risk = 0.5% x Portfolio
Shares    = Max $ risk / (Entry - Stop)
```

---

## 05 -- Options Sizing: The Hard Stop Problem

> Options don't work with tight hard stops. The bid-ask spread and intraday vol can trigger a stop on the option even when the underlying hasn't breached the level.

**Solution: Define risk on the underlying, express via options.**

| Method | How It Works | When to Use |
|--------|-------------|-------------|
| **Stock stop, option position** | Hard stop on the underlying price level. When triggered, close option at market. | Liquid names with tight spreads |
| **Debit spread (defined risk)** | Max loss = debit paid. No stop needed. | IVR 40-70%. Can't monitor intraday. |
| **Long call/put with premium stop** | Close if option loses 50% of premium paid. | IVR < 40%. High conviction. |
| **OCO bracket on underlying** | OCO bracket on small stock position. When stop triggers, close option. | Brokers with conditional orders |

**Rule:** If you cannot define a hard exit mechanism, trade the stock instead.

---

## 06 -- Options Structure Selection

### By Setup Type and IVR

| Setup | IVR < 40% | IVR 40-70% | IVR > 70% |
|-------|-----------|-----------|-----------|
| **Type A (EP)** | Long ATM call 45-60 DTE | Bull call spread | Bull call spread only |
| **Type B (VCP)** | Long ATM call 45-60 DTE | Bull call spread | Bull call spread |
| **Type C (SMA Reclaim)** | Long ATM call 30-45 DTE | Bull call spread 30-45 DTE | Spread only |
| **Type D (Short)** | Long ATM put 45-60 DTE | Bear put spread | Bear put spread only |
| **Extended trend** | PMCC: deep ITM call 90+ DTE, sell OTM call 20 DTE | PMCC | Not recommended |

---

## 06 -- Put/Call Volume as Confirmation

Before entering an options trade, check the option chain:

- **OTM call vol > 3x OI** in next 4 weeks = informed institutional bid -> confirms long setup
- **OTM put vol > 3x OI** = informed downside anticipation -> confirms short setup or warns against long
- **IV term structure steep** (6M IV >> 1M IV) = market overstating long-term risk -> long calls are cheap
- **IV term structure inverted** (near-term IV > long-term) = event risk priced in -> use spreads, not naked longs

---

## 07 -- Trade Management & Exits

### Profit-Taking Staircase

| Stage | Trigger | Action |
|-------|---------|--------|
| Entry | ORB fires | Full size. Hard stop placed immediately. |
| 1st take | 1.5-2R | Close 30-50%. Move stop to break-even. |
| 2nd take | > 4R | Close 30% more. Trail stop on 5 SMA. |
| Runner | -- | Hold until exit signal fires. |

**3-5 day rule** *(Kullamagi):* After 3-5 strong consecutive days, take the first partial regardless of R level. Momentum exhaustion is coming.

**Rule: Never let a 2R winner turn into a loss.**

---

## 07 -- Exit Signals

Any one fires = close remainder.

| Signal | Condition | Source |
|--------|----------|--------|
| **5 SMA** | 2nd consecutive daily close below 5 SMA (longs) / above (shorts) | Kullamagi |
| **ATR candle** | Single candle > 1.5x ATR(14) against position | Institutional exit |
| **20 SMA** | Daily close below 20 SMA on above-avg volume | Kell / Bruzzese |
| **Bollinger break** | Close outside upper BB after extended run = exhaustion | Bollinger + Kell |
| **RS/RW breakdown** | Stock starts underperforming SPY on down days (longs) | Bruzzese |
| **Narrative break** | Catalyst negated -- bad news, guidance cut, thesis invalidated | Camillo |
| **Extension** | Stock > 20% above 10 SMA -- scale out, don't add | Kell |
| **Time stop** | 50 days elapsed, thesis not playing out | Capital redeployment |

---

## 07 -- Bollinger Band Exit Rules

- **Upper BB touch after 3+ weeks of trend:** Begin scaling. Stock is 2 SD extended.
- **Lower BB touch on pullback in uptrend:** Hold if 20 SMA is rising. Only exit on close below 20 SMA.
- **BB squeeze re-forms after breakout:** New consolidation forming -- tighten stop to base low and watch for continuation or reversal.

---

## 08 -- Daily Execution Checklist

### Pre-Market (9:00)

- [ ] SPY regime check: above 20 + 50 SMA? Breadth?
- [ ] VIX level and direction
- [ ] Review watchlist: any gap-ups with catalyst? Any RW breakdowns?
- [ ] Remove stocks reporting earnings today/tomorrow
- [ ] Set ORB alerts on top 5 watchlist names
- [ ] Pre-set buy/sell stop limit orders for pre-identified setups

### First Hour (9:30-10:15)

- [ ] Wait for 15/30min candle to fully form
- [ ] Confirm volume: RVOL building? Breakout candle above avg?
- [ ] Check SPY direction: supporting the move?
- [ ] Orders trigger automatically via pre-set brackets
- [ ] At 10:15 -- stop. No new entries until last hour.

---

## 08 -- Daily Execution Checklist (cont.)

### Dead Zone (10:15-15:15)

- [ ] No entries. Monitor hard stops only.
- [ ] Note stocks holding well / showing RS for last-hour review
- [ ] Check: any positions hit profit targets? -> set partial exit orders

### Last Hour (15:15-15:45)

- [ ] Scan watchlist: which stocks in top/bottom 25% of range?
- [ ] RS/RW confirmation: stock diverging from SPY?
- [ ] Enter via ORB of first 15min candle of last hour
- [ ] Execute any pending partial exits from mid-day targets
- [ ] Hard deadline: no entries after 15:45

### Post-Close (16:00)

- [ ] Log all trades: entry, stop, target, result, notes
- [ ] Review: were rules followed? Any emotion-based decisions?
- [ ] Check daily SMAs on open positions: exit signals forming?
- [ ] Update watchlist for tomorrow

---

## 09 -- Non-Negotiable Rules

1. **Rules, not feelings.** Every decision is pre-defined. If it's not in this playbook, don't do it.
2. **The close counts.** Never enter on a developing candle. Wait for the full 15/30min bar.
3. **Hard stops only.** You're not watching the screen mid-day. Bracket orders are mandatory.
4. **Never add to a loser.** One stop = close the full position. No averaging down.
5. **Exit when premise breaks.** Don't adapt a failing trade. If RS flips, the trade is over.
6. **Respect the timeframe.** 5-50 days. No intraday management between the windows.
7. **Never fight the trend.** Long in bull. Short in bear. Not both at once.
8. **3 failures = pause.** Three consecutive stops means the regime shifted. Reassess.
9. **Options must have defined exits.** If you can't set a hard stop mechanism, trade the stock.
10. **Patience is the edge.** Wait for tight, high-probability setups. Forcing trades destroys edge.

---

## 10 -- Charting Setup: Indicators

| Indicator | Setting | Purpose |
|-----------|---------|---------|
| **SMA 5** | Close | Short-term momentum / exit trail |
| **SMA 10** | Close | Trend reference / secondary exit |
| **SMA 20** | Close | Primary trend filter and stop reference |
| **SMA 50** | Close | Regime filter / long-term trend |
| **Bollinger Bands** | 20 period, 2 SD | Volatility squeeze detection + exhaustion |
| **Volume** | With 50d average overlay | Confirm breakouts, detect VDU |
| **RS line vs SPY** | Custom or broker-provided | Core selection tool -- RS/RW divergence |
| **Put/Call ratio** | Options chain | Informed flow confirmation |

---

## 10 -- Why SMAs (Not EMAs)

- SMAs are smoother and lag more than EMAs -- this is a **feature, not a bug**. You're a swing trader with limited screen time. SMAs filter noise better for end-of-day decisions.
- The 20 SMA is the institutional standard. Velez's "center of gravity," Minervini's trend template, Bruzzese's regime filter -- all reference SMAs.
- Consistency: one indicator type across all timeframes. No mixing EMA and SMA signals.

## Why Bollinger Bands

- **Squeeze detection:** When BB width contracts to a multi-week low, a breakout is imminent. This is the visual equivalent of Minervini's VCP.
- **Exhaustion detection:** Stock touching or exceeding the upper BB after a multi-week trend = 2 SD extended. Begin scaling, not chasing.
- **Pullback context:** Stock touching lower BB during an uptrend = oversold within the trend. If 20 SMA is still rising, this is a Type C entry opportunity, not an exit.

---

## Appendix -- Trader Attribution

| Trader | Taken | Adapted |
|--------|-------|---------|
| **Minervini** | Trend Template, VCP, 7% hard stop, volume dry-up, pivot breakout | Replaced 150/200 SMA with 20/50 SMA. Dropped fundamental acceleration screen. |
| **Bruzzese** | RS/RW vs SPY as core edge, sector confirmation, regime filter | Extended to short side (RW for short setups). Core thesis of the playbook. |
| **Kullamagi** | Episodic Pivot, ORB entry, 3-5 day rule, ADR filter | Kept ORB but restricted to first/last hour only. 15/30min candle. |
| **Kell** | SMA stack, exhaustion at >20% above 10 SMA, Wedge Pop entry | Replaced EMA stack with SMA stack. Added Bollinger squeeze as VCP equivalent. |
| **Velez** | Elephant Bar, Tail Bar, last-hour validation, 40/40/20 rule | Kept execution patterns. Dropped 20 SMA "center of gravity" as sole indicator. |
| **Camillo** | Catalyst/narrative as filter, exit when story breaks | Used as catalyst box (box 04), not standalone strategy. Reduced to filter. |

---

## Appendix -- What's Unique to This Playbook

- **SMA-only charting** -- no EMAs, no indicator mixing
- **Bollinger Bands** for squeeze and exhaustion -- replaces ATR extension table for visual traders
- **Put/call volume** as option flow confirmation before entry
- **First + last hour only** -- no mid-day entries. Hard bracket stops for hands-off management.
- **Long AND short** -- full RS/RW framework in both directions, with regime-gated deployment
- **Options with hard stop solutions** -- four methods for pairing options with defined exits
