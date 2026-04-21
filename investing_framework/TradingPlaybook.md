# Swing Trading Playbook

Relative strength momentum trading — 5–50 days — stocks and options.

| SPEC | |
|------|---|
| Timeframe | 5–50 days |
| Risk per trade | 0.5% of portfolio |
| Direction | Long AND short — RS/RW vs SPY in both directions |
| Execution windows | First hour (9:30–10:15) and last hour (15:15–15:45) only |
| Charting | 5, 10, 20, 50 SMA · Bollinger Bands · Put/Call volume ratio |
| Instruments | Stocks and options on liquid names (Vol > 1M, Price > $5) |

---

## Core Thesis

> Stocks with relative strength against SPY are being accumulated by institutions. Stocks with relative weakness are being distributed. These divergences persist for weeks — not days — because institutional capital moves slowly. Enter on the breakout, ride the follow-through, exit when relative strength breaks.

This thesis synthesises four edges:
- **Minervini:** Supply exhaustion (VCP) creates low-risk breakout entries
- **Bruzzese:** RS/RW vs SPY identifies institutional accumulation before the move
- **Kullamägi:** Episodic pivots and ORB entries capture the ignition moment
- **Kell:** SMA stack and price cycles define trend health and exhaustion

Adapted to your style: SMA-only charting, Bollinger Bands for volatility context, first/last hour execution only, options for leveraged expression with hard stops.

---

## Profit Mechanisms

Each trade must exploit an identified profit mechanism — a specific, repeatable market effect with documented evidence. This section defines the active mechanisms and the framework for adding new ones.

### Active Mechanisms

#### PM-01 — Breakout Momentum *(Grade A · ACTIVE)*

> Momentum persists because institutions can't deploy capital instantly and information diffuses slowly. The ignition event forces large funds to start building. They buy for weeks — not days.

| Attribute | Detail |
|-----------|--------|
| **Market effect** | Price drift following supply exhaustion (VCP) or information shock (EP) |
| **Why it works** | Institutional liquidity constraints + behavioural under-reaction to new information |
| **Academic basis** | Century momentum (Geczy & Samonov 2017); VCP confirmed in Minervini's empirical track record |
| **Timeframe** | 5–50 days (primary leg of the move) |
| **Signals** | VCP tightening + VDU, SMA stack intact, RS leading, volume-confirmed breakout |
| **Setups** | Type A (EP), Type B (VCP), Type C (SMA Reclaim) |
| **Structure** | Long call 45–60 DTE (IVR < 40%) · debit spread (IVR 40–70%) · stock + stop |
| **Invalidation** | Double close back inside the breakout range. RS breakdown vs SPY. |
| **Status** | Fully validated — execute per playbook |

#### PM-02 — Post-Earnings Announcement Drift *(Grade A · ACTIVE)*

> A strong earnings surprise creates 40–60 days of forced institutional accumulation. The gap is day 1, not the full trade.

| Attribute | Detail |
|-----------|--------|
| **Market effect** | Price drift for 40–60 days after an earnings surprise |
| **Why it works** | Institutional under-reaction + forced accumulation by funds tracking quality/momentum screens |
| **Academic basis** | Ball & Brown 1968; Bernard & Thomas 1989, 1993; confirmed across 100+ years and 40+ countries |
| **Timeframe** | 40–60 days from announcement |
| **Signals** | Gap ≥ 10% on 5–10× volume · closes top 25% of range · EPS beat top decile |
| **Amplifiers** | Larger EPS surprise = longer drift · Small/mid cap > large cap · Strong sector confirms |
| **Setups** | Type A (EP) — PEAD is not a separate setup; it is a momentum amplifier inside EP entries |
| **Structure** | Long ATM call 45–60 DTE · bull call vertical if IV spiked on gap |
| **Invalidation** | Gap fills within 3 days. RS flips negative. |
| **Status** | Fully validated — execute as Type A |

#### PM-03 — Pre-Earnings Anticipation *(Grade A · ACTIVE)*

> Markets begin incorporating earnings expectations 10–20 days before the report. IV expansion adds a vega tailwind to directional bets.

| Attribute | Detail |
|-----------|--------|
| **Market effect** | Price drift + IV expansion 10–20 days before earnings announcement |
| **Why it works** | Institutional positioning ahead of expected beat (Richardson & Veenstra 2006) |
| **Timeframe** | T-14 to T-1 (exit day before earnings — never hold through binary event) |
| **Signals** | RS stock in Stage 2 · beat ≥ 3 of last 4 quarters · holding 20 SMA at T-14 · IVR < 30% |
| **Structure** | ATM call or call diagonal. Buy at T-14, exit at T-1 regardless of P&L. |
| **Stop** | Close below 20 SMA or 50% premium loss — whichever first |
| **Invalidation** | Stock gapped unpredictably on any of last 4 reports. IVR > 40% at entry (premium too rich). |
| **Status** | Validated — execute with strict T-1 exit |

#### PM-04 — OTM Informed Flow *(Grade A · ACTIVE)*

> Informed investors prefer OTM options for leverage. Their activity in OTM puts vs calls predicts future stock returns.

| Attribute | Detail |
|-----------|--------|
| **Market effect** | Directional signal from unusual OTM option volume |
| **Why it works** | Informed traders use OTM options to maximize leverage on private information (Pan & Poteshman 2006) |
| **Timeframe** | 5–20 days (signal precedes the move by days to weeks) |
| **Signals** | OTM call vol > 3× OI in next 4 weeks · block prints in multi-leg call spreads in final 2h · put/call ratio on RS stock > 0.7 |
| **Role** | **Confirmation signal, not standalone entry.** Adds to watchlist priority when combined with a Type A/B/C setup. |
| **Status** | Active — used as scan filter and entry confirmation |

#### PM-05 — Relative Strength / Weakness Divergence *(Grade A · ACTIVE)*

> Stocks that hold up when SPY drops are being accumulated by institutions. Stocks that drop more are being distributed. The divergence persists for weeks.

| Attribute | Detail |
|-----------|--------|
| **Market effect** | Price divergence from broad market signals institutional accumulation or distribution |
| **Why it works** | Institutions build/exit positions over weeks. RS/RW is their footprint. (Bruzzese) |
| **Timeframe** | 5–50 days (the divergence is the holding period) |
| **Signals** | RS line at new highs vs SPY (longs) · new lows (shorts) · higher lows while SPY makes lower lows |
| **Role** | **Core selection filter.** Every trade in this playbook requires RS/RW confirmation. |
| **Setups** | Type A/B/C (longs via RS), Type D (shorts via RW) |
| **Invalidation** | RS flips — stock starts underperforming SPY on down days (longs) or outperforming on up days (shorts) |
| **Status** | Active — embedded in all selection and exit criteria |

### Embedded Mechanisms (Context Filters)

These are not standalone trades but are embedded in scanning, selection, or regime decisions.

| # | Mechanism | Role | Status |
|---|-----------|------|--------|
| 04 | Century Momentum | 12-month return filter in trend template (box 01) | Embedded in scan |
| 06 | Positive Drift on Indices | Cross-over to DRIFT portfolio when indices are in Stage 2 | Embedded in regime |
| 10 | VIX Mean Reversion | After VIX spike >25 + "lower high" → SPY/QQQ ORB entry (index trade only) | Active — index ETFs |
| 15 | Rebalancing Tailwind | Avoid new longs in final 3 days of quarter. Enter first week of new quarter. | Embedded in timing |
| 21 | S/R Reversal | Bollinger Band lower touch in uptrend as pullback entry signal (Type C) | Embedded in setup |
| 30 | Options Complexity Alpha | Use multi-leg structures (spreads, diagonals) over simple calls/puts for better risk-adjusted returns | Embedded in structure selection |
| — | Sentiment-Conditional TA | Technical signals work better in high-sentiment regimes. Use sentiment (VIX, put/call ratio, breadth) as a GO/NO-GO overlay for technical entries. *(Sentiment & TA, Hedge Fund Industry study)* | Embedded in regime |
| — | Option Momentum | Options with high recent returns continue outperforming for up to 5 years without reversal. Favour names where recent option returns were positive when selecting structures. *(Option Momentum, cross-sectional study)* | Embedded in structure |

### Validated Supplementary Mechanisms

#### PM-09 — Mean Reversion to Trend *(Grade A · ACTIVE — validated April 2026)*

> Oversold pullbacks within an uptrend revert to the moving average. The Bollinger lower band touch is a quantifiable confirmation signal for Type C entries.

| Attribute | Detail |
|-----------|--------|
| **Backtest** | N=53,400 BB lower touch events (2016–2026). 10d: +1.18% (53.8% WR). 20d: +1.70%. 60d: +2.25% (Sharpe 0.115). |
| **Signals** | Price touches lower BB while 50 SMA rising and price above 200 SMA. Volume declining. RS intact. |
| **Role** | **Confirmation layer for Type C entries.** BB touch validates the pullback is within trend. |
| **Scanner** | #10 (TTM Squeeze) |

### Research Pipeline

Mechanisms under evaluation. Full cards and assessments in `StrategyIdeasAssessment.md`.

| # | Mechanism | Grade | Status | Notes |
|---|-----------|-------|--------|-------|
| 06 | News-Driven Drift | B | Research | Build news-vs-no-news classifier |
| 07 | Retail Attention Contrarian | B | Research | Define "extreme attention" quantitatively |
| 08 | Earnings Volatility Crush | A | Research | Backtest: which underlyings stay within expected move? |
| 10 | Insider & Corporate Action | B | Research | Build SEC Form 4 screen |
| 11 | Short Squeeze | B | Research | Quantify SI% + DTC thresholds |

### Rejected Mechanisms

| # | Mechanism | Result | Date |
|---|-----------|--------|------|
| PM-08 | Overnight Reversal | No edge — 33,935 events, Sharpe -0.011 | April 2026 |

<!-- PM-06 through PM-11 full cards moved to StrategyIdeasAssessment.md -->
<!-- PM-08 Overnight Reversal rejected April 2026: 33,935 events, no edge at any horizon -->
<!-- PM-09 Mean Reversion promoted to Active (above) April 2026 -->


---

## How to Add a New Profit Mechanism

> Use the 4-step Outlier framework. Do not trade a mechanism until all four steps are validated. *(Trading_new.md)*

### Step 1 — Define the Profit Mechanism

Identify the SPECIFIC market effect you're trying to capture.
- Price movement: drift, momentum, breakout/down, mean reversion
- Volatility: expansion, contraction, crush, mean reversion
- Structural: hedging pressure, rebalancing flows, informed flow

### Step 2 — PM Behavior & Signals

Build tools to identify, measure, and time the mechanism.
- Document behavior and conditions — when does it appear? How strong?
- Build a signal list — test each signal against the PM
- Match the best signals; trim to the most predictive set

### Step 3 — Outline Fitting Structures

How to monetise it via the right trade structure.
- Create candidate structures (stock, options, spread, combination)
- Overlay portfolio risk rules and BP constraints
- Identify the best-fit structure for this mechanism and your execution windows

### Step 4 — Build the Strategy

Turn the best structure into a fully defined, testable strategy card.
- Isolate key inputs: instrument, structure, greeks, sizing, entry/exit rules
- Backtest → paper trade → live test (minimum 30 trades)
- Add to mechanism deck only after live validation

**Feedback loop:** After 30+ live trades, review. What changed? What didn't work? Update the mechanism card or archive it.

### Mechanism Card Template

```
#### PM-XX — [Name] *(Grade [A/B/C] · [ACTIVE/RESEARCH/EMBEDDED])*

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

## 01 — Market Regime

> 75% of stocks follow the general market. Context always comes first. *(Bruzzese)*

### GO — All Must Be Present

- SPY above 20 SMA and 50 SMA — both sloping upward
- VIX below 20 or falling after a spike
- Advancing > Declining stocks (breadth expanding)
- At least one sector showing clear RS vs SPY

### NO-GO — Any One Triggers Pause

- SPY below 50 SMA
- VIX > 30 or spiking sharply
- Declining > Advancing 2:1+
- 3+ consecutive stopped-out trades — regime has shifted
- Major macro event within 48h (FOMC / CPI / NFP)

### Direction Bias

| SPY Regime | Long Setups | Short Setups |
|-----------|------------|--------------|
| Above 20 + 50 SMA, both rising | Full deployment | Avoid — drift works against you |
| Above 50 SMA, 20 SMA flattening | Normal size | Selective — weak stocks in weak sectors only |
| Below 50 SMA | Avoid — bear market | Full deployment on RW stocks |

---

## 02 — Scanning & Stock Selection

### Weekly Scan (Weekend)

Run every weekend. Sets the watchlist for the week.

1. **Sector rotation check** — Which sectors gained RS vs SPY this week? Which are building bases?
2. **RS leaders** — Stocks holding near highs while SPY dipped. These are being accumulated.
3. **RW laggards** — Stocks making new lows while SPY holds. These are being distributed. *(Short candidates)*
4. **Put/call volume scan** — Unusual OTM call volume > 3× OI = informed upside anticipation. Unusual put volume on RS stock = squeeze fuel.
5. **Update watchlist** — Cap at 20 names. Rank by setup quality. Top 5 for live trading.

### Daily Scan (Pre-Market)

| Check | What to Look For |
|-------|-----------------|
| Gap-ups/downs | > 5% move with volume > 2× average — look for catalyst |
| RS divergence | Stock green while SPY red (or vice versa) — accumulation/distribution signal |
| SMA reclaim | Stock reclaiming 20 SMA after a pullback — potential Type C entry |
| Volume spike | RVOL > 1.5× before 10am — institutional activity starting |
| Bollinger squeeze | BB width contracting to multi-week low — breakout imminent |
| Earnings proximity | Remove any stocks reporting within 5 days — no binary risk |

### Stock Selection — 5-Box Checklist

All 5 must pass. One fail = skip.

**01 — Trend Template** *(Minervini, adapted to SMAs)*
- Price > 20 SMA > 50 SMA (for longs) · Price < 20 SMA < 50 SMA (for shorts)
- 50 SMA sloping in trade direction
- Price within 25% of 52-week high (longs) or 52-week low (shorts)
- Positive 12-month return for longs · negative for shorts *(century momentum filter)*

**02 — Relative Strength / Weakness** *(Bruzzese)*
- RS line at/near new highs vs SPY (longs) · new lows (shorts)
- Stock holds up when SPY dips (longs) · drops more when SPY dips (shorts)
- Outperforming/underperforming sector peers over 1M
- Higher lows while SPY makes lower lows = institutional support (longs)

**03 — Base Quality** *(Minervini + Kell, adapted)*
- Consolidation of at least 1–3 weeks (tight range)
- Volume contracting during the base (VDU — dry-up)
- ATR 0–6× from base (not extended / parabolic). ADR > 3%.
- SMA stack: 5 > 10 > 20 > 50, all sloping in trade direction
- **Bollinger Band squeeze:** BB width < 20-day average BB width = volatility contraction. Breakout from a squeeze is the highest-probability entry.

**04 — Catalyst** *(Camillo + PEAD)*
- Clear reason for the move: earnings beat, news, sector theme, social narrative
- PEAD (if earnings): gap ≥ 10% AND closed top 25% of day's range
- Put/call ratio: unusual call volume on RS stock = informed bid. Unusual put volume on weak stock = informed distribution.
- No binary event within 5 days

**05 — Risk Parameters**
- Stop defined before entry (base low / ORB low / 20 SMA)
- Stop ≤ 7% from entry (Minervini hard limit)
- Size = (0.5% × portfolio) ÷ (entry − stop)
- R:R ≥ 2:1 minimum

---

## 03 — Setup Types

### Type A — Episodic Pivot *(Kullamägi + PEAD)*

> The gap-up is day 1 of 40–60 days of institutional accumulation.

- Gap ≥ 10% (prefer 15%+) on 5–10× avg volume
- Fundamental catalyst: earnings beat, major news
- Closes top 25% of day's range — institutions buying into the close
- **Entry:** ORB above 15min candle high (morning window)
- Highest velocity — biggest winners. Also highest risk.

### Type B — VCP Breakout *(Minervini)*

> Supply exhaustion — each pullback shakes out weak holders. When the last seller is gone, any buying drives price explosively.

- 2–6 weeks of tightening range, 3+ contraction points
- Volume dries up in final days (VDU)
- **Bollinger squeeze visible:** BB width at multi-week low inside the base
- Breakout above pivot on 40–50%+ above avg volume
- **Entry:** ORB above 30min candle on breakout day

### Type C — SMA Reclaim *(adapted from Kell + Velez)*

> Institutions add to winning positions on pullbacks to moving averages.

- SMA stack intact: 5 > 10 > 20 > 50, all sloping up (longs)
- Pullback to 10 or 20 SMA on contracting volume
- **Elephant Bar:** Candle ≥ 2× average bar size, opens near/below SMA, closes above it near the high. Institutional bid confirmed.
- **Tail Bar:** Long lower wick stings the SMA and recovers — institutions stepped in.
- **Entry:** ORB above 30min candle on reclaim day. Last-hour entry valid if stock in top 25% of range at 15:15.

### Type D — Breakdown / Short *(Bruzzese RW, adapted)*

> Stocks with relative weakness against SPY in a weak regime are being distributed by institutions. Mirror the long playbook.

- Price < 20 SMA < 50 SMA. 50 SMA sloping down.
- RW line at new lows vs SPY. Drops more on SPY down days.
- Base forming below declining SMAs — supply being created
- **Entry:** ORB below 30min candle low on breakdown day
- **Stop:** Above the breakdown candle high or 20 SMA

### Short Framework — Signal Layers (Type D)

> Always have long AND short exposure. The short book operates independently — it captures the downside of the same academic edges.

**Structure:** Put debit spreads (not short stock). Defined max loss eliminates overnight gap risk and borrow problems. Sizing: 0.5% portfolio max loss = debit paid.

**Layer 1 — Universe Filter (always on):**

| Filter | Rule |
|--------|------|
| Trend | Price < declining 50D SMA |
| Short interest | < 20% of float (above 20% = squeeze risk) |
| Liquidity | Vol > 500K |
| Binary events | No earnings/FDA/M&A within 10 days |

**Layer 2 — Primary Trigger (at least one required):**

| Trigger | Signal | Hold | Backtest |
|---------|--------|------|----------|
| **Negative PEAD** (primary) | EPS miss (bottom 25% SUE) + close bottom 25% of range | 10–20d general, 40–60d strong | Strong: -12%/60d, 76.8% WR. General: -2.2%/10d then fades. |
| **RW Breakdown** | RW line at new lows vs SPY + sector Stage 3/4 | 5–30d | Scanner #14 |
| **Scanners:** #13 (Negative PEAD), #14 (RW Breakdown) | | | |

**Layer 3 — Conviction Upgraders (not standalone):**

| Signal | Rule | Finding |
|--------|------|---------|
| Accruals (Sloan 1996) | Top quartile accruals ratio | +3.38%/year L-S spread. Annual signal. |
| F-Score (Piotroski 2000) | F-Score <= 3 | Quality filter. ~0.5%/quarter. |
| 1st miss is strongest | Focus on first-miss quality, not consecutive count | Consecutive misses show *weaker* drift. |

**Key insight from backtest:** General misses drift -2.2% to day 10 then mean-revert. Only strong filtered signals (miss + gap<=-5% + bottom 25% close) persist to 60 days. Use 10–20d hold for general shorts, 40–60d only for strong.

### ATR Extension Table

| ATR Range | Stage | Action |
|-----------|-------|--------|
| 0–3 ATR | Early / Stable Trend | Sweet spot. Sustainable entry. |
| 3–6 ATR | Established Trend | Strong momentum. Diminishing R:R. |
| 7–10 ATR | Extended Trend | Climax risk. Only partial entries. |
| > 10 ATR | Parabolic / Exhausted | High reversion risk. Avoid entries. |

### Setup Priority

| Regime | Priority |
|--------|----------|
| Strong bull (SPY > 20 + 50, breadth expanding) | A > B > C. No shorts. |
| Moderate bull (SPY > 50, 20 flattening) | B > C > A. Selective shorts (Type D) on extreme RW. |
| Correction / bear (SPY < 50) | D only. No longs. Spreads only for options. |

### Scanner Cross-Reference

| Setup | Primary Scanners | View |
|-------|-----------------|------|
| Type A (EP) | #9 PEAD, #11 EP Gap, #2 5-Day Momentum | PEAD/EP |
| Type B (VCP) | #1 52W High, #3 1M Strength, #10 TTM Squeeze | Standard |
| Type C (Reclaim) | #3 1M Strength, #10 TTM Squeeze, #4 Volume | Standard |
| Type D (Short) | #13 Neg PEAD, #14 RW Breakdown | PEAD/EP, Options/Flow |
| Confirmation | #6 High Put, #7 High Call, UOA | Options/Flow |
| Intraday | #8 RVOL Spike (manual) | Intraday |

Full scanner configuration: `BarchartScreeners.md`

---

## 04 — Entry Execution

> You trade the first hour and the last hour. Nothing in between. Set orders, walk away.

### Trading Windows

| Window | Time | Purpose |
|--------|------|---------|
| Pre-market | 9:00–9:30 | Review regime, scan, set ORB alerts |
| **Morning ORB** | **9:30–10:15** | Primary entry window. 15min (Type A) or 30min (Type B/C/D) candle. |
| Dead zone | 10:15–15:15 | No entries. Manage stops only. |
| **Last hour ORB** | **15:15–15:45** | Secondary entry. Stock must be top 25% of range (longs) or bottom 25% (shorts). |
| Post-close | 16:00–16:15 | Log trades, review exits, update watchlist. |

### Entry Rules

- **Wait for the candle to fully close.** Never enter on a developing candle. The close counts.
- **Pre-set buy/sell stop limit orders** above (longs) or below (shorts) the ORB candle high/low. Not a manual click.
- 15min candle for Type A (EP) — moves fast. 30min candle for Type B/C/D.
- RVOL > 1.5× by 9:45 = strong signal. < 1.5× = reduce size or skip.
- If SPY gaps hard against your direction at open — skip morning entries, reassess at last hour.

### Last-Hour Entry (Velez validation)

- **Longs:** Stock in top 25% of day's range at 15:15. Held highs through mid-day (not a recovery).
- **Shorts:** Stock in bottom 25% of day's range at 15:15. Held lows through mid-day.
- Enter above/below high/low of first 15min candle of the last hour.
- RS/RW confirmation: stock outperforming/underperforming SPY in the final 2 hours.

---

## 05 — Stop Loss & Sizing

### Hard Stops — Non-Negotiable

> Due to limited intraday availability, all stops must be hard (bracket orders). No mental stops.

| Entry Type | Stop Placement |
|-----------|---------------|
| ORB entry | Below/above the entry candle low/high |
| VCP breakout | Below the base / consolidation low |
| SMA reclaim | Below the 20 SMA |
| EP (Type A) | Below the gap-open level |
| Short (Type D) | Above the breakdown candle high or 20 SMA |
| **Hard maximum** | **7% from entry — never exceeded** |

### Position Sizing

```
Max $ risk = 0.5% × Portfolio
Shares = Max $ risk ÷ (Entry − Stop)
```

### Options Sizing — The Hard Stop Problem

> Options don't work with tight hard stops in the traditional sense. The bid-ask spread and intraday vol can trigger a stop on the option even when the underlying hasn't breached the level.

**Solution: Define risk on the underlying, express via options.**

| Method | How It Works | When to Use |
|--------|-------------|-------------|
| **Stock stop, option position** | Set hard stop on the underlying price level. When triggered, close the option position at market. | Best for liquid names with tight spreads. |
| **Debit spread (defined risk)** | Max loss = debit paid. No stop needed — the spread defines the risk. | IVR 40–70%. When you can't monitor intraday. |
| **Long call/put with premium stop** | Close if option loses 50% of premium paid. | IVR < 40%. High conviction directional. |
| **OCO bracket on underlying** | Place an OCO (one-cancels-other) bracket order on a small stock position. When the underlying stop triggers, manually close the option. | For brokers that support conditional orders. |

**Rule:** If you cannot define a hard exit mechanism for the options trade, trade the stock instead. Undefined risk is not acceptable.

---

## 06 — Options Structure Selection

### By Setup Type and IVR

| Setup | IVR < 40% | IVR 40–70% | IVR > 70% |
|-------|-----------|-----------|-----------|
| **Type A (EP)** | Long ATM call 45–60 DTE | Bull call spread (negative vega offsets IV spike on gap) | Bull call spread only |
| **Type B (VCP)** | Long ATM call 45–60 DTE | Bull call spread | Bull call spread |
| **Type C (SMA Reclaim)** | Long ATM call 30–45 DTE | Bull call spread 30–45 DTE | Spread only |
| **Type D (Short)** | Long ATM put 45–60 DTE | Bear put spread | Bear put spread only |
| **Extended trend** | PMCC: deep ITM call 90+ DTE, sell OTM call 20 DTE | PMCC | Not recommended |

### Put/Call Volume as Confirmation

Before entering an options trade, check the option chain:

- **OTM call vol > 3× OI** in next 4 weeks = informed institutional bid → confirms long setup
- **OTM put vol > 3× OI** = informed downside anticipation → confirms short setup or warns against long
- **IV term structure steep** (6M IV >> 1M IV) = market overstating long-term risk → long calls are cheap
- **IV term structure inverted** (near-term IV > long-term) = event risk priced in → use spreads, not naked longs

---

## 07 — Trade Management & Exits

### Profit-Taking Staircase

| Stage | Trigger | Action |
|-------|---------|--------|
| Entry | ORB fires | Full size. Hard stop placed immediately. |
| 1st take | 1.5–2R | Close 30–50%. Move stop to break-even. |
| 2nd take | > 4R | Close 30% more. Trail stop on 5 SMA. |
| Runner | — | Hold until exit signal fires. |

**3–5 day rule** *(Kullamägi):* After 3–5 strong consecutive days, take the first partial regardless of R level. Momentum exhaustion is coming.

### Exit Signals — Any One Fires = Close Remainder

| Signal | Condition | Source |
|--------|----------|--------|
| **5 SMA** | 2nd consecutive daily close below 5 SMA (longs) / above (shorts) | Kullamägi |
| **ATR candle** | Single candle > 1.5× ATR(14) against position | Institutional exit signal |
| **20 SMA** | Daily close below 20 SMA on above-avg volume | Kell / Bruzzese |
| **Bollinger break** | Close outside upper BB after extended run (longs) = exhaustion | Bollinger + Kell exhaustion |
| **RS/RW breakdown** | Stock starts underperforming SPY on down days (longs) / outperforming on up days (shorts) | Bruzzese |
| **Narrative break** | Catalyst negated — bad news, guidance cut, thesis invalidated | Camillo |
| **Extension** | Stock > 20% above 10 SMA — scale out, don't add | Kell |
| **Time stop** | 50 days elapsed, thesis not playing out | Capital redeployment |

**Rule: Never let a 2R winner turn into a loss.**

### Bollinger Band Exit Rules

- **Upper BB touch after 3+ weeks of trend:** Begin scaling. Stock is 2 SD extended.
- **Lower BB touch on pullback in uptrend:** Hold if 20 SMA is rising. Only exit on close below 20 SMA.
- **BB squeeze re-forms after breakout:** New consolidation forming — tighten stop to base low and watch for continuation or reversal.

---

## 08 — Daily Execution Checklist

### Pre-Market (9:00)

- [ ] SPY regime check: above 20 + 50 SMA? Breadth?
- [ ] VIX level and direction
- [ ] Review watchlist: any gap-ups with catalyst? Any RW breakdowns?
- [ ] Remove stocks reporting earnings today/tomorrow
- [ ] Set ORB alerts on top 5 watchlist names
- [ ] Pre-set buy/sell stop limit orders for pre-identified setups

### First Hour (9:30–10:15)

- [ ] Wait for 15/30min candle to fully form
- [ ] Confirm volume: RVOL building? Breakout candle above avg?
- [ ] Check SPY direction: supporting the move?
- [ ] Orders trigger automatically via pre-set brackets
- [ ] At 10:15 — stop. No new entries until last hour.

### Dead Zone (10:15–15:15)

- [ ] No entries. Monitor hard stops only.
- [ ] Note stocks holding well / showing RS for last-hour review
- [ ] Check: any positions hit profit targets? → set partial exit orders for last hour

### Last Hour (15:15–15:45)

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

## 09 — Non-Negotiable Rules

1. **Rules, not feelings.** Every decision is pre-defined. If it's not in this playbook, don't do it.
2. **The close counts.** Never enter on a developing candle. Wait for the full 15/30min bar.
3. **Hard stops only.** You're not watching the screen mid-day. Bracket orders are mandatory.
4. **Never add to a loser.** One stop = close the full position. No averaging down.
5. **Exit when premise breaks.** Don't adapt a failing trade. If RS flips, the trade is over.
6. **Respect the timeframe.** 5–50 days. No intraday management between the windows.
7. **Never fight the trend.** Long in bull. Short in bear. Not both at once.
8. **3 failures = pause.** Three consecutive stops means the regime shifted. Reassess.
9. **Options must have defined exits.** If you can't set a hard stop mechanism, trade the stock.
10. **Patience is the edge.** Wait for tight, high-probability setups. Forcing trades destroys edge.

---

## 10 — Charting Setup

### Indicators

| Indicator | Setting | Purpose |
|-----------|---------|---------|
| **SMA 5** | Close | Short-term momentum / exit trail |
| **SMA 10** | Close | Trend reference / secondary exit |
| **SMA 20** | Close | Primary trend filter and stop reference |
| **SMA 50** | Close | Regime filter / long-term trend |
| **Bollinger Bands** | 20 period, 2 SD | Volatility squeeze detection + exhaustion |
| **Volume** | With 50d average overlay | Confirm breakouts, detect VDU |
| **RS line vs SPY** | Custom or broker-provided | Core selection tool — RS/RW divergence |
| **Put/Call ratio** | Options chain | Informed flow confirmation |

### Why SMAs (Not EMAs)

- SMAs are smoother and lag more than EMAs — this is a feature, not a bug. You're a swing trader with limited screen time. SMAs filter noise better for end-of-day decisions.
- The 20 SMA is the institutional standard. Velez's "center of gravity," Minervini's trend template, Bruzzese's regime filter — all reference SMAs.
- Consistency: one indicator type across all timeframes. No mixing EMA and SMA signals.

### Why Bollinger Bands

- **Squeeze detection:** When BB width contracts to a multi-week low, a breakout is imminent. This is the visual equivalent of Minervini's VCP — supply exhausted, volatility compressed, ready to expand.
- **Exhaustion detection:** Stock touching or exceeding the upper BB after a multi-week trend = 2 SD extended. Begin scaling, not chasing.
- **Pullback context:** Stock touching lower BB during an uptrend = oversold within the trend. If 20 SMA is still rising, this is a Type C entry opportunity, not an exit.

---

## Appendix — Trader Attribution

What was taken from each source and what was adapted.

| Trader | Taken | Adapted |
|--------|-------|---------|
| **Minervini** | Trend Template, VCP, 7% hard stop, volume dry-up, pivot breakout | Replaced 150/200 SMA template with 20/50 SMA. Dropped fundamental acceleration screen. |
| **Bruzzese** | RS/RW vs SPY as core edge, sector confirmation, regime filter | Extended to short side (RW for short setups). Core thesis of the playbook. |
| **Kullamägi** | Episodic Pivot, ORB entry, 3–5 day rule, ADR filter | Kept ORB but restricted to first/last hour only. 15/30min candle, not 1/5/60min. |
| **Kell** | SMA stack, exhaustion at >20% above 10 SMA, Wedge Pop entry | Replaced EMA stack with SMA stack. Added Bollinger squeeze as VCP equivalent. |
| **Velez** | Elephant Bar, Tail Bar, last-hour validation, 40/40/20 rule | Kept execution patterns. Dropped his 20 SMA "center of gravity" as sole indicator. |
| **Camillo** | Catalyst/narrative as filter, exit when story breaks | Used as catalyst box (box 04), not as standalone strategy. Reduced to filter, not entry. |

### What's Unique to This Playbook

- **SMA-only charting** — no EMAs, no indicator mixing
- **Bollinger Bands** for squeeze and exhaustion — replaces ATR extension table for visual traders
- **Put/call volume** as option flow confirmation before entry
- **First + last hour only** — no mid-day entries. Hard bracket stops for hands-off management.
- **Long AND short** — full RS/RW framework in both directions, with regime-gated deployment
- **Options with hard stop solutions** — four methods for pairing options with defined exits
