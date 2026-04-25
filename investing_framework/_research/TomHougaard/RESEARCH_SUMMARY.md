# Hougaard Research — Systematic Techniques Summary

Extracted from 15 PDFs authored by Tom Hougaard. Focus: concrete, codeable rules only.
All times are local exchange time unless stated otherwise.

---

## Named Strategies

### School Run Strategy (SRS)

**Instrument:** DAX index (primary). Concept tested but not recommended for FTSE.
**Time frame:** 15-minute chart
**Origin:** Market-maker order-flow hypothesis — first 30 minutes used to execute urgent client orders; the "real" trend emerges after.

**Setup Rules:**
1. DAX opens at 09:00 Frankfurt local time (08:00 UK / 03:00 New York EST).
2. Identify the **2nd 15-minute candle** — the bar that runs from 09:15 to 09:30 Frankfurt time. This is the **signal bar**.
3. Mark a horizontal line 2 points **above** the signal bar's high and 2 points **below** the signal bar's low.
4. Place a **buy stop order** 2 points above the high; place a **sell stop order** 2 points below the low.
5. Both orders are live simultaneously (OCO implied); whichever triggers first becomes the trade.

**Entry Rules:**
- If DAX trades **above** the high of the signal bar + 2-point buffer → go long.
- If DAX trades **below** the low of the signal bar − 2-point buffer → go short.
- Both long and short signals can trigger in the same session; Hougaard takes both if each triggers.
- No time cutoff is explicitly specified; he has been filled in the afternoon on this approach.

**Stop Loss Rules (three options, choose based on bar size):**
- **Rule 1:** Stop above/below the extremes of the signal bar. (Often impractical — bar can be very wide.)
- **Rule 2:** Fixed 40-point stop loss. Calibrated when DAX was near 12,000; scale proportionally if DAX is significantly higher/lower.
- **Rule 3:** Stop at the midpoint of the signal bar (± trader discretion). Use a 5-minute chart to find a logical nearby level.

**Exit Rules:**
- No mechanical exit rule. Hougaard reads the chart and accepts the outcome.
- When the trade moves against him and then returns to profit, he sometimes **doubles the position** and tightens the stop.
- Move stop to breakeven as soon as reasonable profit is achieved; trail thereafter.

**Statistical note:** ~57% of School Run trades are profitable (implied: ~43% are straight losers). Non-linear income profile with good runs and losing runs.

---

### Advanced School Run Strategy (ASRS)

**Instrument:** DAX index
**Time frame:** 5-minute chart
**Relationship to SRS:** Standalone strategy, not a replacement. Higher signal frequency due to shorter time frame.

**Setup Rules:**
1. DAX opens at 09:00 Frankfurt local time.
2. Count 5-minute candles from the open: candle 1 = 09:00–09:05, candle 2 = 09:05–09:10, candle 3 = 09:10–09:15, **candle 4 = 09:15–09:20**. This is the **primary signal bar (4th bar)**.
3. Once candle 4 is complete, mark the high and low.
4. Place **buy stop** 2 points above the 4th-bar high; place **sell stop** 2 points below the 4th-bar low.
5. If the 4th bar's range is very small (narrow, overlapping bars), **wait for candle 5 (09:20–09:25)** and use it instead. Candle 5 has lower priority than candle 4 but is preferred in low-volatility, tight-range open conditions.

**Entry Rules:**
- Above 4th-bar high + 2 pts → long.
- Below 4th-bar low − 2 pts → short.
- Buffer of 1 point (instead of 2) is also used by some practitioners.
- Both long and short orders are placed simultaneously; the non-triggered side acts as the stop loss for the triggered side.
- Research validity: Hougaard has not formally validated ASRS signals that appear more than 2 hours into the trading session. In practice he focuses on the first 2 hours (09:00–11:00 Frankfurt).

**Stop Loss:**
- The stop loss for the long position = the short entry price (the low order level).
- The stop loss for the short position = the long entry price (the high order level).
- This creates a hard symmetric stop defined by the range of the signal bar + 4 points (2-point buffer each side).

**Exit Rules:**
- No mechanical profit target.
- Move stop to breakeven at first opportunity once position is profitable.
- Hougaard's rule of thumb: "If the position came back to entry after being against me, I add to my position." — interpreted as: a return to entry after adversity is treated as a re-entry or add signal, not a close signal.
- If filled on both long and short on the same day, accept both outcomes independently.

**Risk Management:**
- When the 4th-bar range is unusually large (e.g. >30 points in DAX), reduce position size to 50%.
- Pre-define the maximum number of ASRS entries per morning before the session opens.

---

### 1BN / 1BP — First Bar Negative / First Bar Positive (FTSE Strategy)

**Instrument:** FTSE 100 index (primary). Tested on Dow and Nasdaq August 2024 with promising Nasdaq results. DAX: not recommended.
**Time frame:** 5-minute chart
**Data sample:** 1,286 trading days (~58 months) for FTSE.

**Statistics:**
- First bar is positive 50.6% of the time; negative 49.4% of the time.
- In FTSE over 58 months: both 1BN and 1BP offer trades with defined directional edge (detailed in 39-page research doc).
- August 2024 FTSE 1BP: 10 of 21 trading days triggered; 60% win rate; 194 points profit on winners vs 31 points lost on losers.
- August 2024 Nasdaq 1BP: 10 of 22 days triggered; 8 of 10 were "great" or better.
- DAX 1BP: Only 2 of 12 August 2024 days produced tradeable results. Not recommended.
- Dow 1BP: Mixed — 4 of 11 outstanding, 4 at breakeven, 3 losses.

**FTSE Open Time:** 08:00 UK time (= 09:00 Frankfurt time = 03:00 EST). This is the official London Stock Exchange open.

**1BN (First Bar Negative) Rules:**
- After the 08:00 candle closes **negative** (close < open):
  - **Long signal:** DAX/FTSE trades **below** the low of the 1BN bar → go long below that low.
  - **Short signal:** DAX/FTSE trades **above** the high of the 1BN bar → go short above that high.
- Long is the statistically stronger signal for FTSE.
- Short above the 1BN high is less reliable statistically.

**1BP (First Bar Positive) Rules:**
- After the 08:00 candle closes **positive** (close > open):
  - **Short signal:** FTSE trades **below** the low of the 1BP bar → go short below that low.
  - No corresponding mechanical long signal on 1BP (market may "run away" higher without providing an entry).

**Stop Loss Rules:**
- Default: **Stop = length of the 1st bar's range** (high − low of the signal bar, placed on the opposite side of the trade).
- If the 1st bar range is unusually large (e.g. >30 points FTSE), reduce position size by 50% and keep nominal stop at bar length.
- Specific example: 1BN on 06/08/2024: bar range 43 points → 50% position size, 43-point stop.
- Hougaard's real-time stop management: "I move my stop loss by 1 point for every 1 point I have in open profit." (active trailing rule — 1:1 trail)
- Move stop to breakeven once 6–10 FTSE points of profit are achieved.

**Exit Rules:**
- No fixed profit target (explicitly stated).
- Target area: "Where the market just came from on the previous swing" (previous day's close, yesterday's low, prior S/R level).
- FTSE tends to "back and fill" — rarely sustains momentum; take profits more actively in FTSE vs DAX.
- Add to winner when stop on first entry can be moved to breakeven.
- "When I am 10 points in profit, I consider taking profit or adding." (Not a hard rule — conditional on momentum feel.)
- High of the 1st bar is a reference target for 1BN long trades.

---

### Rule of 4 (FOMC Strategy)

**Instrument:** Dow Jones (primary), SP500, Nasdaq (untested but likely correlated).
**Time frame:** 10-minute chart
**Trigger event:** FOMC interest rate announcement (and other major scheduled news events)

**Rules:**
1. At the time of the FOMC announcement, begin counting 10-minute bars from bar 1.
2. Wait for the **4th 10-minute bar** (the bar that closes ~40 minutes after the announcement) to complete.
3. Mark the **high and low of bar 4** as the signal bracket.
4. Place **buy stop** above the high of bar 4; place **sell stop** below the low of bar 4.
5. Whichever side triggers first is the trade.
6. Both sides can trigger independently in the same session.

**Entry Rules:**
- If market trades above bar 4 high → long.
- If market trades below bar 4 low → short.
- Use buffer (unstated exact amount; consistent with 2-point convention used elsewhere).

**Stop Loss:**
- Not explicitly stated in the document. Implied: stop on the long = bar 4 low; stop on the short = bar 4 high (symmetric bracket = built-in stop).

**Exit Rules:**
- Not mechanical. "I take profit when I sense the momentum is running out of steam."
- Entry is mechanical; exit is discretionary.

**Notes:**
- Hougaard reported no losing trades from the Rule of 4 across multiple live FOMC trades. No guarantee of continuation.
- Applicable to any large scheduled macro news event (not just FOMC). Tested on FOMC dates: 22 March 2023, 1 February 2023.

---

### Fishing Technique (Fake-Out / Fake-Down)

**Also called:** Fake-Out, Fake-Down, Reverse-Breakout
**Instruments:** Any index or FX pair; preferred on 10-minute chart and above (5-minute minimum).
**Time frame:** 5-minute and above; Hougaard primarily uses 10-minute.

**Setup Identification:**
- Market forms a **double top** or **double bottom** pattern.
- Price probes **above the old high** (for double top) or **below the old low** (for double bottom) — running stop orders.
- The probe is not too far beyond the old extreme (just enough to run stops; if it extends far, the setup is invalid).

**Buy Signal (Fishing for Double Bottom):**
1. Market makes a low, then rallies, then makes another low (second low = the "fishing expedition").
2. The second low goes below the first low (runs stops).
3. The bar with the **lowest low** is the key bar.
4. Condition: the market must then **trade above** the high of the bar with the lowest low AND **close above** that bar's high.
5. Entry: buy when the current bar closes above the high of the lowest-low bar.

**Sell Signal (Fishing for Double Top):**
1. Market makes a high, then falls, then makes another high (second high = the "fishing expedition").
2. The second high goes above the first high (runs stops).
3. The bar with the **highest high** is the key bar.
4. Condition: the market must then **trade below** the low of the bar with the highest high AND **close below** that bar's low.
5. Entry: sell short when the current bar closes below the low of the highest-high bar.

**Stop Loss:**
- Not specified precisely. Context: "stop where you would have been wrong." Implied: above the extreme of the fake-out bar for sells; below for buys.

**Target:**
- "Where the market just came from on the previous swing" — the other end of the trading range.

**Note:** This is also called a "Fake-Out" (upside probe that fails) and "Fake-Down" (downside probe that fails). Works across many time frames.

---

### Buy On Close / Sell On Close (Scalping Entry)

**Instrument:** DAX, any index; 5-minute chart scalping.
**Time frame:** 5-minute chart primary; applicable to others.

**Buy On Close Criteria:**
1. Current bar has a **strong bullish close** — close is at or near the high of the bar with minimal or no upper tail.
2. The bar's range is **larger** (expanding) than the preceding 2–3 bars — range expansion confirms conviction.
3. The bar is in the direction of the broader context (buy signals are better when cleared of nearby resistance).
4. Entry: on the close of the qualifying bar (at bar close).
5. Stop loss — three options:
   - Preferred: below the **opposite side of the entry bar** (below the bar's low for buys).
   - More aggressive: below the **halfway point of the bar** (higher risk of stop-out; smaller monetary risk).
   - Conservative: below the **day's low** (more room; less likely to be stopped out).

**Sell On Close Criteria:**
1. Current bar has a **strong bearish close** — close is at or near the low of the bar with minimal or no lower tail.
2. The bar's range is **larger** (expanding) than the preceding 2–3 bars.
3. Not an inside bar (range must be expanding, not contracting).
4. Entry: on the close of the qualifying bar.
5. Stop loss: above the most recent high (or above the high of the bearish entry bar).

**Context Filter:**
- A qualifying bar at major support/resistance is a **lower-quality setup** (sellers sitting at resistance will fight your long).
- Best Buy On Close signals come after the market has cleared a sideways congestion zone.
- A Sell On Close after the market has already trended for many bars is an **exhaustion bar** — do not enter.
- Inside bars (range contracting) do not qualify.

**Volatility-Adaptive Scalp Parameters:**

| Dow Daily True Range | Approach | Stop | Target | Win Rate |
|---|---|---|---|---|
| < 300 points | Large stop / small target | 20 pts | 5 pts | ~90% |
| 300–500 points | Balanced | 25 pts | 25 pts | ~60% |
| > 500 points | Small stop / large target | 10 pts | 50 pts | ~25–30% |

- In a **very low volatility** environment: use large stops and small targets; high win rate; occasional large loss.
- In a **very high volatility** environment: use small stops and large targets; low win rate; occasional large win.

---

### 4-Bar Fractal

**Instruments:** Any freely traded instrument.
**Time frame:** Any (longer time frames produce more reliable signals).
**Origin:** Attributed to Dr David Paul.

**Buy Signal (4-Bar Fractal Long):**
- Label the current bar as bar 1, the previous bar as bar 2, two bars ago as bar 3, three bars ago as bar 4.
- Condition 1: `Close(bar 1) > High(bar 2)`
- Condition 2: `Close(bar 1) > High(bar 4)`
- If both conditions are met on the close of bar 1 → **buy signal**.

**Sell Signal (4-Bar Fractal Short):**
- Condition 1: `Close(bar 1) < Low(bar 2)`
- Condition 2: `Close(bar 1) < Low(bar 4)`
- If both conditions are met on the close of bar 1 → **sell signal**.

**Usage:**
- Used standalone as a trend-change entry.
- Most powerful when combined with **divergence** (stochastics or other oscillator diverging from price) — the divergence provides the setup; the 4-bar fractal provides the trigger.
- Example: market makes lower lows on price but higher lows on stochastics (bullish divergence) → wait for 4-bar fractal buy signal to confirm.

---

## Price Action Rules

### Trend Definition Rules

- **Up-trend:** Market makes **higher highs AND higher lows** in sequence.
- **Down-trend:** Market makes **lower highs AND lower lows** in sequence.
- **Trend change from up to down requires:**
  1. Market makes a lower high.
  2. Market makes a lower low.
  3. Market surpasses (closes below) that lower low.
  → Trend change confirmed.
- The reversal of the above logic applies for down-trend → up-trend transitions.

### 3-Bar Break / 3-Day Swing Rule

- Look at the highest high and lowest low of the **previous 3 bars** (days on daily chart; applicable to any time frame).
- If price during the current bar **breaks above** the 3-bar high → **bullish trend signal**.
- If price during the current bar **breaks below** the 3-bar low → **bearish trend signal**.
- Used on weekly and daily charts to establish directional bias for the week/day.
- Bias from weekly chart guides which direction to look for entries on lower time frames.

### Moving Average for Trend Filter

- **89-period moving average** (any time frame).
- Price **above** 89 MA → trend is **up** on that time frame.
- Price **below** 89 MA → trend is **down** on that time frame.
- If direction relative to 89 MA is ambiguous → no trending condition on that time frame.

### Engulfing Bar Rules

- **Definition:** The body of the current bar fully engulfs the body of the previous bar (shadows not considered).
- **Bearish Engulfing:** Current bar's body opens above and closes below the previous bar's body.
  - Combined, the two bars resemble a Shooting Star.
  - Does not require confirmation.
- **Bullish Engulfing:** Current bar's body opens below and closes above the previous bar's body.
  - Combined, the two bars resemble a Hammer.
  - Requires a prior reasonable trend in place.

**Entry options (3 levels of certainty):**
1. **Aggressive:** Enter mid-bar on the assumption the engulfing bar will complete; exit immediately if it does not.
2. **Normal:** Wait for the engulfing bar to close completely, then enter.
3. **Cautious:** Wait for the next bar to trade through the high (for bullish) or low (for bearish) of the engulfing bar.

**Preferred time frames:** 10-minute and above; works on FTSE, DAX, SP500, Dow.

**Exit after Engulfing:**
- If market immediately runs in the anticipated direction → move stop to breakeven.
- If a trending move develops → trail stop progressively.
- If a Doji or other reversal pattern forms → tighten stop immediately.
- If volume spikes into old support/resistance → tighten or take profit.
- A good 10-minute engulfing setup can run for a full hour or more.

### Extended Bar / Marubozu / Road Map Bar Rules

**Definition:** A bar with little or no tail (shadow) that is visibly larger in body than preceding bars. Also called: Marubozu, Extended Bar (EB), Road Map Bar, Aggressive Bar, Hit and Run bar.

- Extended bar in the direction of trend: **conviction signal** — expect continuation.
- Extended bar that appears **after the market has already trended for 10–15 bars** → likely **exhaustion bar**; do not enter new positions in the trend direction at this bar; watch for reversal.
- Extended bar against the trend = potential "key reversal."
- **Stop placement on Extended Bar:** Use the **midpoint of the bar** as the stop loss (trail stop to midpoint after entry). If confidence in position is high, midpoint provides adequate cover.
- After an Extended Bar, a normal 1–2 bar retracement is expected before trend resumes; do not exit on the retracement alone.

### Spinning Top / Doji Rules

- **Spinning Top:** Small real body, long shadows on both sides; market has little directional conviction. May warn of pause or end of trend.
  - Trading rule: If market has been trending lower (bearish bias), a Spinning Top followed by a close below the Spinning Top's low is a continuation sell signal.
- **Doji (Long-Legged / Doji Star):** Open and close at same price, long shadows. Reversal warning; do not sell short on the Doji itself — wait for confirming weakness.
- **Gravestone Doji:** Closes at its low after reaching up. Bearish reversal signal; sell short only if market then closes below the Gravestone Doji low.
- **Dragonfly Doji:** Closes at its high after reaching down. Bullish; buy orders can be placed where the market ran stops (below the Dragonfly low) — this is where other traders placed their stops.

### Umbrella Pattern Rules

- **Hammer** (at market bottom): Small body, no upper shadow, lower shadow ≥ 2–3× body length. Close **above** prior bar's close is preferred. Requires a prior downtrend. Confirmation: next bar closes above the Hammer's high. Bullish reversal.
- **Hanging Man** (at market top): Same structure as Hammer but appears in an uptrend. Bearish if next 1–3 bars close below the Hanging Man's low.

### Shooting Star / Inverted Hammer Rules

- **Shooting Star:** Small body, long upper shadow, forms at a **new high in an uptrend**. Bearish reversal. Trade only on new highs in uptrend.
- **Inverted Hammer:** At market bottom; small body, long upper shadow. Signals strong support; requires confirmation before buying.

### ABCD / Harmonic Retracement

- Market makes move A to B, pulls back to C, then continues.
- If the retracement from B to C equals the retracement from the last prior pullback (measured price distance), the point D can be forecasted.
- Entry: place buy or sell order at the projected point D (harmonic/equal retracement).
- Stop loss: placed outside the range of the prior retracement ("over-balance" concept — if correction exceeds the last correction, trend may have changed).
- Also used as a target: if market has reached point D and reverses, the target is the prior swing high/low (point B).

### "Who is Making Money" Framework (Scalping)

For each 5-minute bar, ask four questions:
1. Are **limit order buyers** (bought below the prior bar's low) making money?
2. Are **stop order sellers** (shorted below the prior bar's low) making money?
3. Are **limit order sellers** (shorted above the prior bar's high) making money?
4. Are **stop order buyers** (bought above the prior bar's high) making money?

If only stop order buyers and limit order buyers are making money → bullish; market is likely to continue higher.
If only stop order sellers and limit order sellers are making money → bearish.

### Three Pushes / Wedge Pattern

- Market makes **3 successive pushes** in the same direction.
- The 3rd push tests the 2nd push high (double-top test at channel top) → potential reversal.
- Combined with a bull channel, the 3 pushes form a **Wedge top** (higher high double top on a trend channel boundary).
- Trade: short below the low of the 3rd push if context supports (overall structure bearish).

### Double Top / Double Bottom Reversal

- Before trading a reversal at support/resistance, look for the market to test the level **twice**.
- First reversal at a strong trend: often a minor pause that fails (market resumes).
- Second reversal at the same level has higher probability of becoming a swing reversal.
- Rule: "Suspect must go lower to find buyers" after a second failed high.

### Bar-by-Bar Logic (15-Minute Time Frame)

Rules derived from bar-by-bar analysis in the Price Action Higher Time document:

- In a **downtrend**: no bar should close above the high of the prior bar. The first bar that closes above a prior bar's high is the first bullish signal.
- In a **downtrend**: every bar should make a low below the prior bar. A bar that fails to make a new low signals momentum loss.
- If a bar's **low is never tested** after a strong bullish bar, the trend is intact until that low is taken.
- "Closing below a prior bar" is more significant on a higher time frame (15-minute > 5-minute > 1-minute).
- Gap down + early bullish bars → 48% probability the gap fills on the same day (applicable to Dow and FTSE).

---

## Stop Loss Framework

### Core Rules

- **Never move a stop loss further away** from the entry. Stops are a one-way street: for longs, only raise the stop; for shorts, only lower it.
- Always place stops **before** entering a trade, not during the trade under stress.
- Every trader has a stop loss — even those who claim otherwise (they have a mental number at which they will exit).

### Stop Loss Techniques

**1. Hard Stop (fixed points)**
- Simplest; used on prop desks. Flaw: does not adapt to volatility.
- Hougaard does not recommend for retail traders. Used when a firm's risk manager imposes a maximum loss per trade.

**2. Volatility Stop — ATR Method (Method I)**
- Measure the **14-period ATR on the daily chart** (High − Low average over 14 days = daily range average).
- For **intraday scalping** (1-minute or 5-minute chart): use **20% of daily ATR** as stop loss.
  - Example: FTSE daily ATR = 55 points → intraday stop = 55 × 0.20 = 11 points.
  - Example: DAX daily ATR = 120 points → intraday stop = 120 × 0.20 = 24 points.
- For **swing trades** (multi-day): use **50% of daily ATR** as stop loss.
  - Example: FTSE daily ATR = 55 points → swing stop = 28 points.
- Target (for ATR stop approach): 150% of the stop distance.
  - 20% ATR stop → 30% ATR target.
  - Move stop to breakeven when position exceeds 10-point profit (FTSE).

**3. Volatility Stop — Counter-Move Method (Method II)**
- Observe the **size of prior corrections** in the current trend.
- If corrections have been 9–12 points, set stop just outside that range.
- Trail stop to stay approximately 15 points from current market price (never moving the stop against the trend).
- This is the "over-balance" concept: if the current correction **exceeds** the size of prior corrections, the trend structure has changed ("over-balanced") — exit the position.

**4. 2-Bar Stop Loss**
- If long: stop = just above the **high of the two bars preceding the current bar**.
- After each bar closes, reassess: the 2-bar stop is the highest high of the two most recently completed bars.
- Trail by updating after each bar closes if the new 2-bar high is lower than the previous 2-bar stop.

**5. Marubozu / Extended Bar Stop**
- After a long Marubozu/Extended bar in your favour, normal trailing stops lag too far behind.
- Use the **midpoint of the Marubozu bar** as the stop loss.
- For high-confidence positions, the midpoint provides adequate protection; a close through the midpoint constitutes the exit signal.

**6. Indicator-Based Exit Stop**
- Use a fast oscillator (RSI, Williams %R, Stochastics, CCI) to signal exit.
- When the oscillator generates a counter-signal while you are in a profitable position → exit the trade.
- Advantage: often signals exit before the 2-bar stop is triggered; reduces give-back.
- Preferred oscillators: Stochastics (12-26-9 MACD for trend-following stop approach also demonstrated).

### FTSE-Specific Stop Guidance

- FTSE "back and fills" (overlapping bars) frequently; trail stops aggressively.
- Move stop to breakeven once 6–10 FTSE points in profit.
- Do not attempt large adds in FTSE; the index rarely sustains momentum for pyramid adding.
- When stop is at breakeven and market is slow and directionless → consider closing and redeploying elsewhere.

### DAX-Specific Stop Guidance

- DAX is more momentum-oriented than FTSE; wider stops are tolerated because moves are larger.
- Rule of thumb from School Run: 40-point stop when DAX is near 12,000.
- Scale the stop proportionally to DAX level at time of trade.
- When ASRS bar range is very large (>30 points), reduce size by 50% and keep the stated stop.

---

## Time & Session Rules

### Daily Session Timeline (London / UK times)

| Time (UK) | Event |
|---|---|
| 22:00 prior | Asian markets active; quiet for European instruments |
| 06:00 | FX volatility increases as Frankfurt/European traders arrive |
| 07:00 | DAX futures open on EUREX |
| 08:00 | All European stock markets open; FTSE 100 opens |
| 08:00–10:00 | First busy period; primary trading window for European indices |
| 09:00 | DAX cash open (Frankfurt) = FTSE open + 1 hour |
| 09:15–09:30 | School Run signal bar on DAX 15-minute chart |
| 09:15–09:20 | ASRS 4th-bar window on DAX 5-minute chart |
| 09:20–09:25 | ASRS 5th-bar window (fallback if 4th bar range too narrow) |
| 10:00 | European data releases; first 2 hours of European session over |
| 11:00 | European traders wind down; pre-lunch quiet period begins |
| 11:00–13:00 | Lower volatility; reduced trading opportunity (not zero) |
| 13:30 | US economic data releases (major: NFP, GDP, etc.) |
| 14:30 | US markets open; first few minutes typically volatile/whippy |
| 14:30–15:00 | **Highest volatility window in Dow** (avg 120 pts on 10min bars when TR > 500) |
| 15:00 | Secondary US data releases (ISM, business surveys) |
| 16:00–16:30 | DAX and FTSE close for official trading |
| 16:30 | European stock markets officially closed |
| 19:00 | FOMC statements released (when applicable) |
| 21:00 | US stock markets official close |
| 21:30 | US equity futures reopen |

### Intraday Volatility by Session (Dow, 10-minute bars)

| Day True Range | 14:30–15:00 | 15:00–16:00 | 17:00–19:00 | 20:30–21:00 |
|---|---|---|---|---|
| > 500 pts | 120 pts | 96 pts | 73 pts | 94 pts |
| 300–500 pts | 81 pts | 61 pts | 53 pts | 59 pts |
| < 300 pts | 53 pts | 40 pts | 30 pts | 30 pts |

- **Highest intraday volatility** is consistently at the US open (14:30 UK = 09:30 EST).
- **Lowest intraday volatility** is mid-European afternoon before the US open (12:00–14:30 UK).

### Session-Specific Strategy Timing

- **School Run / ASRS:** DAX 09:00–11:00 Frankfurt time. Research validity of ASRS signals after 2 hours of active trading is not confirmed.
- **1BN / 1BP:** FTSE/Nasdaq 08:00 UK time (first 5-minute bar). No fixed time cutoff for the signal to trigger; Hougaard has been filled "hours into the session" on 1BP.
- **Rule of 4 / FOMC:** Begins at the FOMC announcement time (typically 19:00 UK). Signals come 40 minutes later (4th 10-minute bar after announcement).
- **US Open scalping:** Best scalping window is 14:30–15:30 UK (09:30–10:30 EST). Adjust stop/target approach based on current day's volatility regime (check daily ATR that morning).

### Pre-Market Preparation Rules

1. Check Asian market performance (overnight) as a bias indicator.
   - If US closed lower but Asia is positive → look for long opportunities at European open.
   - If Australian mining stocks rallied overnight → FTSE (commodity-heavy) likely to open stronger.
2. Check Forex Factory for red-flagged news events for the day. Only red-flag items require attention.
3. Apply the 3-day break method to daily and weekly charts → establish directional bias.
4. Identify prior day's high, low, and close. Market "has memory" of these levels.
5. Note whether today's open is a **gap up** or **gap down** from yesterday's close.
   - 48% of gaps in Dow/FTSE fill on the same day.
   - A gap down that is not immediately filling after the first hour suggests continued weakness.

---

## Instrument Notes

### DAX Index

- Opens 09:00 Frankfurt local time (08:00 UK); primary focus is 09:00–11:00.
- Most momentum-oriented of the European indices; large moves that sustain.
- School Run and ASRS strategies are designed specifically for DAX.
- 1BP does not statistically work in DAX.
- ASRS default stop: 2-point buffer each side of 4th-bar range; stop is defined by the opposite bracket.
- SRS stop guidance: 40 points when DAX near 12,000; scale with index level.
- Highly correlated with FTSE on many days; when divergence occurs, trade each separately on its own terms.
- DAX is also correlated with Dow direction (US sentiment), but do not trade one based solely on the other.

### FTSE 100 Index

- Opens 08:00 UK local time.
- Index is commodity-heavy (mining stocks, oil); watch overnight Australian and Asian mining performance for gap direction clues.
- Different character from DAX: "back and fills," overlapping bars, rarely sustains momentum for scaling in.
- 1BN and 1BP strategies work well. SRS (School Run) has not produced comparable results in FTSE.
- Preferred stop management: aggressive trailing (1 point per 1 point of profit).
- Move stop to breakeven at 6–10 FTSE points profit.
- When scaling in (adding), FTSE is less reliable than DAX; Hougaard reduced his FTSE scaling in later research.
- 1BN long signal is the stronger of the two 1BN signals; short above 1BN high is less reliable.

### Nasdaq

- Significantly higher volatility than FTSE or DAX.
- 1BP in Nasdaq: strong August 2024 result (8 of 10 triggered days produced good or great trades).
- Scalp parameters in Nasdaq are wider due to volatility (100–400-point swing targets on 1BP moves).
- When profitable, "make money fast" due to index volatility.

### Dow Jones

- Rule of 4 designed around Dow. SP500 assumed equivalent. Nasdaq not formally tested but likely applicable.
- 1BP in Dow: mixed results (August 2024: 4 outstanding, 4 breakeven, 3 losses).
- Intraday volatility benchmarks (True Range) used for DAX and Dow scalp parameter selection.

### FOMC and News Events

- US economic news: 13:30 UK (08:30 EST) for major releases; 15:00 UK (10:00 EST) for secondary releases.
- FOMC statements: 19:00 UK (14:00 EST).
- Rule of 4 applicable to any large scheduled event; first developed for FOMC announcements.
- Avoid entering new positions immediately at the news release; wait for bar 4 of the post-news price action.

---

## Stop Loss Framework — Sizing Rules

### Position Sizing Formula

```
Account size × risk percentage = monetary risk per trade (£/€/$)
Entry price − stop price = points at risk
Stake size = monetary risk / points at risk
```

Example: £1,000 account, 5% risk, buy DAX at 7920, stop at 7910 (10 pts) → stake = £50 / 10 = £5/pt.

Hougaard's starting guideline: risk 1–2% of account per trade. This is for early-stage traders; experienced traders adjust.

### Scaling In (Adding to Winners)

- Add at a price where the stop on the original position can be moved to breakeven.
- The stop on the added position = the stop on the original position (breakeven for original = stop for add).
- This means the total monetary risk does not increase beyond the initial risk.
- Hougaard's cited scaling example (4-hour FX framework): add at +75 points (double the position); stop for the add = original entry; combined target = 150+ points (combined profit vs. 75-point risk = 3:1 effective R:R).
- "The longer the time frame, the more important adding to a position becomes."
- **Command 3 (Best Loser Wins rule):** "I add when you subtract and subtract when you add." — The 5% behaviour is to add to winning positions as the market moves further in your favour, not to take early partial profits. Conversely, cut losing positions rather than adding to them.
- From STEP UP document: "I added to my winner, but without moving the stop loss on the first position. That was a mistake." → When adding, always move stop on the first position at the same time.
- If a position goes against you and returns to entry: treat as a signal to add (not close). Hougaard: "Whenever I have been in a position and losing for a while, and it comes back to my entry, I see it as a sign to add."

### Scaling Out (Taking Partial Profits)

- Conceptual framework from Mark Douglas (Trading in the Zone):
  - Exit 1/3 of position at +20 points (FX, 4-hour context).
  - Exit 2nd 1/3 at next significant swing high/low; move stop to breakeven.
  - Let remaining 1/3 run to target.
- Hougaard personally does not scale out; prefers full-position management. He views partial exits as a "psychological limitation" rather than a systematic strategy.

---

## Glossary

| Term | Definition |
|---|---|
| **Signal bar** | The specific candle/bar identified by a strategy rule, used to set up bracket orders. In SRS: the 2nd 15-min DAX bar. In ASRS: the 4th 5-min DAX bar. In 1BN/1BP: the first 5-min bar at the exchange open. |
| **School Run** | SRS — the strategy using the 2nd 15-minute DAX bar as a signal bar, named because it occurs during morning school-run traffic time. |
| **ASRS** | Advanced School Run Strategy — the strategy using the 4th 5-minute DAX bar as a signal bar. |
| **1BN** | First Bar Negative — the first 5-minute bar at the FTSE/Nasdaq open is a bearish bar (close < open). |
| **1BP** | First Bar Positive — the first 5-minute bar at the FTSE/Nasdaq open is a bullish bar (close > open). |
| **Rule of 4** | FOMC strategy using the 4th 10-minute bar after a major news announcement as the signal bar. |
| **Fishing Technique** | Hougaard's name for the Fake-Out / Fake-Down pattern: a false breakout above a double top high or below a double bottom low that then reverses, triggering an entry in the opposite direction. |
| **Extended Bar** | A bar with little or no tail (shadow), visibly larger in body than preceding bars. Synonymous with Marubozu, Road Map Bar, Aggressive Bar. Indicates strong conviction in the direction of the bar. |
| **Road Map Bar** | Alternative name for Extended Bar. Used when the bar is so large it creates a "map" of the range to come. Implies the bar's range defines likely support/resistance on a retracement. |
| **Noise** | The back-and-forth price oscillation that does not change the underlying trend. Defined practically as the range within which price oscillates without making new highs or lows in the trend direction. Stop losses should be placed outside the noise. |
| **Over-balance** | When a correction in a trend exceeds the size of prior corrections. Signals a potential trend change. Derived from geometry/W.D. Gann / Bryce Gilmore terminology. Used as a stop-loss methodology: if the current pullback "over-balances" prior pullbacks, exit the position. |
| **Harmonic Retracement** | A retracement from a swing that equals a prior retracement in the same trend. Used for entry orders ("if the market retraces by the same amount as last time, I enter here"). Also called: measured move, 1-to-1, equal retracement, AB=CD. |
| **AB=CD** | Two-wave correction pattern where the B-C leg equals the A-B leg in price distance. Also called harmonic retracement. Signals a likely continuation entry at point D. |
| **3-bar break** | Mechanical trend indicator: if price breaks above/below the highest high/lowest low of the prior 3 bars, the trend is declared to have changed. Used on weekly and daily charts for bias. Attributed to Bryce Gilmore. |
| **4-bar fractal** | Entry trigger: bar 1 close > high of bar 2 AND high of bar 4 (for buy). Attributed to Dr David Paul. A mechanical measure of a turning point. |
| **Fake-Out** | Price breaks above a prior high, runs stops, then reverses back into the range. Entry on the reversal back through the breakout level. Part of the Fishing Technique. |
| **Fake-Down** | Price breaks below a prior low, runs stops, then reverses back into the range. Entry on the reversal back through the breakdown level. |
| **Buy On Close (BOC)** | Scalping entry method: enter a long position at the close of a strong bullish bar with no upper tail and expanding range. |
| **Sell On Close (SOC)** | Scalping entry method: enter a short position at the close of a strong bearish bar with no lower tail and expanding range. |
| **Situational Analysis** | Hougaard's term for his category of strategies. Identifies a specific market situation (e.g., first bar negative), waits for it to occur, then trades a pre-defined response. Designed to eliminate real-time decision-making. |
| **Spinning Top** | Candle with a small real body and long shadows on both sides. Signals indecision; possible pause or end of trend. |
| **Doji** | Candle where open and close are at the same price (or very close). Long-Legged Doji / Doji Star: long shadows on both sides. Gravestone Doji: long upper shadow, close at low. Dragonfly Doji: long lower shadow, close at high. |
| **Pin Bar** | Hougaard uses this informally; equivalent to a Dragonfly or Gravestone Doji (one-sided wick pattern indicating rejection). He notes Linda Raschke also developed this concept. |
| **Engulfing bar** | The body of the current bar completely covers the body of the previous bar. Bullish or bearish depending on direction. One of the "Essential 8" candle patterns. |
| **Marubozu** | A bar with no upper or lower shadow; opens on the high (bearish) or opens on the low (bullish). Very common after news releases. |
| **Divergence** | Oscillator (e.g. stochastics) makes a higher high/lower low while price makes the opposite. Warning sign. Traded in combination with 4-bar fractal for timing. |
| **ATR** | Average True Range; 14-period standard. Used for volatility-adaptive stop sizing. |
| **Pay-in phase** | Losing period in a strategy's equity curve; expected and accepted as part of non-linear trading income. |
| **Pay-out phase** | Winning period. Hougaard explicitly plans for alternating pay-in and pay-out phases rather than expecting linear profits. |
| **Situation / Context** | The broader market state that determines whether a technical signal is high or low quality. E.g., a bullish engulfing bar right at major resistance = lower quality because sellers are positioned there. |
