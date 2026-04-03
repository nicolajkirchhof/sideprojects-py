# Trading rulebook

## Goal

- 20% Grow YoY
- Fully automated 0DTE options and futures trading
- Discretionary 7 - 120DTE options trading

---

## Investment strategy

- 80% In an globally diversified portfolio of ETFs
- 20% Day Trading & Options Play
- XX Automatic trading systems as soon as ready

---

## Long Term investments AKA the tanker

Long term investment portfolio that is filled using a weekly/monthly buy-in

### Portfolio Contents

Based on the five factor model and the Warrens Alpha paper

---

## General Safeguards to be honored at all time

### Portfolio Management

- Do not user more than 50% BP in total for options to make sure you have enough margin for volatility spikes
- Do not use more than 2% BP per Trade (max loss)
- Try to stay delta neutral
- Have a theta profile of approx .4% (TBD)
- Do not compound risk by assigning too much BP (> 30%) to one strategy

### Trading

- If you lose more than 150€ on a trading position/250€ on an options position in a week it resets all progress for that instrument
- Max 10 trades per day in paper trading account to avoid overtrading
- Additional times to "play around" are fine but need to be logged beforehand in tradevis
- Max 1 negative Trade per Ticker (one strike)
- A weekly positive P/L on Ticker/Trade-Type gives you the right to take that kind of Ticker/Trade-Type once the following week in Live Trading
- This accumulates per week so that you can switch to live trading after 5 weeks. Accumulated P/L of Paper/Live counts
- A negative P/L week removes one live trading permission
- A negative 3 days streak resets all live trading permission

### Options

- Determine new strategies in paper Trading and simulation
- Each strategy needs to be proven (pos P/L) for 3 Month in order to be tested in live trading to gain confidence and make sure you don't do the FOMO mistakes of the DAX again when you are not fully confident, that the strategy works or your simulations are correct
- Each strategy needs to be defined below
- Each trade has to be monitored at least once a day in tradesvis

---

## Options trading

- Sell mostly premium on commodities and indices
- Try to stay directional neutral
- Use mostly options on futures because they trade almost 24/5 and are more likely to respect your SL
- Rather close and re-open then roll a position

### General rules

- Take loss @200%
- SL should be applied to individual Shorts, not spreads to make sure the SL is executed
- 0DTE
  - Strategies should be automated
- 7DTE - 21DTE
  - Close trades @80% or let them expire if save
- 45DTE - 60DTE
  - Take profit @50% when selling long term premium
  - Try to close before 14 DTE to avoid Gamma risk

### XXY Trade

- Sell X approx 20-30Δ PDS e.g. in ES Sell 5050 PUT and Buy 5100 PUT
- Expiration based on Chart and Risk portfolio. Needs to be determined
- Sell Y approx 5-10Δ PUT at the same expiration or half of that expiration
- Half gives option to use the PDS as a free hedge for other trades
- Manage both independently
- X, Y is to be determined. There are different pros and cons. More Y increases the risk on volatility spikes & large losses
- Potential 111 or 221 would be save options to test out
- TK trades 112 which needs higher margins and has two times the downside risk as 111

### Sell premium (Strangle / Naked Put)

- Sell approx 5Δ PUT & CALL
- Use SL instead of IC if possible but check your Margin impact to see if feasible
- Independent SL for long and short side
- Check implication of IV vs HV on decision

### PMCC

- Buy 80Δ ITM Call >120DTE, SELL before 45DTE to avoid large loss on extrinsic value
- Sell ATM Calls 40-20Δ every week to collect premium
- Close before time value is gone to make sure to not get assigned

---

## Daily active trading - ON PAUSE

- Daily ( 9:00 - 10:00, 21:00 - 22:00)
- Based on
  - Matt Diamonds opening range breaks
  - Tom Hougaards index trading

### General Advices

- Check key levels
- Pre-Market High/Low, Previous Close
- Check Market/Sector Momentum
- Keep in one Timeframe with key position!
- If you are able to size (stocks) you can try taking parts of the position <25% to a higher timeframe

---

## Trading Index Futures

Us well known indices like ES, NQ, DAX, FTSE. Trade on researched strategies and pure price action

#### Preparation

Try to find the market sentiment before trading and log it into Tradevis

#### Current state

Trial period: If you don't explicitly specify play around time beforehand, every trade counts to your PnL

### Scalping

#### Setup

- Contract trades in overlapping candles
- Overlaps are at least 25-50%
- PLACE YOUR ENTRY WHERE THE OTHERS HAVE THEIR STOP LOSSES!!!

#### Entry / Target / Exit

- In direction of trend
  - Wait for next candle that has low < previous low in up-trend and high > last high in down trend
  - Get in right above the last candle
  - Target is the high/low of the previous candle
- In counter direction of trend
  - Get in right at the high/close of a candle in counter trend direction
  - Target is 0.25-0.5 height of previous candle.
  - Only makes sense on large movements where a strong pull back is expected
- Get out FAST

#### Scale

Only on accidental range break. 'Ride the trend'

### Swing-Trading

#### Setup

Entry should be based on major S/R levels like

- VWAP,
- 2-day-AWVAP,
- Multi-Timeframe-Levels,
- Prior day close

If the contract trades in a defined trading range and you have a sentiment, it makes sense to get into the trade at the opposite side of the range to place a tight stop loss.

In case there is no breakout, this could still be a nice scalp on the other side of the range

Ask yourself the following questions before getting in the Trade

- What is the expected move?
- Where do I need to place my Stop-Loss based on the trading range?
- What is the probability based on historical data?

#### Stop-Loss

- Max Stop-Loss 100€ per Trade (5%)
- Max Stop-Loss per Day 100€
- Move stop loss bar by bar with an offset of approx. 10% ATR
- Mind the S/R levels and rather place it on a S/R level then on the low of the last bar

#### Entry

- On possible S&R levels
- Use the 5min timeframe
- Entry on bull/bear flag
- At best three bounces between a range
- Bracket with sell/buy stop order
- Max time to profit should be 1min

#### Scale

On support & resistance levels. If trade shows continuation on next SR-Level, double contract size. For Futures these are usually 25,50,100 levels. A good scale in for the DAX would be 1-3 contracts after crossing the next 25 level to get a profit of 50-100 for the next 50 ticks.

Take off scale after a P/L of 2:1 was reached, keep remaining position for additional gains

#### Exit

- SL on last SR-Level / Bar with offset
- After profit target (2x STOP distance) has been surpassed set trailing stop limit for 50-90% of position size with tight
- Let the remaining size run bar-by-bar

#### Strategies

- Rule of four
- School run
- Expresso

#### Specific trading rules

##### DAX

- Sentiment trade based on Intraday VWAP
  - Entry/Exit should be based +- 10 Ticks around VWAP, depending on the rate of change
  - Initial Stop loss should be approx.  50 Ticks away on a strong S/R level but can be tightened if DAX strongly changes direction
- Range break
  - Bracket range based on position relative to VWAP
  - Wait until low of candle - 2-5 TICKS is >= break even
  - Set trailing stop limit based on ATR or move stop loss bar by bar (default trail 10-15 on 2 min Chart)
- Scale in
  - After +10 Ticks with 10 pts SL
  - When a clear area of resistance was overcome

### Options Trading

| Percentage | Market | Reference |
|---|---|---|
| 40% | US Total Market |  |
| 10% | US Small cap value |  |
| 10% | Euro Stoxx 600 |  |
| 10% | FTSE 100 |  |
| 10% | MSCI Emerging Markets Small Cap |  |
| 10% | Japan |  |
