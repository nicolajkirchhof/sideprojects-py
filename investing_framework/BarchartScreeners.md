# Barchart Screener Configuration

Concrete screener setups for the swing trading playbook. These feed into the Trade Analyst
pipeline via Gmail (label: `TradeAnalyst`) — Barchart sends daily summary emails with CSV
attachments that the pipeline parses automatically.

**References:** `BreakoutStrategy.md` §02 (theory) · `TradingPlaybook.md` (rules)

---

## Global Base Filters

Applied to ALL screeners. Match the playbook minimums.

| Filter | Value | Playbook Rule |
|--------|-------|---------------|
| Last Price | > $3 | Avoid penny stocks, options liquidity |
| 20D Avg Volume | > 1,000,000 | Options must be tradeable |
| Market Cap | > $300M | Reduce noise from micro-caps |

> **Note:** Current base uses Price > $1 and Volume > 500K. Tighten to match the playbook
> when scanner results are too noisy.

---

## Screener Definitions

### 1. 52-Week High

**Purpose:** Find stocks near ATH with no overhead supply — Stage 2 uptrend candidates (Box 1).

| Filter | Value |
|--------|-------|
| 52W %/High | Within 5% |
| 20D RelVol | > 1.0 |

**Maps to:** Box 1 (Trend Template), Type B (VCP near highs)

---

### 2. 5-Day Momentum Leaders *(recommended — add this)*

**Purpose:** Century momentum signal — this week's leaders. Catches ignition events and EPs.

| Filter | Value |
|--------|-------|
| 5D %Chg | > 5% |
| 20D RelVol | > 1.0 |

**Maps to:** PM-01 (Breakout Momentum), PM-02 (PEAD), Type A (Episodic Pivot)

---

### 3. 1-Month Sustained Strength *(recommended — add this)*

**Purpose:** Sustained Stage 2 stocks — not one-day wonders. Best for Type B/C setups.

| Filter | Value |
|--------|-------|
| 1M %Chg | > 10% |
| Price | > 50D SMA |
| 50D SMA | > 200D SMA |

**Maps to:** Box 1+2 (Trend Template + RS), Type B (VCP), Type C (SMA Reclaim)

---

### 4. 100% Volume / 20D Average

**Purpose:** Ignition event detection — something significant is happening today.

| Filter | Value |
|--------|-------|
| 20D RelVol | > 1.75 |

**Maps to:** All setup types (volume confirmation), Type A (EP requires 5-10× volume)

---

### 5. Trend Seeker Buy/Sell

**Purpose:** Proprietary Barchart signal — supplementary confirmation, not primary.

| Filter | Value |
|--------|-------|
| Signal | New Buy or Sell |
| Strength | Strong or Maximum |
| Direction | Strengthening or Strongest |

**Maps to:** Supplementary — confirms momentum direction, not a standalone entry signal.

---

### 6. High Put Ratio

**Purpose:** Unusual put activity on strong stocks = potential short squeeze fuel (PM-04).

| Filter | Value |
|--------|-------|
| 5D Avg Put/Call Volume Ratio | > 1.0 |
| Put/Call Volume Ratio | > 1.0 |
| 1M Avg Options Volume | > 5,000 |

**Maps to:** Box 4 (Catalyst — squeeze potential), PM-04 (Informed Flow)

---

### 7. High Call Ratio

**Purpose:** Call-dominant flow = informed upside anticipation (PM-04).

| Filter | Value |
|--------|-------|
| 5D Avg Put/Call Volume Ratio | < 0.25 |
| Put/Call Volume Ratio | < 0.25 |
| 1M Avg Options Volume | > 5,000 |

**Maps to:** Box 4 (Catalyst — informed call buying), PM-04 (OTM Informed Flow)

> **Consider widening to P/C < 0.5** for a broader net. Keep P/C < 0.25 as an aggressive
> sub-scan if desired.

---

### 8. Intraday RVOL Spike *(manual — run during execution windows only)*

**Purpose:** Catch ignition events in real-time — volume spikes happening NOW. Finds Type A
Episodic Pivots and surprise movers not on last night's watchlist.

| Filter | Value |
|--------|-------|
| 20D RelVol | > 2.0 |
| %Change | > 2% |
| Base filters | Price > $3, Vol > 1M, MktCap > $300M |

**When to run (manually on Barchart website):**
- **9:45** — after first 15min candle closes, scan for morning ORB candidates
- **15:00** — before last-hour window, scan for late-day accumulation

**NOT an email scanner.** Run manually during execution windows only. End-of-day scanners
capture settled data; this captures live ignition.

**Quick evaluation (< 30 seconds per stock):**
1. Already on watchlist? → skip (already evaluated)
2. Earnings today/tomorrow? → skip
3. Above 50D SMA? → if no, skip (no longs in downtrend)
4. Clear catalyst visible? (gap, news, sector move) → if no, skip
5. ORB candle clean? (not choppy inside bar) → if yes, set buy stop above 15/30min candle

**Maps to:** Type A (EP on gap + volume), Type C (reclaim with volume confirmation)

**What to avoid:** Stocks already up 10%+ and extended from all SMAs — that's FOMO, not a setup.

---

## Options Screener: Unusual Options Activity (UOA)

Separate from the stock screeners above. Returns individual option contracts, not
underlyings. The pipeline aggregates by underlying symbol and feeds into Box 4 scoring.

**Purpose:** Detect informed directional bets — someone opened a large new position today
on a specific strike/expiry. Predicts stock returns over 1–4 weeks (Pan & Poteshman 2006,
Ge et al. 2016). Confirmation signal for PM-04 (OTM Informed Flow).

### Filters

| Filter | Value | Reason |
|--------|-------|--------|
| Probability of Profit | 10%–75% | Exclude deep ITM (hedging) and far OTM (lottery) |
| Next Earnings Date | Exclude within 7 days | Avoid earnings-driven vol, want directional signal |
| Volume/OI Ratio | > 2.0 | Today's volume exceeds existing positions — new bets |
| 1M Avg Options Volume | > 5,000 | Liquid options chain |
| 1M Avg Open Interest | > 5,000 | Established market, not thin |
| Delta | -0.4 to 0.4 | OTM focus — where informed flow concentrates |
| Days to Expiration | < 35 | Near-term = conviction. Longer DTE = hedging noise |
| Expirations | Monthly + Weekly | Include all liquid expirations |
| Today's Option Volume | > 1,000 | Meaningful size |
| Option Open Interest | > 500 | Existing liquidity |

### Recommended additions

| Filter | Value | Reason |
|--------|-------|--------|
| Underlying Price | > $3 | Match stock screener base filter |
| Underlying Avg Volume | > 1M | Match stock screener liquidity filter |

### Column View (Options Screener)

| Column | Use |
|--------|-----|
| Symbol | Underlying ticker + contract info |
| Price~ | Underlying price |
| 1 Std~ | 1 standard deviation move — context for strike selection |
| IV Pctl | IV Percentile — options structure decision |
| Imp Vol | Implied volatility of this contract |
| 1D IV Chg | IV change today — spiking IV = demand |
| 5D IV Chg | IV trend over week |
| Type | Call or Put — determines directional bias |
| Strike | Strike price |
| IV %Chg | IV percentage change on this contract |
| Exp Date | Expiration date |
| Delta | Delta value — distance from ATM |
| Moneyness | ITM/ATM/OTM classification |
| ITM Prob | Probability of being in the money at expiry |
| Vol/OI | Volume to open interest ratio — the core UOA signal |
| Bid / Ask | Spread — liquidity quality |
| Volume | Today's volume on this contract |
| Vol %Chg | Volume change vs average |
| Open Int | Existing open interest |
| OI %Chg | OI change — positive = new positions being opened |
| Theta | Time decay — context for holding period |
| Exp B4 Earnings | Whether this contract expires before next earnings |
| Links | Barchart detail links |

### Key columns for the pipeline

The pipeline aggregates option contracts to underlying-level signals:

| Aggregated Metric | How | Box 4 Interpretation |
|---|---|---|
| UOA Call Count | Count of unusual call contracts per underlying | Bullish informed flow |
| UOA Put Count | Count of unusual put contracts per underlying | Bearish or squeeze signal |
| Max Vol/OI | Highest vol/OI ratio among contracts | Conviction strength |
| Avg Delta (calls) | Average delta of unusual call contracts | How OTM = how leveraged the bet |
| Net Call/Put | UOA calls minus UOA puts per underlying | Directional bias |
| IV Percentile | From underlying | Options structure selection |

### How to evaluate

**Strong bullish signal (upgrades Box 4):**
- Multiple unusual CALL contracts on same underlying
- Stock already passes Boxes 1–3 (trend, RS, base)
- Vol/OI > 5 on any single contract
- Delta 0.2–0.4 (slightly OTM, not lottery tickets)
- OI %Chg positive (new positions, not closing)

**Ignore:**
- UOA on stocks with earnings within 7 days (already filtered, but double-check)
- Single contract with Vol/OI barely above 2 — could be one retail trader
- UOA puts on strong RS stocks — likely hedging, not bearish signal
- Wide bid/ask spread — illiquid contract, noisy signal

**Maps to:** Box 4 (Catalyst — informed flow), PM-04 (OTM Informed Flow)

---

## Column View (Stock Screeners)

Single column layout shared across all screeners. Provides everything for the 5-box
checklist at a glance.

| Column | 5-Box Use |
|--------|-----------|
| Symbol | Identity |
| Latest | Current price |
| %Change | Today's move |
| 5D %Chg | Momentum (Box 1) |
| 1M %Chg | Sustained strength (Box 1+2) |
| 3M %Chg | Intermediate trend context |
| 6M %Chg | Stage 2 confirmation |
| 52W %Chg | Century momentum / 12M return (Box 1) |
| 52W %/High | Overhead supply (Box 1) |
| Volume | Today's volume |
| 20D RelVol | RVOL — ignition detection (Box 3) |
| 20D ATRP | ATR as % — stop distance check (Box 5) |
| % 50D MA | Distance from 50D SMA — trend check (Box 1) |
| BB% | Bollinger position — squeeze detection (Box 3) |
| 5D P/C Vol | Put/Call ratio — options flow (Box 4) |
| IV Pctl | IV Percentile — options structure selection |
| Short Vol% | Short volume context |
| Short Int %Chg | Short interest change — squeeze fuel |
| Days2Cover | Days to cover — squeeze potential (Box 4) |
| Market Cap, $K | Size filter |
| Latest Earnings | Binary event proximity (Box 4) |
| Sector | Sector RS check (Box 2) |
| %Chg(Pre) | Pre-market move (gap detection for Type A) |
| 20Y %/High | Multi-decade context |
| BB% | Bollinger Band percentile |
| 5D IV Chg | IV trend for options timing |

---

## Pipeline Integration

The screeners are configured to send daily email summaries to Gmail.

**Gmail filter:**
- From: `noreply@barchart.com`
- Action: Apply label `TradeAnalyst`, skip inbox

**Pipeline flow:**
1. `finance.apps.analyst` fetches emails from `TradeAnalyst` label
2. CSV attachments auto-downloaded and parsed
3. Column mapping configured in `finance/apps/analyst/config.yaml`
4. All fields feed into 5-box scoring + Claude analysis

**Config reference** (`config.yaml` column mapping):
```yaml
column_mapping:
  Symbol: symbol
  Latest: price
  "%Change": change_pct
  "5D %Chg": change_5d_pct
  "1M %Chg": change_1m_pct
  "52W %Chg": change_52w_pct
  "52W %/High": high_52w_distance_pct
  "20D RelVol": rvol_20d
  "20D ATRP": atr_pct_20d
  "% 50D MA": pct_from_50d_sma
  "BB%": bb_pct
  "5D P/C Vol": put_call_vol_5d
  "IV Pctl": iv_percentile
  Sector: sector
  "Latest Earnings": latest_earnings
  ...
```

---

## Daily Schedule

| Time | Action | Scanners |
|------|--------|----------|
| **After 16:00** | Run pipeline | All 7 email scanners (automated) |
| **Evening / pre-market** | Review Daily Prep | None — review `/daily-prep`, set ORB alerts, remove earnings |
| **9:45** | Manual scan | Intraday RVOL Spike (#8) — morning ORB candidates |
| **10:15–15:00** | Dead zone | No scanning, no entries |
| **15:00** | Manual scan | Intraday RVOL Spike (#8) — last-hour candidates |
| **15:15–15:45** | Execute | Last-hour ORB entries from watchlist + RVOL scan |

**Email scanners** (1–7) are configured for end-of-day delivery → pipeline runs once daily.
**Intraday RVOL** (8) is manual on Barchart website — run only during execution windows.

---

## Pipeline Integration

The email scanners are configured to send daily summaries to Gmail.

**Gmail filter:**
- From: `noreply@barchart.com`
- Action: Apply label `TradeAnalyst`, skip inbox

**Pipeline run:**
1. `uv run python -m finance.apps analyst`
2. Fetches emails from `TradeAnalyst` label → downloads CSV attachments
3. Parses all CSVs → enriches with IBKR data → 5-box scores → Claude analysis
4. Pushes results to Tradelog → view at `/daily-prep`

**Manual fallback** (when pipeline is unavailable):
1. Run each screener on Barchart
2. Apply 5-box checklist using the column view
3. Add passing stocks to watchlist — max 20 names, prioritize top 5
