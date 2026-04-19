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

## Column View (All Screeners)

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

## Scanning Process (Daily)

1. **Barchart sends** screener emails automatically (end of day)
2. **Gmail filter** labels them `TradeAnalyst`
3. **Run pipeline:** `uv run python -m finance.apps analyst`
4. Pipeline fetches emails → parses CSVs → enriches with IBKR data → 5-box scores → Claude analysis
5. Results pushed to Tradelog → view at `/daily-prep`

Manual scanning (when pipeline is unavailable):
1. Run each screener on Barchart
2. Apply 5-box checklist mentally using the column view
3. Add passing stocks to watchlist — max 20 names, prioritize top 5
