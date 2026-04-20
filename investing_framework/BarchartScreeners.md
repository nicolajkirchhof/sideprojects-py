# Barchart Screener Configuration

Concrete screener setups for the swing trading playbook. These feed into the Trade Analyst
pipeline via Gmail (label: `TradeAnalyst`) — Barchart sends daily summary emails with CSV
attachments that the pipeline parses automatically.

**References:** `BreakoutStrategy.md` §02 (theory) · `TradingPlaybook.md` (rules)

---

## Global Base Filters

Applied to ALL screeners unless noted otherwise.

| Filter | Value | Reasoning |
|--------|-------|-----------|
| Last Price | > $5 | ORB execution quality — $3–5 stocks have wide % spreads and noisy ticks. Kullamägi floor is $5; options on sub-$5 names have 1.5–3% bid/ask spreads |
| 20D Avg Volume | > 1,000,000 | ORB entries demand real intraday liquidity in the first 15/30 min. 500K avg volume stocks often have dead first-hour action |
| Market Cap | > $200M | Keep small-cap PEAD plays (strongest drift in research) while filtering micro-cap noise. Volume filter handles quality — any stock at 1M vol + $200M cap has institutional interest |
| Next Earnings | Exclude within 5 days | Playbook Box 4 rule — no binary events on entries. Filter upfront to reduce scanner noise instead of scoring then failing |

**Long-only scanners** (1–5, 10, 11) add:

| Filter | Value | Reasoning |
|--------|-------|-----------|
| % 50D MA | > 0% | Eliminate Stage 4 downtrends. Box 1 requires Price > 50D SMA. Reduces noise ~40% |
| Slope of 50D SMA | Rising | Box 1 requires 50D SMA rising — more precise than just % from SMA |
| 200D MA Direction | Up | Completes Trend Template — 200D must be sloping upward for Stage 2 confirmation. Without this, stocks in late Stage 3 tops pass the 50D filter |
| 20D ADR% | > 3% | Box 3 requires sufficient daily range for 2R+ within the 5–50 day timeframe. Stocks with ADR < 3% can't deliver ORB entries with tight stops |
| Weighted Alpha | > 0 | 12-month momentum weighted to recent activity — better century momentum proxy than raw 52W %Chg |

**Do NOT add trend filters to:** High Put Ratio (#6), High Call Ratio (#7) — these
intentionally scan across all trend states. UOA scanner has its own filters.

---

## Screener Definitions

### 1. 52-Week High

**Purpose:** Stage 2 uptrend candidates near ATH — no overhead supply (Box 1).
**View:** Standard

| Filter | Value | Reasoning |
|--------|-------|-----------|
| 52W %/High | Within 5% | Near all-time high = clean breakout territory |
| 20D RelVol | > 1.0 | Volume interest confirms the move |
| TTM Squeeze | On | Bollinger/Keltner squeeze = volatility contraction before breakout (Box 3) |

**Maps to:** Box 1 (Trend Template), Box 3 (Base Quality), Type B (VCP near highs)

---

### 2. 5-Day Momentum Leaders

**Purpose:** This week's leaders — catches ignition events, EPs, early PEAD entries.
**View:** Standard

| Filter | Value | Reasoning |
|--------|-------|-----------|
| 5D %Chg | > 5% | Significant weekly move |
| 20D RelVol | > 1.0 | Volume confirms the move is real |
| Performance vs Market 5D | > 0% | Outperforming SPY this week (Box 2 pre-filter) |

**Maps to:** PM-01 (Breakout Momentum), PM-02 (PEAD day 1–5), Type A (EP)

---

### 3. 1-Month Sustained Strength

**Purpose:** Sustained Stage 2 stocks — not one-day wonders. Type B/C setup candidates.
**View:** Standard

| Filter | Value | Reasoning |
|--------|-------|-----------|
| 1M %Chg | > 10% | Sustained strength over a full month |
| Performance vs Market 1M | > 0% | RS vs SPY over 1 month (Box 2) |
| TTM Squeeze | On | Volatility contraction after strong run = base forming (Box 3) |

**Maps to:** Box 1+2 (Trend Template + RS), Box 3 (Base Quality), Type B (VCP), Type C (SMA Reclaim)

---

### 4. Volume Spike (RVOL > 1.75)

**Purpose:** Ignition event detection — something significant is happening NOW.
**View:** Standard

| Filter | Value | Reasoning |
|--------|-------|-----------|
| 20D RelVol | > 1.75 | Strong volume spike vs average |

**Maps to:** All setup types (volume confirmation), Type A (EP requires 5–10× volume)

---

### 5. Trend Seeker Buy/Sell

**Purpose:** Proprietary Barchart signal — supplementary confirmation, not primary.
**View:** Standard

| Filter | Value | Reasoning |
|--------|-------|-----------|
| Signal | New Buy or Sell | Fresh signal only |
| Strength | Strong or Maximum | Filter weak signals |
| Direction | Strengthening or Strongest | Momentum accelerating |

**Maps to:** Supplementary — confirms momentum direction, not a standalone entry signal.

---

### 6. High Put Ratio

**Purpose:** Unusual put activity = potential short squeeze fuel (PM-04, PM-11).
**View:** Options/Flow

| Filter | Value | Reasoning |
|--------|-------|-----------|
| 5D Avg Put/Call Volume Ratio | > 1.0 | Sustained put-heavy flow |
| Put/Call Volume Ratio | > 1.0 | Today also put-heavy |
| 1M Avg Options Volume | > 5,000 | Liquid options chain |

**No trend filters** — intentionally includes weak stocks (squeeze candidates where heavy
put buying can fuel a reversal).

**Maps to:** Box 4 (Catalyst — squeeze potential), PM-04 (Informed Flow), PM-11 (Short Squeeze)

---

### 7. High Call Ratio

**Purpose:** Call-dominant flow = informed upside anticipation (PM-04).
**View:** Options/Flow

| Filter | Value | Reasoning |
|--------|-------|-----------|
| 5D Avg Put/Call Volume Ratio | < 0.5 | Sustained call-heavy flow |
| Put/Call Volume Ratio | < 0.5 | Today also call-heavy |
| 1M Avg Options Volume | > 5,000 | Liquid options chain |

**No trend filters** — informed call buying sometimes appears on names that haven't
broken above the 50D yet (anticipating the move).

**Maps to:** Box 4 (Catalyst — informed call buying), PM-04 (OTM Informed Flow)

---

### 9. PEAD Candidates

**Purpose:** Post-Earnings Announcement Drift — stocks that gapped on earnings within the
last 7 days. PM-02 is the highest-conviction mechanism in the playbook (40–60 day drift).
**View:** PEAD/EP

| Filter | Value | Reasoning |
|--------|-------|-----------|
| Latest Earnings | Within past 7 days | Earnings event just happened |
| 5D %Chg | > 10% | Playbook minimum gap for PM-02 |
| 20D RelVol | > 2.0 | Earnings day volume spike (5–10× is ideal) |
| Earnings Surprise% | > 5% | Larger surprise = longer drift (Bernard & Thomas 1989). Filters stocks that moved 10% on guidance/revenue but had a weak EPS beat |
| Weighted Alpha | > 0 | Stock was already in positive trend (Stage 2) |
| Performance vs Market 5D | > 0% | Outperforming SPY post-earnings (RS confirmation) |

**Maps to:** PM-02 (PEAD), Type A (Episodic Pivot on earnings)

**Evaluation priority:** Highest. PEAD has the strongest academic backing (Ball & Brown 1968,
Bernard & Thomas 1989). Enter on day 2–5 via ORB; 35–55 days of drift remaining.

---

### 10. TTM Squeeze Breakout

**Purpose:** Stocks emerging from a Bollinger/Keltner squeeze — volatility expansion imminent.
Directly maps to Box 3 (base quality) and VCP pattern detection.
**View:** Standard

| Filter | Value | Reasoning |
|--------|-------|-----------|
| TTM Squeeze | Fired (momentum positive) | Squeeze just ended, momentum is bullish |
| 20D RelVol | > 1.0 | Volume expanding on the squeeze release |
| 20D ATRP | < 7% | Stop distance within playbook 7% limit (Box 5) |

**Maps to:** Box 3 (Base Quality — BB squeeze), Box 5 (Risk — ATR confirms tight stop), Type B (VCP)

**Note:** TTM Squeeze is a Barchart built-in that combines Bollinger Bands and Keltner
Channels — exactly what Box 3 evaluates manually. This scanner automates that detection.

---

### 11. EP Gap Scanner

**Purpose:** Detect Episodic Pivots — the highest-priority setup in the playbook. Catches
gap-ups from earnings, news, or sector catalysts on the day they happen. Screener #9
catches PEAD after the fact (within 7 days); this catches the gap live for next-day ORB.
**View:** PEAD/EP

| Filter | Value | Reasoning |
|--------|-------|-----------|
| Gap Up Percent | > 8% | EP threshold (10% ideal per playbook; 8% catches near-misses that close strong) |
| 20D RelVol | > 3.0 | EP requires 5–10× volume; 3× is the practical minimum for Barchart filtering |
| Performance vs Market 5D | > 0% | Outperforming SPY — RS confirmation |

**Maps to:** PM-01 (Breakout Momentum), PM-02 (PEAD), Type A (Episodic Pivot)

**Evaluation:** Check the PEAD/EP view — Earnings Surprise% columns show whether this is an
earnings-driven gap (PM-02) or news/catalyst-driven (PM-01/PM-06). Earnings-driven gaps
with Surprise% > 10% are the highest-conviction entries.

---

### 12. Short Squeeze Watchlist

**Purpose:** Stocks with high short interest where a catalyst could trigger forced covering.
PM-11 research candidate — watchlist scanner, not direct entry signal.
**View:** Options/Flow

| Filter | Value | Reasoning |
|--------|-------|-----------|
| Short Float | > 15% | High short interest — squeeze fuel |
| Days to Cover | > 4 | Multi-day covering event if triggered |
| 20D RelVol | > 1.5 | Volume building — early sign of covering or catalyst |
| % 50D MA | > 0% | RS turning positive — shorts are losing |

**No earnings filter** — squeezes can happen around earnings events.

**Maps to:** PM-11 (Short Squeeze Setup), Box 4 (Catalyst — squeeze potential)

**Evaluation:** Cross-reference with High Put Ratio (#6). Stocks appearing on both are the
strongest squeeze candidates. Still requires full 5-box checklist before entry.

---

### 13. Negative PEAD Candidates *(short framework)*

**Purpose:** Stocks that gapped down on earnings within the last 7 days — the mirror of
scanner #9. Negative PEAD drift is as persistent on the downside as the upside (Bernard &
Thomas 1989). Feeds the short framework defined in `PortfolioAssessment.md`.
**View:** PEAD/EP

| Filter | Value | Reasoning |
|--------|-------|-----------|
| Latest Earnings | Within past 7 days | Earnings event just happened |
| 5D %Chg | < –5% | Significant downside move post-earnings |
| Earnings Surprise% | < –5% | Negative surprise — bottom quartile SUE |
| 20D RelVol | > 2.0 | Volume spike on earnings day |
| Short Float | < 20% | Squeeze filter — above 20% is squeeze risk, not signal |
| % 50D MA | < 0% | Below 50D SMA — confirms Stage 3/4 distribution |

**No long-only trend filters** — this scanner intentionally finds stocks in downtrends.

**Maps to:** Negative PEAD (short framework Layer 2A), Type D (Breakdown)

**Evaluation priority:** Highest during earnings season. Use the PEAD/EP view — Earnings
Surprise% columns show the beat history. Stocks with 2+ consecutive misses (columns 19–21
all negative) are the strongest short candidates (Layer 2B: consecutive miss).

---

### 14. RW Breakdown Candidates *(short framework)*

**Purpose:** Stocks with sustained relative weakness vs SPY breaking down through support.
The short-side mirror of screener #3 (sustained strength). Feeds Type D short entries.
**View:** Options/Flow

| Filter | Value | Reasoning |
|--------|-------|-----------|
| Perf vs Market 5D | < –3% | Underperforming SPY significantly this week |
| Perf vs Market 1M | < 0% | Sustained weakness vs SPY (RW confirmation) |
| % 50D MA | < 0% | Below 50D SMA — in downtrend |
| Slope of 50D SMA | Declining | SMA confirms distribution |
| Short Float | < 20% | Squeeze filter |
| 20D RelVol | > 1.0 | Volume interest on the breakdown |

**No earnings filter** — breakdowns can happen independent of earnings events.

**Maps to:** PM-05 RW Divergence (short side), Type D (Breakdown), short framework Layer 2C

**Evaluation:** Cross-reference with High Put Ratio (#6). Stocks showing RW + heavy put
buying are being actively distributed by institutions. Use Options/Flow view to check
short interest dynamics — rising SI confirms informed shorting (Layer 3).

---

### 8. Intraday RVOL Spike *(manual — run during execution windows only)*

**Purpose:** Catch ignition events in real-time — volume spikes happening NOW. Finds Type A
Episodic Pivots and surprise movers not on last night's watchlist.
**View:** Intraday

| Filter | Value |
|--------|-------|
| 20D RelVol | > 2.0 |
| %Change | > 2% |
| % 50D MA | > 0% |
| Base filters | Price > $5, Vol > 1M, MktCap > $200M |

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

## Column Views

Barchart limits custom views to 25 columns. Four views are defined — each screener
references which view to use. The UOA options screener has its own separate view.

### Standard View

Used by screeners: **1, 2, 3, 4, 5, 10** (the core trend/momentum scanners).

Covers all 5 boxes at a glance. Optimized for end-of-day email evaluation.

| # | Column | 5-Box Use |
|---|--------|-----------|
| 1 | Symbol | Identity |
| 2 | Latest | Current price |
| 3 | %Change | Today's move |
| 4 | 5D %Chg | Weekly momentum (Box 1) |
| 5 | 1M %Chg | Sustained strength (Box 1+2) |
| 6 | 52W %Chg | Century momentum / 12M return (Box 1) |
| 7 | 52W %/High | Overhead supply (Box 1) |
| 8 | Weighted Alpha | 12M momentum weighted to recent (Box 1) |
| 9 | Perf vs Market 5D | RS vs SPY this week (Box 2) |
| 10 | Perf vs Market 1M | RS vs SPY this month (Box 2) |
| 11 | 3M % Change from Index | RS vs SPY over 3 months — sustained RS confirmation (Box 2) |
| 12 | Volume | Today's volume |
| 13 | 20D RelVol | RVOL — ignition detection (Box 3) |
| 14 | 20D ADR% | Daily range — must be > 3% for ORB viability (Box 3) |
| 15 | 20D ATRP | ATR as % — stop distance check, max 7% (Box 5) |
| 16 | % 50D MA | Distance from 50D SMA (Box 1) |
| 17 | Slope of 50D SMA | SMA direction — rising required for Box 1 |
| 18 | 200D MA Direction | 200D SMA direction — rising required for Stage 2 (Box 1) |
| 19 | TTM Squeeze | Squeeze status — on/fired/off (Box 3 base quality) |
| 20 | Bollinger Bands Rank | 0–100 position within bands — replaces BB% for cleaner reads (Box 3) |
| 21 | 5D P/C Vol | Put/Call ratio (Box 4, PM-04) |
| 22 | IV Pctl | IV Percentile — options structure selection |
| 23 | Market Cap, $K | Size context |
| 24 | Latest Earnings | Binary event proximity (Box 4, PEAD detection) |
| 25 | Sector | Sector RS check (Box 2) |

**Changes from prior view:** Added 3M % Change from Index, 20D ADR%, 200D MA Direction,
Bollinger Bands Rank. Dropped %Chg(Pre) (intraday view only), BB% (replaced by Bollinger
Bands Rank), Short Int %Chg (options/flow view), Days2Cover (options/flow view), 5D IV Chg
(options/flow view).

---

### PEAD / EP View

Used by screeners: **9, 11** (earnings drift and gap scanners).

Adds earnings surprise history and gap data. Enables PM-02 and PM-03 evaluation in one row.

| # | Column | 5-Box Use |
|---|--------|-----------|
| 1 | Symbol | Identity |
| 2 | Latest | Current price |
| 3 | %Change | Today's move |
| 4 | Gap Up % | EP gap magnitude — must be > 10% for Type A (Box 4, PM-01) |
| 5 | 5D %Chg | Weekly momentum (Box 1) |
| 6 | 1M %Chg | Sustained strength (Box 1+2) |
| 7 | 52W %/High | Overhead supply (Box 1) |
| 8 | Weighted Alpha | 12M momentum weighted to recent (Box 1) |
| 9 | Perf vs Market 5D | RS vs SPY post-event (Box 2) |
| 10 | Perf vs Market 1M | RS vs SPY this month (Box 2) |
| 11 | Volume | Today's volume |
| 12 | 20D RelVol | RVOL — confirms institutional volume on the event (Box 3) |
| 13 | 20D ADR% | Daily range check (Box 3) |
| 14 | 20D ATRP | Stop distance check (Box 5) |
| 15 | % 50D MA | Distance from 50D SMA (Box 1) |
| 16 | Slope of 50D SMA | SMA direction (Box 1) |
| 17 | 200D MA Direction | 200D SMA direction (Box 1) |
| 18 | Earnings Surprise% | Current quarter surprise — core PEAD signal (PM-02) |
| 19 | Earnings Surprise% 1-Qtr Ago | Beat history Q-1 — PM-03 requires ≥3 of 4 beats |
| 20 | Earnings Surprise% 2-Qtrs Ago | Beat history Q-2 |
| 21 | Earnings Surprise% 3-Qtrs Ago | Beat history Q-3 |
| 22 | IV Pctl | IV Percentile — options structure selection |
| 23 | 5D P/C Vol | Put/Call ratio — informed flow around earnings (Box 4) |
| 24 | Latest Earnings | Earnings date — proximity and recency (Box 4) |
| 25 | Sector | Sector context (Box 2) |

**Key usage:** Columns 18–21 show the last four quarters of earnings surprises. PM-03
(pre-earnings drift) requires beat ≥3 of 4 — visible at a glance. PM-02 prioritizes
Surprise% > 10% (top decile) for the strongest drift signal.

---

### Options / Flow View

Used by screeners: **6, 7, 12** (put/call ratios, short squeeze watchlist).

Surfaces options flow, IV dynamics, and short interest. Drops trend detail in favour of
flow and squeeze data.

| # | Column | 5-Box Use |
|---|--------|-----------|
| 1 | Symbol | Identity |
| 2 | Latest | Current price |
| 3 | %Change | Today's move |
| 4 | 5D %Chg | Weekly momentum (Box 1) |
| 5 | 1M %Chg | Sustained strength (Box 1+2) |
| 6 | 52W %/High | Overhead supply (Box 1) |
| 7 | Weighted Alpha | 12M momentum (Box 1) |
| 8 | Perf vs Market 5D | RS vs SPY this week (Box 2) |
| 9 | Volume | Today's volume |
| 10 | 20D RelVol | RVOL — volume context (Box 3) |
| 11 | % 50D MA | Distance from 50D SMA (Box 1) |
| 12 | Slope of 50D SMA | SMA direction (Box 1) |
| 13 | 5D P/C Vol | Put/Call ratio — 5-day average (Box 4, PM-04) |
| 14 | 1M Put/Call Vol | Put/Call ratio — 1-month average, smooths daily noise (PM-04) |
| 15 | IV Pctl | IV Percentile — options structure selection |
| 16 | 5D IV Chg | IV trend this week — spiking IV = demand (PM-04) |
| 17 | 1M IV Chg | IV trend this month — sustained IV expansion (PM-04) |
| 18 | Short Interest, K | Absolute short position size (PM-11) |
| 19 | Short Float | Short interest as % of float — squeeze threshold > 15% (PM-11) |
| 20 | Short Int %Chg | Short interest change — rising = shorts building (PM-11) |
| 21 | Days to Cover | Days to cover at avg volume — > 4 = squeeze potential (PM-11) |
| 22 | 1M Total Vol | 1-month total options volume — liquidity context |
| 23 | 1M Total OI | 1-month total open interest — position size context |
| 24 | Total Volume/OI Ratio | Aggregate Vol/OI — simplified UOA signal at underlying level |
| 25 | Latest Earnings | Binary event proximity (Box 4) |

**Key usage:** Columns 13–17 show the options flow picture. Columns 18–21 show the short
squeeze picture. Cross-referencing High Put Ratio (#6) results with Short Float > 15% from
screener #12 identifies the highest-probability squeeze setups.

---

### Intraday View

Used by screener: **8** (manual RVOL spike scan during execution windows).

Adds pre-market and gap data for real-time evaluation. Used on the Barchart website, not
in the email pipeline.

| # | Column | Use |
|---|--------|-----|
| 1 | Symbol | Identity |
| 2 | Latest | Current price |
| 3 | %Change | Today's move |
| 4 | %Chg(Pre) | Pre-market gap detection — Type A EP signal |
| 5 | Gap Up % | Gap magnitude — > 8% = potential EP |
| 6 | 5D %Chg | Weekly momentum (Box 1) |
| 7 | 1M %Chg | Sustained strength (Box 1+2) |
| 8 | 52W %/High | Overhead supply (Box 1) |
| 9 | Weighted Alpha | 12M momentum (Box 1) |
| 10 | Perf vs Market 5D | RS vs SPY this week (Box 2) |
| 11 | Volume | Today's volume — live during session |
| 12 | 20D RelVol | RVOL — must be > 2.0 for this scanner |
| 13 | 20D ADR% | Daily range check — ADR > 3% (Box 3) |
| 14 | 20D ATRP | Stop distance check (Box 5) |
| 15 | % 50D MA | Distance from 50D SMA (Box 1) |
| 16 | Slope of 50D SMA | SMA direction (Box 1) |
| 17 | 200D MA Direction | 200D SMA direction (Box 1) |
| 18 | TTM Squeeze | Squeeze status (Box 3) |
| 19 | Bollinger Bands Rank | Band position 0–100 (Box 3) |
| 20 | 5D P/C Vol | Put/Call ratio (Box 4) |
| 21 | IV Pctl | IV Percentile — structure selection |
| 22 | Daily Closing Range | Where it closed in the day's range — top 25% required for last-hour entries |
| 23 | Market Cap, $K | Size context |
| 24 | Latest Earnings | Binary event check (Box 4) |
| 25 | Sector | Sector context (Box 2) |

**Key usage:** At 9:45, sort by 20D RelVol descending, filter %Change > 2%. At 15:00,
additionally check Daily Closing Range — only enter stocks in top 25% of range.

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
| Underlying Price | > $5 | Match stock screener base filter |
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

## View-to-Screener Map

| View | Screeners | Email / Manual |
|------|-----------|----------------|
| Standard | 1, 2, 3, 4, 5, 10 | Email |
| PEAD/EP | 9, 11, 13 | Email |
| Options/Flow | 6, 7, 12, 14 | Email |
| Intraday | 8 | Manual (Barchart website) |
| UOA (options) | UOA | Email |

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

**Config reference** (`config.yaml` column mapping):

Mapping covers all four stock views. Fields absent from a given view's CSV are left null.

```yaml
column_mapping:
  # Identity
  Symbol: symbol
  Latest: price
  Sector: sector
  "Market Cap, $K": market_cap_k

  # Price change
  "%Change": change_pct
  "%Chg(Pre)": change_pre_pct          # Intraday view only
  "Gap Up %": gap_up_pct               # PEAD/EP + Intraday views
  "5D %Chg": change_5d_pct
  "1M %Chg": change_1m_pct
  "52W %Chg": change_52w_pct
  "52W %/High": high_52w_distance_pct

  # Momentum & RS
  "Weighted Alpha": weighted_alpha
  "Perf vs Market 5D": perf_vs_market_5d
  "Perf vs Market 1M": perf_vs_market_1m
  "3M % Change from Index": perf_vs_market_3m  # Standard view

  # Volume & range
  Volume: volume
  "20D RelVol": rvol_20d
  "20D ADR%": adr_pct_20d              # Standard + PEAD/EP + Intraday views
  "20D ATRP": atr_pct_20d

  # Trend
  "% 50D MA": pct_from_50d_sma
  "Slope of 50D SMA": slope_50d_sma
  "200D MA Direction": direction_200d_sma  # Standard + PEAD/EP + Intraday views

  # Base quality
  "TTM Squeeze": ttm_squeeze            # Standard + Intraday views
  "Bollinger Bands Rank": bb_rank       # Standard + Intraday views
  "Daily Closing Range": daily_closing_range  # Intraday view only

  # Options flow
  "5D P/C Vol": put_call_vol_5d
  "IV Pctl": iv_percentile
  "1M Put/Call Vol": put_call_vol_1m    # Options/Flow view
  "5D IV Chg": iv_chg_5d               # Options/Flow view
  "1M IV Chg": iv_chg_1m               # Options/Flow view
  "1M Total Vol": options_vol_1m        # Options/Flow view
  "1M Total OI": options_oi_1m          # Options/Flow view
  "Total Volume/OI Ratio": vol_oi_ratio # Options/Flow view

  # Short interest
  "Short Interest, K": short_interest_k  # Options/Flow view
  "Short Float": short_float             # Options/Flow view + PEAD/EP (short framework)
  "Short Int %Chg": short_int_chg_pct    # Options/Flow view
  "Days to Cover": days_to_cover         # Options/Flow view

  # Earnings
  "Latest Earnings": latest_earnings
  "Earnings Surprise%": earnings_surprise_pct        # PEAD/EP view
  "Earnings Surprise% 1-Qtr Ago": earnings_surprise_q1  # PEAD/EP view
  "Earnings Surprise% 2-Qtrs Ago": earnings_surprise_q2  # PEAD/EP view
  "Earnings Surprise% 3-Qtrs Ago": earnings_surprise_q3  # PEAD/EP view
```

**Manual fallback** (when pipeline is unavailable):
1. Run each screener on Barchart with the appropriate view
2. Apply 5-box checklist using the column layout
3. Add passing stocks to watchlist — max 20 names, prioritize top 5

---

## Daily Schedule

| Time | Action | Scanners |
|------|--------|----------|
| **After 16:00** | Run pipeline | All 12 email scanners (automated) |
| **Evening / pre-market** | Review Daily Prep | None — review `/daily-prep`, set ORB alerts, remove earnings |
| **9:45** | Manual scan | Intraday RVOL Spike (#8) — morning ORB candidates |
| **10:15–15:00** | Dead zone | No scanning, no entries |
| **15:00** | Manual scan | Intraday RVOL Spike (#8) — last-hour candidates |
| **15:15–15:45** | Execute | Last-hour ORB entries from watchlist + RVOL scan |

**Email scanners** (1–7, 9–14) are configured for end-of-day delivery → pipeline runs once daily.
**Intraday RVOL** (8) is manual on Barchart website — run only during execution windows.
