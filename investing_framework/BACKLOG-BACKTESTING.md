# Backtesting Backlog

Goal: every active and pilot strategy has at least one backtest run with a go/no-go verdict.
"Verified" means: positive net EV after realistic transaction costs over a minimum sample of 100 trades
or 3 years of data (whichever is less), with a clearly stated confidence level.

Strategies that cannot be backtested due to data or rule-definition blockers are tracked separately
in `BACKLOG-BACKTESTING-HOLD.md`.

---

## Epic BT-0 — Project Restructure

One-time housekeeping before any backtest code is written. Must complete before BT-1.

### BT-0-S1: Restructure `intraday_pm/` and `swing_pm/`

Status: Done

**As a** developer,
**I want to** impose a consistent subdirectory layout on the two profit-mechanism directories,
**So that** data prep, exploratory analysis, and formal backtests are clearly separated and easy to navigate.

**Changes required:**

`intraday_pm/`:
- [ ] Create subdirectories: `data/`, `analysis/`, `backtests/`
- [ ] Move existing flat scripts into `analysis/` (range break, vwap extrema, thursday_friday_monday, etc.)
- [ ] Move data-building scripts (if any) into `data/`
- [ ] Delete `backtester/` entirely (InfluxDB-coupled Backtrader; no path back to relevance)
- [ ] Leave `paused/` in place

`swing_pm/`:
- [ ] Create subdirectories: `data/`, `analysis/`, `backtests/`
- [ ] Move `create_data_*.py` and `refresh_sensor_data.py` → `data/`
- [ ] Move `eval_*.py`, `ath_run_analysis_pipeline.py`, `atrx_*.py` → `analysis/`
- [ ] Leave `paused/` in place (create if absent)

**Acceptance criteria:**
- [ ] No flat `.py` files remain directly under `intraday_pm/` or `swing_pm/` (except `__init__.py`)
- [ ] `intraday_pm/backtester/` is deleted
- [ ] All existing imports still resolve (update any cross-file imports broken by the move)
- [ ] Existing tests in `finance/tests/` still pass

**Dependencies:** None
**Complexity:** S

---

### BT-0-S2: Restructure `_data/` and update `dolt_data.py`

Status: Done

**As a** developer,
**I want to** consolidate Dolt mount points, rename ambiguous directories, and create output directories,
**So that** raw sources, computed datasets, and backtest results are clearly distinguished.

**Directory changes:**

| Current | New | Reason |
|---------|-----|--------|
| `_data/earnings/` | `_data/dolt/earnings/` | Group all Dolt mount points under `dolt/` |
| `_data/stocks/` | `_data/dolt/stocks/` | Same |
| `_data/options/` | `_data/dolt/options/` | Same |
| `_data/rates/` | `_data/dolt/rates/` | Same |
| `_data/hist_earnings/` | `_data/earnings_source/` | Clarify it is a static CSV source, not a dolt DB |
| `_data/earnings_cleaned/` | `_data/research/swing/earnings/` | Derived dataset → belongs under `research/` |
| `_data/momentum_earnings/` | `_data/research/swing/momentum_earnings/` | Same |
| `_data/swing_data/` | `_data/research/swing/swing_data/` | Same |
| `_data/ath_run_analysis/` | `_data/research/swing/ath_run_analysis/` | Analysis output, not raw data |
| *(new)* | `_data/research/intraday/` | Home for intraday analysis outputs |
| *(new)* | `_data/backtest_results/intraday/` | Formal backtest outputs — BT-4, BT-5, BT-6 |
| *(new)* | `_data/backtest_results/swing/` | Formal backtest outputs — BT-2, BT-3 |

**Code changes:**
- [ ] Update `utils/dolt_data.py` connection strings: `localhost:3306/{db}` paths unchanged (MySQL); update any file-path references to `_data/{db}/` → `_data/dolt/{db}/`
- [ ] Update `swing_pm/data/create_data_earnings.py`: input path `hist_earnings/` → `earnings_source/`; output path `earnings_cleaned/` → `research/swing/earnings/`
- [ ] Search for any other scripts referencing the renamed directories and update paths

**Acceptance criteria:**
- [ ] All Dolt databases accessible via updated `dolt_data.py`
- [ ] `create_data_earnings.py` runs without error against new paths
- [ ] `finance/utils/intraday.py` still resolves `_data/intraday/` (unchanged — confirm no breakage)
- [ ] `_data/backtest_results/` and `_data/research/` directories exist
- [ ] Existing tests pass

**Dependencies:** BT-0-S1
**Complexity:** S

---

### BT-0-S3: Migrate pkl persistence to Parquet

Status: Done

**As a** developer,
**I want to** replace all `.pkl` files with Parquet equivalents,
**So that** data loads are faster, files are smaller, and there are no pickle/pandas-version compatibility surprises.

**Scope — files to migrate:**

| Current pkl path | New Parquet path | Writer |
|------------------|------------------|--------|
| `_data/state/liquid_stocks.pkl` | `_data/state/liquid_stocks.parquet` | `swing_pm/data/create_data_liquid_underlying.py` |
| `_data/state/liquid_etfs.pkl` | `_data/state/liquid_etfs.parquet` | same |
| `_data/state/delisted.pkl` | `_data/state/delisted.parquet` | `swing_pm/data/refresh_sensor_data.py` |
| `_data/research/swing/swing_data/*.pkl` | `_data/research/swing/swing_data/*.parquet` | `utils/swing_trading_data.py` cache |
| `_data/research/swing/momentum_earnings/*.pkl` | `_data/research/swing/momentum_earnings/*.parquet` | `swing_pm/data/create_data_momentum_earnings_analysis.py` |
| `_data/research/swing/ath_run_analysis/*.pkl` | `_data/research/swing/ath_run_analysis/*.parquet` | `swing_pm/analysis/ath_run_analysis_pipeline.py` |
| `_data/research/swing/all_ath_run_analysis.pkl` | `_data/research/swing/all_ath_run_analysis.parquet` | same |
| `N:/…/future_following_range_break/**/*.pkl` | same path, `.parquet` | `intraday_pm/backtests/range_break_summary.py` (aggregate once) |

**Acceptance criteria:**
- [ ] All writers updated to write `.to_parquet()` instead of `.to_pickle()`
- [ ] All readers updated to read `.read_parquet()` instead of `.read_pickle()`
- [ ] Old `.pkl` files deleted after confirming readers work
- [ ] `utils/swing_trading_data.py` cache uses Parquet (check `CACHE_DIR` usage)
- [ ] `utils/underlyings.py` reads `liquid_stocks.parquet` and `liquid_etfs.parquet`
- [ ] `utils/dolt_data.py` reads `delisted.parquet`
- [ ] Existing tests in `finance/tests/` still pass
- [ ] Google Drive backtest pkl files: if migration is impractical due to volume (1 000+ files per symbol), create a one-time consolidation step that reads all daily pkls and writes one Parquet per symbol × timeframe instead

**Dependencies:** BT-0-S1, BT-0-S2
**Complexity:** M

---

## Epic BT-1 — Infrastructure

Shared prerequisites. All other epics depend on at least one story here.

### BT-1-S1: Define backtesting framework and cost model

Status: Done

**As a** systematic trader,
**I want to** agree on one backtesting framework and a realistic cost model before writing any strategy tests,
**So that** all results are comparable and not inflated by zero-cost assumptions.

**Acceptance criteria:**
- [ ] Framework selected: extend existing `finance/intraday_pm/backtester/` (Backtrader) OR migrate to vectorbt/custom pandas — document decision with rationale
- [ ] Cost model defined per instrument class:
  - Equity (stocks): commission + 0.05% estimated slippage per side
  - Index futures (ES, NQ, DAX, FTSE): tick spread + $5–12 commission per side
  - Options: model as Black-Scholes mid ± 20% of bid-ask width (same approach as `__noon_to_close_evaluation.py`)
- [ ] Forward-looking bias guard: no future bar data used in signal computation (walk-forward or strict look-back window enforcement)
- [ ] Minimum sample standard: ≥ 100 trades OR ≥ 3 years of daily data

**Dependencies:** None
**Complexity:** M

---

### BT-1-S2: Confirm intraday data pipeline coverage

Status: Done

**As a** developer,
**I want to** confirm which instruments and timeframes are available in the existing IBKR data pipeline,
**So that** I know which backtest stories can proceed immediately vs need data collection first.

**Acceptance criteria:**
- [x] Inventory of available symbols in `finance/_data/intraday/` — 88 symbols across 5 schemas (cfd, forex, future, index, stock); InfluxDB decommissioned, replaced by per-symbol Parquet at 1-min resolution
- [x] Confirmed availability for: IBDE40 (DAX), IBGB100 (FTSE), IBEU50 (ESTX50), IBUS30 (Dow), IBUS500 (SPX), IBUST100 (NDX), IBJP225 — all present in `cfd/` schema
- [x] Confirmed date range per symbol — all 7 targets: 2013-05-24 → 2026-04-17 (≥ 13 years; well past ≥ 2020 requirement)
- [x] Identified gaps: none — no backfill required for any target symbol

**Findings:** Bars stored as 1-min OHLCV Parquet with columns `o,h,l,c,v,a,bc,hvo,hvh,hvl,hvc,ivo,ivh,ivl,ivc`. Resample to 5-min or 15-min in-memory at backtest time.

**Dependencies:** None
**Complexity:** S

---

### BT-1-S3: Confirm dolt earnings DB fields and coverage

Status: Done

**As a** developer,
**I want to** confirm what fields and date coverage the dolt earnings database provides,
**So that** PEAD, negative PEAD, accruals, and F-score backtests can proceed without a separate data acquisition step.

**Acceptance criteria:**
- [x] Connect to dolt earnings DB and list available tables — 10 tables: `eps_history`, `eps_estimate`, `earnings_calendar`, `income_statement`, `cash_flow_statement`, `balance_sheet_assets`, `balance_sheet_liabilities`, `balance_sheet_equity`, `sales_estimate`, `rank_score`
- [x] PEAD fields confirmed: `eps_history` (act_symbol, period_end_date, reported, estimate); `earnings_calendar` (act_symbol, date, when); SUE computable as `(reported − estimate) / |estimate|`
- [x] Accruals fields confirmed: `cash_flow_statement`.net_income, `cash_flow_statement`.net_cash_from_operating_activities, `balance_sheet_assets`.total_assets — all quarterly
- [x] F-score fields confirmed: all 9 Piotroski components computable from `income_statement`, `cash_flow_statement`, `balance_sheet_assets`, `balance_sheet_liabilities`
- [x] Date coverage: `eps_history` 2016–2026, financials 2012–2026; ~8 000–9 000 US symbols. ≥ 2019 confirmed.
- [x] BT-3-S3 (accruals) and BT-3-S4 (F-score) stories created

**Dependencies:** None
**Complexity:** S

---

### BT-1-S4: Explore dolt options DB for DRIFT cross-validation

Status: Pending

**As a** developer,
**I want to** understand what the dolt options database contains,
**So that** I can cross-validate Optionsomega DRIFT simulations and unblock H3-02 (OTM informed flow).

**Acceptance criteria:**
- [ ] Connect to dolt options DB and list available tables, symbols, and date range
- [ ] Confirm whether historical OI by strike is available for individual US equities (→ unblocks H3-02)
- [ ] Confirm whether IV timeseries is available for ESTX50 underlyings (→ enables XSP/ESTX50 iron condor cross-validation)
- [ ] Document findings in a comment in this story; update HOLD document if H3-02 is unblocked

**Dependencies:** None
**Complexity:** S

---

## Epic BT-2 — Swing Momentum Strategies (EOD)

Source: `TradingPlaybook.md` — Active mechanisms PM-01 through PM-03, PM-08, PM-09, plus VIX trade.

### BT-2-S1: PM-01 Episodic Pivot (EP / PEAD-entry)

Status: Pending

**Backtest definition:**
- Universe: US stocks, Avg Vol > 1M, Price > $3
- Signal: gap ≥ 10% on ≥ 5× avg volume, catalyst confirmed, closes top 25% of day's range
- Entry: ORB above 15-min candle, first 45 min only
- Stop: below entry candle low; hard max 7%
- Exit: first of — 2R take (30–50%), 4R take (further 30%), 5 EMA 2-close break, 50 days elapsed
- Direction: long only

**Acceptance criteria:**
- [ ] At least 3 years of backtest data (2021–2024 minimum)
- [ ] Minimum 100 triggering events
- [ ] Metrics reported: win rate, avg win/loss, expectancy, max drawdown, Sharpe
- [ ] Costs applied per BT-1-S1 cost model
- [ ] Results segmented by market regime (GO vs NO-GO per playbook rules)
- [ ] Net expectancy > 0 in GO regime → VERIFIED; else flag for review

**Data:** Intraday 15-min bars for stocks + earnings calendar
**Dependencies:** BT-1-S1, BT-1-S2, BT-1-S3
**Complexity:** L

---

### BT-2-S2: PM-01 VCP Breakout (Type B)

Status: Pending

**Backtest definition:**
- Universe: US stocks, Stage 2 (price > 50d > 200d SMA, 200d rising), within 25% of 52-week high
- Signal: 2–6 week base with ≥ 3 contraction points, VDU in final week, volume ≥ 40% above avg on breakout day
- Entry: ORB above 30-min candle on breakout day
- Stop: below base low; max 7%
- Exit: same as EP; 3–5 day rule applies (first partial after 3 strong days)
- Direction: long only

**Acceptance criteria:**
- [ ] Base detection algorithm defined algorithmically (contraction in ATR, volume profile, candle range)
- [ ] ≥ 100 triggering events over ≥ 3 years
- [ ] Same metrics as BT-2-S1
- [ ] Net expectancy > 0 → VERIFIED

**Data:** EOD OHLCV + intraday 30-min for entry confirmation
**Dependencies:** BT-1-S1, BT-1-S2
**Complexity:** XL (base detection algorithm is the hard part)

---

### BT-2-S3: PM-01 EMA Reclaim / Wedge Pop (Type C)

Status: Pending

**Backtest definition:**
- Universe: same as VCP; stock must be in confirmed Stage 2
- Signal: pullback to 10d or 20d EMA on below-average volume, followed by a bullish Elephant Bar or Tail Bar off EMA
- Entry: ORB above 15-min or 30-min candle on reclaim day
- Stop: below 10d EMA; max 7%
- Exit: same staircase; time stop at 50 days

**Acceptance criteria:**
- [ ] EMA pullback depth defined: 10d EMA ± 1 ATR tolerance
- [ ] ≥ 100 events over ≥ 3 years
- [ ] Same metrics as BT-2-S1
- [ ] Segmented by: pullback to 10d vs 20d; volume on pullback day (light vs avg+)
- [ ] Net expectancy > 0 → VERIFIED

**Data:** EOD OHLCV + intraday for entry
**Dependencies:** BT-1-S1, BT-1-S2
**Complexity:** L

---

### BT-2-S4: PM-02 PEAD Window (post-entry drift validation)

Status: Pending

**Backtest definition:**
- Universe: stocks with positive EPS surprise ≥ top 25th percentile, gap ≥ 10%, closes top 25% of range on earnings day
- Measurement: forward return on days 1–60 post-announcement (no entry rule — pure drift quantification)
- Segments: surprise magnitude buckets (0–25%, 25–50%, 50%+ SUE); market cap buckets; GO vs NO-GO regime

**Acceptance criteria:**
- [ ] Positive average forward return in days 1–60 across all SUE buckets → confirms PEAD is present in data
- [ ] SUE buckets produce differentiated returns (larger surprise = longer/stronger drift)
- [ ] At least 200 events (earnings surprises) over ≥ 5 years
- [ ] This validates the drift mechanism underlying PM-01 EP entries

**Data:** Earnings surprise data (BT-1-S3) + EOD OHLCV
**Dependencies:** BT-1-S1, BT-1-S3
**Complexity:** M

---

### BT-2-S5: PM-03 Pre-Earnings Anticipation

Status: Pending

**Backtest definition:**
- Universe: RS stocks (positive 12-1 month momentum), Stage 2, ≥ 3 of last 4 quarters beat consensus
- Entry: T-14 (14 trading days before earnings) if stock is holding 20d EMA with RS intact
- Exit: T-1 (close one day before earnings — hard rule)
- Stop: close below 20d EMA or 50% of premium loss (options: ATM call 45–60 DTE)

**Acceptance criteria:**
- [ ] Average return T-14 to T-1 measured for qualifying stocks vs random entry control group
- [ ] Win rate and avg return compared: qualifying stocks vs all stocks entering earnings window
- [ ] At least 150 events over ≥ 3 years
- [ ] IV expansion measured (entry IVR vs T-5 IVR) — does pre-earnings drift carry a vega tailwind?
- [ ] Net expectancy > 0 vs random → VERIFIED

**Data:** Earnings calendar + EOD OHLCV + IV history (IBKR)
**Dependencies:** BT-1-S1, BT-1-S3
**Complexity:** M

---

### BT-2-S6: PM-08 Overnight Reversal

Status: Pending

**Backtest definition:**
- Universe: RS stocks (positive 12-1 month momentum, above 200d SMA)
- Signal: stock down > 2% intraday on above-average volume (no fundamental breakdown — news filter required)
- Entry: MOC (market-on-close) on signal day
- Exit: market-on-open next morning (pure overnight hold)
- Comparison: random-entry overnight returns as control

**Acceptance criteria:**
- [ ] Average overnight return on signal days vs random days — statistically significant difference
- [ ] Segmented by: intraday decline magnitude (2–3%, 3–5%, > 5%); VIX regime; day of week
- [ ] At least 200 events over ≥ 3 years
- [ ] Net expectancy > 0 after costs (minimal — overnight stock hold) → VERIFIED

**Data:** EOD OHLCV (close and next-day open only)
**Dependencies:** BT-1-S1
**Complexity:** S
**Note:** `BACKLOG-RESEARCH.md` already notes this as "easiest to research" — already partially in plan.

---

### BT-2-S7: PM-09 Mean Reversion to Trend (Bollinger Touch / Type C quantification)

Status: Pending

**Backtest definition:**
- Universe: stocks in Stage 2 (20d SMA rising, above 50d and 200d SMA)
- Signal: price touches or closes below lower Bollinger Band (2σ, 20d) while 20d SMA slope is positive
- Entry: next-day open after signal close
- Exit: close above 20d SMA (or 5–15 days elapsed)
- Stop: close below 50d SMA

**Acceptance criteria:**
- [ ] Average forward return 1, 3, 5, 10 days after Bollinger touch in Stage 2 uptrend
- [ ] Compared to: same signal in Stage 3/4 (non-uptrend) as control
- [ ] Confirms: the Bollinger touch in an uptrend reliably precedes mean reversion to SMA
- [ ] At least 300 events over ≥ 3 years (will be frequent)
- [ ] Net positive 10-day expectancy → VERIFIED

**Data:** EOD OHLCV
**Dependencies:** BT-1-S1
**Complexity:** S
**Note:** `BACKLOG-RESEARCH.md` already designed the data query for this; start here.

---

### BT-2-S8: VIX Mean Reversion Index Trade

Status: Pending

**Backtest definition:**
- Universe: SPY, QQQ (index ETFs only)
- Signal: VIX spikes > 25, then forms a "lower high" on the daily chart (day N close < day N-1 high, both > 25)
- Entry: ORB above 15-min candle on the lower-high confirmation day
- Stop: below ORB candle low; max 5%
- Exit: VIX falls below 20 OR 3R reached OR 20 days elapsed

**Acceptance criteria:**
- [ ] Win rate and average R on qualifying VIX signals vs random index ORB entries
- [ ] At least 20 qualifying events (VIX spikes are infrequent; 2010–2024 covers ~30)
- [ ] Net expectancy positive → VERIFIED

**Data:** VIX daily + SPY/QQQ intraday 15-min
**Dependencies:** BT-1-S1, BT-1-S2
**Complexity:** S

---

## Epic BT-3 — Factor-Driven Short Strategies

Source: `StrategyIdeasAssessment.md` — Idea 1.

### BT-3-S1: Negative PEAD Short (put debit spread)

Status: Pending

**Backtest definition:**
- Universe: US stocks, Avg Vol > 1M, < 20% float short, Stage 3/4 (price below declining 30-week MA)
- Signal: negative EPS surprise (bottom 25% SUE), stock closes bottom 25% of range on earnings day
- Entry: first recovery day (price moves up 1–3% from post-earnings low) — enter put debit spread or short stock
- Stop: +15% against position (hard close) or 20% of spread width remaining
- Exit: 20–60 days elapsed, or short stop triggered, or stock recovers above 50% of post-earnings range

**Acceptance criteria:**
- [ ] Average forward return on days 1–60 after negative SUE events (control: all earnings days)
- [ ] Negative drift confirmed in bottom SUE quartile vs baseline
- [ ] Entry-day timing test: recovery-day entry vs gap-day entry (which produces better fills)
- [ ] At least 200 negative PEAD events over ≥ 3 years
- [ ] Net expectancy > 0 after borrow cost proxy (0.5% annualised on notional) → VERIFIED

**Data:** Earnings surprise (BT-1-S3) + EOD OHLCV
**Dependencies:** BT-1-S1, BT-1-S3
**Complexity:** M

---

### BT-3-S2: Consecutive Miss + Guidance Cut

Status: Pending

**Backtest definition:**
- Universe: same as BT-3-S1
- Signal: second consecutive negative surprise AND forward guidance below prior consensus
- Entry / Stop / Exit: same as BT-3-S1
- Comparison: single-miss events (BT-3-S1) vs double-miss events (this story) — does repeat miss add incremental edge?

**Acceptance criteria:**
- [ ] Average 60-day forward return for double-miss + guidance cut vs single miss
- [ ] At least 80 qualifying events over ≥ 3 years (will be fewer than single miss)
- [ ] Return is materially worse than single-miss baseline → confirms the signal adds value as a selection filter
- [ ] Net expectancy > 0 → VERIFIED

**Data:** Earnings history ≥ 2 quarters per ticker (BT-1-S3) + guidance revision data
**Dependencies:** BT-1-S1, BT-1-S3
**Complexity:** M

---

### BT-3-S3: Accruals Factor Short Screen

Status: Pending

**Backtest definition:**
- Universe: US stocks, Avg Vol > 1M, Price > $3
- Signal: accruals ratio in bottom decile = `(net_income − CFO) / total_assets` (most negative = highest quality; most positive = aggressive earnings)
- Direction: short stocks with **highest** accruals ratio (earnings not backed by cash); long stocks with **lowest** accruals ratio as control
- Holding period: quarterly rebalance (one quarter after signal quarter)
- Stop: hard 25% position stop; position-level only (no intraday)

**Acceptance criteria:**
- [ ] Accruals decile returns computed quarterly (2019–2025)
- [ ] Top-decile (aggressive accruals) underperforms bottom-decile (cash-backed earnings) by statistically significant margin
- [ ] At least 5 years of quarterly observations (≥ 20 rebalance periods)
- [ ] Net forward return negative for top decile (short signal confirmed)
- [ ] Costs applied: stock commission model per BT-1-S1

**Data:** `cash_flow_statement` (net_income, net_cash_from_operating_activities) + `balance_sheet_assets` (total_assets) + EOD OHLCV
**Dependencies:** BT-1-S1, BT-1-S3
**Complexity:** M

---

### BT-3-S4: Piotroski F-Score Long/Short Filter

Status: Pending

**Backtest definition:**
- Universe: US stocks, Avg Vol > 500K, Price > $3, within 30% of 52-week low (value territory)
- Signal: compute 9-point Piotroski F-Score from quarterly financials:
  - Profitability (4): ROA > 0, CFO > 0, delta ROA > 0, accruals < 0
  - Leverage/Liquidity (3): delta long-term debt < 0, delta current ratio > 0, no equity dilution
  - Operating efficiency (2): delta gross margin > 0, delta asset turnover > 0
- Long: F-Score ≥ 8; Short: F-Score ≤ 2
- Holding period: annual rebalance (one year after signal)

**Acceptance criteria:**
- [ ] F-Score 8–9 group annual return vs F-Score 1–2 group (long-short spread)
- [ ] At least 5 years of annual observations (2019–2025)
- [ ] Long group outperforms short group by ≥ 5% annualised → edge confirmed
- [ ] Segmented by market cap bucket (large/mid/small) — does F-Score work better in small caps?
- [ ] Costs applied: stock commission model per BT-1-S1

**Data:** `income_statement`, `cash_flow_statement`, `balance_sheet_assets`, `balance_sheet_liabilities` + EOD OHLCV
**Dependencies:** BT-1-S1, BT-1-S3
**Complexity:** M

---

## Epic BT-4 — European Session Intraday (Hougaard Situational Strategies)

Source: `StrategyIdeasAssessment.md` — Idea 3; `_research/TomHougaard/RESEARCH_SUMMARY.md`.

All strategies in this epic use **one mechanical exit rule**: 2-bar trailing stop (trail stop to the high/low of the two most recently completed bars, updated on every bar close).

### BT-4-S1: Hougaard 1BP on FTSE (highest priority)

Status: Done

**Backtest definition:**
- Instrument: IBGB100 (FTSE 100 futures), 5-min bars
- Session: 08:00 UK open
- Signal: first 5-min bar closes positive (close > open) → **1BP**
- Entry: sell-stop order 2 pts below the 1BP bar's low (short signal)
- Stop: 1BP bar's range (high − low) above entry price
- Exit: 2-bar trailing stop (update after each bar close)
- Regime filter: no red-flag macro events at or before 08:00 (Forex Factory check)
- Both directions (1BN long + 1BP short) tested independently

**Acceptance criteria:**
- [x] Test period: 2020–2025 (5 years minimum; matches Hougaard's 58-month dataset)
- [x] Metrics per direction: win rate, avg win (pts), avg loss (pts), expectancy per trade, Sharpe
- [x] Segmented by: daily volatility regime (daily ATR < 50, 50–80, > 80 FTSE points)
- [x] Costs applied: 2-pt spread equivalent per trade
- [x] Compare 2-bar stop vs ATR trailing stop (20% daily ATR) — which produces better Sharpe?
- [ ] ~~Net expectancy > 0 in at least one direction and one volatility regime → VERIFIED~~ — **NOT MET**

**Findings:** 1BP: 34.7% win rate, −2.06 pts expectancy (2-bar stop). 1BN: 36.4% win rate, −2.18 pts. ATR trailing stop marginally better on Sharpe for 1BP (−0.084 vs −0.133) but still negative EV. No regime shows positive expectancy. **VERDICT: No-go.** The 08:00 IBGB100 open bar reversal does not survive transaction costs over 2020–2025. Full results in `finance/intraday_pm/RESULTS.md`.

**Data:** IBGB100 5-min bars (IBKR pipeline)
**Dependencies:** BT-1-S1, BT-1-S2
**Complexity:** M

---

### BT-4-S2: Hougaard ASRS on DAX (4th 5-min bar bracket)

Status: Done

**Backtest definition:**
- Instrument: IBDE40 (DAX futures), 5-min bars
- Session: 09:00 Frankfurt time
- Signal: 4th 5-min bar (09:15–09:20) completes
- Entry: OCO bracket — buy-stop 2 pts above bar high, sell-stop 2 pts below bar low
- Stop: opposite side of bracket (symmetric; stop = bracket range + 4 pts)
- Exit: 2-bar trailing stop; session close if not stopped out
- Fallback: if 4th bar range < 5 pts, use 5th bar instead
- Regime filter: no red-flag news at or before 09:00 Frankfurt

**Acceptance criteria:**
- [ ] Test period: ≥ 3 years
- [ ] Both long and short sides tested independently
- [ ] Segmented by: signal bar range size (narrow vs normal vs wide); day-of-week
- [ ] Costs applied: 2-pt spread per side
- [ ] Net expectancy > 0 in at least one direction → VERIFIED
- [ ] Comparison: ASRS (4th 5-min) vs SRS (2nd 15-min) — which delivers better Sharpe on DAX?

**Data:** IBDE40 5-min and 15-min bars
**Dependencies:** BT-1-S1, BT-1-S2
**Complexity:** M

---

### BT-4-S3: Hougaard SRS on DAX (2nd 15-min bar bracket)

Status: Done

**Backtest definition:**
- Instrument: IBDE40 (DAX futures), 15-min bars
- Session: 09:00 Frankfurt; signal bar = 2nd 15-min candle (09:15–09:30)
- Entry: OCO bracket ± 2 pts of signal bar high/low
- Stop: 20% of daily ATR(14) — scaled to DAX level (40 pts at DAX 12,000; scale proportionally)
- Exit: 2-bar trailing stop (15-min bars); session close fallback

**Acceptance criteria:**
- [ ] Same structure as BT-4-S2
- [ ] Cross-compared directly with BT-4-S2 (ASRS) on same date range

**Data:** IBDE40 15-min bars
**Dependencies:** BT-1-S1, BT-1-S2, BT-4-S2
**Complexity:** S (shares infrastructure with BT-4-S2)

---

### BT-4-S4: Hougaard Rule of 4 (FOMC bracket)

Status: Done
**Verdict: Go** — FOMC Rule of 4: +7.37 pts expectancy, 47.5% win rate, Sharpe +0.21; Non-FOMC control: −2.07 pts; FOMC edge vs control: +9.44 pts/trade

**Backtest definition:**
- Instrument: IBUS30 (Dow Jones) or IBUS500, 10-min bars
- Trigger: FOMC announcement date and time (public calendar)
- Signal: 4th 10-min bar after announcement closes
- Entry: OCO bracket ± 2 pts of bar 4 high/low
- Stop: opposite bracket side
- Exit: 2-bar trailing stop; 3-hour session cutoff

**Acceptance criteria:**
- [ ] All FOMC announcement dates 2015–2024 used as the event set (≈ 8 per year = ≈ 80 events)
- [ ] Win rate, avg points per trade, expectancy
- [ ] Comparison: Rule of 4 return vs random 10-min bracket on non-FOMC days (same instrument, same time)
- [ ] Net expectancy > 0 → VERIFIED

**Data:** IBUS30 or IBUS500 10-min bars + FOMC calendar (public)
**Dependencies:** BT-1-S1, BT-1-S2
**Complexity:** S

---

## Epic BT-5 — Generic Intraday ORB (US Session)

Source: `StrategyIdeasAssessment.md` — Idea 2.

### BT-5-S1: Opening Range Breakout — ES / NQ (15-min, US open)

Status: Done
**Verdict:** IBUS500 all No-go; IBUST100 long Go (15m: +4.16 pts, 30m: +4.63 pts); gap_up filter on IBUST100 30m long = +13.85 pts

**Backtest definition:**
- Instruments: IBUS500 (ES equivalent), IBUST100 (NQ equivalent), 5-min bars
- Session: 14:30–10:15 UK (09:30–10:15 EST) — first 45-min ORB window
- Signal: price breaks above 15-min OR 30-min opening candle high/low after first candle closes
- Entry: stop order 1 tick above/below the ORB candle
- Stop: opposite side of the ORB candle
- Exit: 2× ORB range target (2R) OR session close fallback
- Test variants: 15-min candle vs 30-min candle; with and without direction filter (intraday trend vs counter-trend)

**Acceptance criteria:**
- [ ] Test period: ≥ 3 years
- [ ] Both long and short sides tested
- [ ] Costs applied: 2-tick spread on ES/NQ (≈ $25 per round trip)
- [ ] Segmented by: day-of-week; VIX regime; gap open (gap up/down vs flat open)
- [ ] Net expectancy > 0 after costs in at least one variant → VERIFIED
- [ ] Compare to Hougaard EU strategies on same instruments where overlap exists

**Data:** IBUS500, IBUST100 5-min bars
**Dependencies:** BT-1-S1, BT-1-S2
**Complexity:** M

---

## Epic BT-6 — Proprietary `intraday_pm` Strategies

Code already exists. Stories here are about reading evaluation output, finalising parameter selection, and producing a go/no-go verdict.

### BT-6-S1: Following Range Break — finalise optimal variant

Status: Done

**What's done:** `future_following_range_break.py` runs the full backtest; `_evaluation.py` computes per-variant metrics. Results are rendered at runtime but not documented.

**Work required:**
- [x] Run `future_following_range_break_evaluation.py` across all instruments and timeframes
- [x] Record: loss pct, move mean/median, move_max per variant × instrument × timeframe
- [x] Identify the optimal stop variant (best expectancy:drawdown ratio) per instrument
- [x] Document findings in `finance/intraday_pm/RESULTS.md`
- [x] Verdict: which instruments × timeframes produce net positive EV → VERIFIED for those; others flagged

**Acceptance criteria:**
- [x] All 7 variants evaluated for each of: IBDE40, IBGB100, IBES35, IBUS500 on 5m and 10m bars
- [x] Optimal variant identified and documented with supporting metrics
- [x] At least one instrument × timeframe × variant combination shows net EV > 0

**Findings:** `02_pct` long is dominant across all instruments. Short side is no-go everywhere after costs. Best instruments: IBJP225 5m (+0.045% net EV), IBUST100 5m (+0.029%), IBJP225 10m (+0.035%). IBES35 is no-go at all timeframes. Full details in `finance/intraday_pm/RESULTS.md`.

**Dependencies:** BT-1-S1 (cost model to apply)
**Complexity:** S

---

### BT-6-S2: VWAP Extrema Bracket — extract high-edge time slots

Status: Done
**Verdict: Go** — All 4 symbols positive EV after costs. IBDE40 +45.80 pts, IBGB100 +17.40 pts, IBUS500 +10.69 pts, IBUST100 +52.36 pts (filtered to 52-66 high-edge time slots per symbol). Strong session-open concentration (09:00-09:30 for DAX, 08:30-09:00 for US)

**What's done:** `futures_vwap_extrema.py` computes `success_rate` (directional win rate) per 5-min candle time slot and saves PNGs per symbol.

**Work required:**
- [ ] Review saved PNGs for each symbol; record time slots where `success_rate` > 55% consistently (≥ 3 of 5 instruments)
- [ ] Define a time-filtered bracket strategy: only trade the bracket at identified high-edge time slots
- [ ] Implement the filtered strategy as a formal backtest (entry at identified slot, stop per bracket, 2-bar exit)
- [ ] Document findings in `finance/intraday_pm/RESULTS.md`

**Acceptance criteria:**
- [ ] At least 3 time slots identified with sustained `success_rate` > 55% across multiple instruments
- [ ] Filtered bracket backtest shows net EV > 0 after costs on identified slots → VERIFIED

**Dependencies:** BT-1-S1, BT-6-S1 (shares infrastructure)
**Complexity:** M

---

### BT-6-S3: Noon Iron Butterfly — resume and verify

Status: Pending

**What's done:** `paused/__noon_to_close_evaluation.py` prices iron butterflies at noon using Black-Scholes with actual IV; P&L tracked per day. Paused for unknown reason.

**Work required:**
- [ ] Investigate why paused — check if issue is IV data availability, instrument coverage, or P&L results
- [ ] If data issue: fix and re-run; if P&L issue: review the distribution before deciding
- [ ] Key analysis: P&L distribution on high-volatility days (VIX > 25) — are tail losses unacceptably large?
- [ ] Compute: win rate, avg win, avg loss, EV per day, Sharpe
- [ ] Segment by: day-of-week, VIX regime, time-of-year

**Acceptance criteria:**
- [ ] Pause reason documented
- [ ] P&L distribution on high-IV days reviewed and acceptable (tail loss < 5× average daily premium)
- [ ] Net EV > 0 over ≥ 2 years of data → VERIFIED; else document failure mode

**Dependencies:** BT-1-S1
**Complexity:** M

---

### BT-6-S4: Consolidate filter signal research into documented findings

Status: Done

**Covers:** `thursday_friday_monday.py`, `futures_close_to_min.py`, `underlying_extreme_days.py`

**Work required:**
- [x] Run all three scripts; extract key numeric findings
- [x] Thursday/Friday/Monday: record weekday × prior-day-structure hc/lc probability table per instrument
- [x] PDC proximity: record mean/median distance to prior close per 30-min window per instrument
- [x] Post-extreme-day: record average 1-week, 2-week, 4-week returns following ±2% days per instrument
- [x] Document all three in `finance/intraday_pm/FILTER_SIGNALS.md`
- [x] Define how each filter will be applied in BT-4 and BT-5 strategies

**Acceptance criteria:**
- [x] All three studies produce numerical findings (not just PNGs)
- [ ] ~~At least one weekday shows statistically different directional probability from the mean~~ — **NOT MET** (hc/lc rates uniformly high across all weekdays; no actionable bias)
- [x] Post-extreme-day returns confirm or deny usefulness as DRIFT entry timing signal — **confirmed**: positive mean reversion 2–4 weeks after down extreme days
- [x] Filter application rules written up for BT-4 and BT-5 evaluation

**Findings:**
- Weekday bias: none found (no-go as filter)
- PDC proximity: first 30-min session window has lowest distance (0.06–0.18%); use as confidence filter in BT-4/BT-5
- Post-extreme-day: consistent mean-reversion 2–4 weeks after ±2% days; supports DRIFT entry timing after down extreme days

**Dependencies:** None (runs independently)
**Complexity:** S

---

## Implementation Order

> **Revised 2026-04-20** after portfolio assessment (`PortfolioAssessment.md`).
> Key change: short framework (BT-3) promoted to Phase 1. Intraday (BT-4/5/6) deferred
> to Phase 3. Rationale: the portfolio has zero short exposure — the single largest
> structural risk. Building the short side is higher-impact than exploring intraday
> strategies with thin post-cost edges.

```
Phase 0 — Infrastructure (Done)
  BT-0-S1  Code restructure                  ✓
  BT-0-S2  _data restructure + paths         ✓
  BT-0-S3  Migrate pkl → Parquet             ✓
  BT-1-S1  Cost model + framework            ✓
  BT-1-S2  Data pipeline inventory           ✓
  BT-1-S3  Earnings data source              ✓

Phase 1 — Short Framework (IMMEDIATE PRIORITY)
  BT-2-S4  PEAD drift window                 (validates both long + short drift simultaneously)
  BT-3-S1  Negative PEAD short               (primary short trigger; mirrors long PM-02)
  BT-3-S3  Accruals factor short screen      (Layer 3 confirmation signal)
  BT-3-S4  Piotroski F-Score filter          (Layer 3 confirmation signal)
  BT-3-S2  Consecutive miss + guidance cut   (higher conviction, lower frequency)

Phase 2 — Swing Long Strategies
  BT-2-S7  Bollinger touch / Type C          (simplest swing backtest; EOD data only)
  BT-2-S6  Overnight reversal                (close-to-open; very simple)
  BT-2-S8  VIX mean reversion                (small sample; fast)
  BT-2-S5  Pre-earnings anticipation         (requires BT-1-S3; builds on Phase 1 PEAD)
  BT-2-S1  EP backtest                       (most complex; uses execution framework)
  BT-2-S2  VCP backtest                      (most complex; base detection algorithm)
  BT-2-S3  EMA Reclaim                       (builds on VCP infrastructure)

Phase 3 — Intraday (deferred until Phase 1+2 validated)
  BT-6-S1  Range Break — read evaluation     ✓ (already done)
  BT-6-S4  Filter signals — consolidate
  BT-4-S1  1BP FTSE
  BT-4-S2  ASRS DAX
  BT-4-S3  SRS DAX
  BT-4-S4  Rule of 4 FOMC
  BT-5-S1  Generic ORB US session
  BT-6-S2  VWAP extrema — filtered bracket
  BT-6-S3  Noon iron butterfly — resume
```

---

## Results Register

Document go/no-go verdicts here as stories complete.

| Strategy | Status | EV/trade | Win rate | Sample | Verdict | Date |
|----------|--------|----------|----------|--------|---------|------|
| Range Break `02_pct` long — IBJP225 5m | BT-6-S1 | +0.045% (net) | 51.8% | 1768 trades | **Go** | 2026-04-20 |
| Range Break `02_pct` long — IBUST100 5m | BT-6-S1 | +0.029% (net) | 54.5% | 1368 trades | **Go** | 2026-04-20 |
| Range Break shorts — all instruments | BT-6-S1 | negative | ~35–38% | 3.6M rows | **No-go** | 2026-04-20 |
| Hougaard 1BP — IBGB100 5m | BT-4-S1 | −2.06 pts | 34.7% | 574 filled | **No-go** | 2026-04-20 |
| Hougaard 1BN — IBGB100 5m | BT-4-S1 | −2.18 pts | 36.4% | 500 filled | **No-go** | 2026-04-20 |
