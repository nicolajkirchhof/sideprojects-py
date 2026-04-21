# Portfolio Playbook

> Three strategies, one thesis: equity markets have structural upward drift.
> Capture it through momentum ignition, variance risk premium, and factor premia.

---

## The System

| Strategy | Playbook | Edge Source | Timeframe | Direction |
|----------|----------|-----------|-----------|-----------|
| **Momentum Swing** | `TradingPlaybook.md` | Institutional friction, PEAD, century momentum | 5–50 days | Long + Short |
| **DRIFT** | `InvestingPlaybook.md` §02 | Variance risk premium, positive drift | 30–360 DTE | Long delta + Neutral |
| **Long-Term ETF** | `InvestingPlaybook.md` §01 | Factor premia (size, value, profitability) | 5+ years | Long only |

**Why three:** Each strategy exploits a different, academically documented edge. They are not correlated by mechanism — momentum persistence, volatility overpricing, and factor premia are independent phenomena. The system earns in different market conditions: swing captures trending moves, DRIFT earns in flat/slightly-down markets, long-term compounds across cycles.

---

## Capital Allocation

| Strategy | BP Share | Risk Budget | When Active |
|----------|---------|-------------|-------------|
| Momentum Swing | Variable (full in GO) | 0.5% per trade, max 5 concurrent | GO regime only (SPY > 50d + 200d SMA) |
| DRIFT | 30–80% of options BP | 2% per trade, 50% BP cap | Always — scales with drawdown |
| Long-Term ETF | Separate account | Add-only contributions | Always — never sell |
| Short Book (Swing) | Max 30% of swing allocation | 0.5% per trade | Especially when SPY < 50d SMA |

---

## Regime Interaction

| Scenario | Swing Long | Swing Short | DRIFT Directional | DRIFT Neutral | Long-Term | Net Effect |
|----------|-----------|-------------|-------------------|---------------|-----------|------------|
| **Bull** (SPY > 20+50 SMA) | Full deployment | Minimal | Earning (drift + VRP) | Earning | Growing | All positive |
| **Flat** | Selective (B/C setups) | Selective | VRP income, no drift | Earning | Flat | DRIFT carries |
| **Correction** (–10 to –20%) | Stopped/idle | Active | Losing, scaling in | Earning/flat | Losing | Short book offsets |
| **Bear** (–30%+) | Idle | Active (strong signals only) | Stopped, LEAPS only | Partially earning | Losing | Maximum stress |

**The short book's role:** It is not a hedge — it is a separate alpha source. In corrections and bear markets, when the long swing book goes dormant, the short book provides active income and negative delta. This addresses the portfolio's largest structural weakness (W1 in the assessment): overwhelming positive delta.

---

## Validated Edges — Backtest Results (April 2026)

All backtests on 2016–2026 data, screener-aligned universe (Price > $5, Vol > 1M).
Full results: `BACKLOG-BACKTESTING.md` Results Register.

### Long Side (ranked by Sharpe)

| Edge | Setup Type | N | 60d Return | Win% | Sharpe | Scanner |
|------|-----------|---|------------|------|--------|---------|
| Strong EP (beat + gap>=10% + top 25%) | Type A | 368 | +31.70% | 84.8 | 0.791 | #9, #11 |
| Green Line Breakout (ATH after consolidation) | — | 13,796 | +7.16% | 67.4 | 0.260 | #1 |
| EMA Reclaim (pullback to 10/20 MA, recovery) | Type C | 60,410 | +5.63% | 61.8 | 0.222 | #3, #10 |
| PEAD Beats (earnings beat, any magnitude) | Type A | 15,699 | +5.43% | 57.5 | 0.208 | #9 |
| Pre-Earnings (T-14, Stage 2, >=3 beats) | PM-03 | 2,243 | +1.99% (14d) | 58.4 | 0.200 | #9 |
| BB Lower Touch (mean reversion in trend) | Type C confirm | 53,400 | +2.25% | 55.1 | 0.115 | #10 |

### Short Side

| Edge | N | 10d Return | 60d Return | Win% (short) | Scanner |
|------|---|------------|------------|--------------|---------|
| Strong Short (miss + gap<=-5% + bottom 25%) | 418 | -15.77% | -12.00% | 76.8 | #13 |
| 1st Earnings Miss (any magnitude) | 3,292 | -2.83% | fades | 60.5 (10d) | #13 |
| RW Breakdown (RS weakness + distribution) | — | — | — | — | #14 |

### Confirmation Signals (Layer 3)

| Signal | Role | Finding |
|--------|------|---------|
| Accruals (Sloan 1996) | Quarterly conviction filter | +3.38%/year long-short spread. Annual signal, not swing. |
| F-Score (Piotroski 2000) | Quality filter | F>=7 upgrades conviction, F<=3 upgrades short conviction |
| Consecutive misses | — | 1st miss is strongest. Consecutive misses show *weaker* drift. Focus on first-miss quality. |

### Rejected

| Edge | N | Finding |
|------|---|---------|
| Overnight Reversal (PM-08) | 33,935 | No edge at any horizon. Removed from active mechanisms. |

---

## Scanner-to-Strategy Map

Each Barchart scanner feeds a specific strategy and mechanism. Full scanner configuration in `BarchartScreeners.md`.

| Scanner | Name | Strategy | Mechanism | Setup Type |
|---------|------|----------|-----------|------------|
| #1 | 52-Week High | Swing Long | PM-01 Breakout | Type B (VCP near highs) |
| #2 | 5-Day Momentum | Swing Long | PM-01 / PM-02 | Type A (EP) |
| #3 | 1-Month Strength | Swing Long | PM-01 | Type B / C |
| #4 | Volume Spike | Swing Long | PM-01 | All types |
| #5 | Trend Seeker | Swing Long | Supplementary | Confirmation |
| #6 | High Put Ratio | Swing Long/Short | PM-04 / PM-11 | Squeeze / Flow |
| #7 | High Call Ratio | Swing Long | PM-04 | Flow confirmation |
| #8 | Intraday RVOL | Swing Long | PM-01 | Type A / C (manual) |
| #9 | PEAD Candidates | Swing Long | PM-02 | Type A (EP on earnings) |
| #10 | TTM Squeeze | Swing Long | PM-01 / PM-09 | Type B / C |
| #11 | EP Gap Scanner | Swing Long | PM-01 / PM-02 | Type A |
| #12 | Short Squeeze | Swing Long | PM-11 | Watchlist |
| #13 | Negative PEAD | **Swing Short** | PM-02 (short) | Type D |
| #14 | RW Breakdown | **Swing Short** | PM-05 (short) | Type D |
| UOA | Unusual Options | Both | PM-04 | Flow confirmation |

---

## Risk Framework

### Per-Trade Limits

| Strategy | Max Risk | Stop Rule | Position Sizing |
|----------|---------|-----------|-----------------|
| Swing Long | 0.5% portfolio | 7% hard max from entry | (0.5% × portfolio) ÷ (entry − stop) |
| Swing Short | 0.5% portfolio | Spread debit = max loss | Debit paid = 0.5% portfolio |
| DRIFT | 2% portfolio | 50Δ hard stop / 200% credit | Per InvestingPlaybook |
| Long-Term | N/A | Never sell | Contribution-based |

### Portfolio-Level Limits

- Total swing exposure: max 5 concurrent positions (2.5% portfolio at risk)
- DRIFT BP deployed: max 50% (30% normal, scales to 80% in deep correction)
- Short book: max 30% of swing allocation (start at 15%, scale after 30 live trades)
- Single underlying: max 30% of any strategy's allocation

### Hedge Framework

Summarised from `InvestingPlaybook.md` §02 Portfolio Hedging:
- 1–2% annual budget for tail protection (far-OTM SPX puts + VIX call spreads)
- Buy in calm (VIX < 15), monetize in stress (VIX > 30)
- Gold allocation (10–20% in Long-Term) provides passive crisis hedge
- DRIFT neutral block (UNG, WEAT, GLD) earns theta while diversifying

---

## Reference Documents

| Document | Purpose |
|----------|---------|
| `TradingPlaybook.md` | Swing trading rules: mechanisms, setup types, entry/exit, short framework |
| `InvestingPlaybook.md` | DRIFT premium selling + Long-Term ETF allocation |
| `BarchartScreeners.md` | Scanner configuration, column views, pipeline integration |
| `OptionsStrategyReference.md` | Options structure selection by IVR and market view |
| `DashboardProfitMechanismMap.md` | PyQtGraph dashboard widget-to-mechanism mapping |
| `BACKLOG-BACKTESTING.md` | Backtest stories, results register, research gaps |
| `DaytradingPlaybook.md` | Intraday automation research (not active) |

---

## Review Cadence

- **Daily:** Run scanner pipeline, review Daily Prep, execute ORB entries
- **Weekly:** DRIFT position review, swing exit signal check, watchlist refresh
- **Monthly:** Performance review, mechanism compliance check
- **Quarterly:** Full portfolio assessment, rebalance Long-Term contributions
- **After 30 short trades:** Evaluate short framework for full-size promotion
