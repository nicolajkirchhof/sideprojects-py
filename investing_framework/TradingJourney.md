# Trading Journey

Personal motivation, roadmap, goals, learnings, and performance tracking.
Living document — updated quarterly.

---

## Why I Trade

Sold part of my company and started questioning investment advisors. Researched the topic for months. Discovered I'm not likely to outperform the market without experience by picking stocks or sectors. Long-term portfolio went into globally diversified ETFs — that covers the passive side.

What I genuinely like: working with data, chart analysis, and building structured systems around risk management and consistency. That's the real draw — not just making money.

**2026-03 reflection:** After a year of ups and downs, my enthusiasm goes only so far as not to consume me all day and night. I don't want to make this my life. I choose low-risk, systematic approaches with strong index exposure. What I love is the chart work and building stuff around it.

---

## The Roadmap

| Year | Phase | Focus | Status |
|------|-------|-------|--------|
| Y1 | Paper Trade | Gain experience. Fail technically. Learn the craft. | Complete |
| **Y2** | **Profitable** | **Stick to playbook. Be disciplined. Manage risk above all.** | **YOU ARE HERE** |
| Y3 | Pace Market | Profit target set. Improve strategy & management skills. | — |
| Y4 | Outperform | Consistent edge. Scale. Show the money. | — |

### Y2 Focus — Profitable (2026)

Building discipline and profitability — not about making the most money. Executing the system correctly on every trade is the goal.

- Build default strategy for positive drift (DRIFT portfolio — SPY, IWM, QQQ, ESTX, GLD, SLV + commodities)
- Manage risk on speculative momentum trades — 0.5% max per trade
- Improve trade log → review weekly for systematic errors and rule violations
- Learn to scale — partial profits, trailing stops, position sizing by setup type
- Continue with strong index exposure (DRIFT), add momentum swing trades on high-conviction setups only

| Metric | Target | Notes |
|--------|--------|-------|
| Win rate | ≥ 50% | Track per setup type |
| Avg R:R achieved | ≥ 2:1 | Review monthly |
| Max drawdown | < 10% | Hard pause at 10% |
| Trades logged | 100% | No exceptions |

---

## Research Timeline

### Q4 2025 — Momentum *(Completed)*

- [x] Gap evaluation based on price/earnings → mapped to trade mechanics
- [x] Backtest trade mechanics based on available options data
- [x] Add volume, market cap, gap candle, IV/HV to gap evaluation
- [x] Yearly strength: evaluate 3/6/9/12 month behaviour
- [x] Evaluate performance of large intraday gainers

### Q1 2026 — Earnings *(Completed)*

- [x] PEAD: finish gap evaluation, map to trade mechanics
- [x] Pre-earnings drift: evaluate IV expansion behaviour
- [x] Backtest trade mechanics with available options data
- [x] Yearly strength: evaluate 3/6/9/12 month behaviour

**Learning:** PEAD setups require large underlying moves (>10%) and strong volume. Pre-earnings drift most reliable on stocks with ≥3 consecutive beats.

### Q2 2026 — Momentum Swing *(Active)*

- [ ] Find best 5–50 day momentum entry rules for stocks
- [ ] Define clean exit rules — code them into process
- [ ] Build execution checklist for ORB entries
- [ ] First live trades with full rule set — log everything
- [ ] Core PM backtest: validate VRP on commodities (GLD, SLV, TLT, USO, UNG, WEAT)

### Q3 2026 — Refinement *(Template)*

- [ ] Review Q2 trade population — which setup types are working?
- [ ] Add 1–2 new profit mechanisms from research pipeline (PM-06 through PM-11)
- [ ] Scale DRIFT portfolio if Q2 performance supports it
- [ ] [ Add goal ]

### Q4 2026 — Scale *(Template)*

- [ ] Full AAR on Y2 — update annual metrics
- [ ] Define Y3 targets based on actual Y2 results
- [ ] [ Add goal ]
- [ ] [ Add goal ]

---

## After Action Review — Performance Evaluation

Run at weekly, monthly, and quarterly intervals. Assess across all trades AND per setup type.

### Portfolio & Strategy Checklist

| Item | Description |
|------|------------|
| P&L Performance | Total P&L, win rate, profit factor vs benchmark (SPY) |
| Benchmarking | Are you outperforming a passive SPY hold over the same period? |
| ROC / ROIC | Return on capital deployed per strategy and in total |
| MDD / Volatility | Max drawdown, worst run length, volatility of equity curve |
| Rolling P&L Trends | Weekly / monthly rolling curve — improving or degrading? |
| Expected return & components | Break down by setup type A/B/C/D — which is actually working? |
| Risk:Reward | Actual R:R achieved vs planned — are you getting what you expected? |
| Distribution of returns | Skewed toward winners? Or small wins + large losses? |
| Setup-type win rate | A vs B vs C vs D — which generates the most reliable edge right now? |
| Rule compliance rate | Were all non-negotiables followed? Track violations explicitly. |
| Execution notes | Entry slippage, timing errors, pre-set vs manual — learn from each |
| Research questions | What patterns need quantification or backtesting next quarter? |

### Population vs Sample Analysis

| Group | Description |
|-------|------------|
| Population | All trades — full P&L scatter, rolling P&L trend, avg P&L, return distribution |
| Sample A | Type A (EP) only — isolated to Episodic Pivot setups |
| Sample B | Type B (VCP) only — is the VCP edge holding up in current regime? |
| Sample C | Type C (SMA Reclaim) only — most reliable but smallest wins? |
| Sample D | Type D (Breakdown/Short) only — does the short side work for you? |
| DRIFT | Short vol positions — separate from directional swing trades |

### Review Cadence

| Interval | Focus |
|----------|-------|
| Weekly | P&L vs target · execution notes · any rule violations? |
| Monthly | Rolling P&L trend · win rate per setup · R:R · worst run |
| Quarterly | Full AAR (all 12 items) · update goals · add learnings · update playbook |

---

## Y2 Actuals — Update at Year-End

| Metric | Result |
|--------|--------|
| Win Rate | — |
| Avg R:R | — |
| Max DD | — |
| Total Trades | — |
| P&L (Trading) | — |
| P&L (DRIFT) | — |
| P&L (Total) | — |
| Biggest Lesson | — |

---

## Key Learnings

Hard-won lessons from real trades. Each represents a mistake made, recognised, and codified. Add new ones as they occur.

### Y2 Learnings (2026)

1. **Managing too soon creates losses.** If a trade hasn't hit a defined premise (SL, range break), don't touch it. ORCL taught this twice — leaving the initial trade would have prevented losses.

2. **The close is the only decision point.** THE CLOSE COUNTS. Never make decisions based on a forming candle. Wait. The developing candle can look like anything.

3. **Never adapt a failing trade.** Exit immediately when a premise is invalidated. "Adapting" a losing trade is a lie you tell yourself. Close it. Re-enter on a clean setup.

4. **Never sell premium without IVP/IVR filter.** Undefined premium only on large indices. Never naked premium on individual stocks unless comfortable with full assignment at that price.

5. **Always check BP before entry.** ALWAYS look at buying power and maintenance before placing a trade. One forgotten BP check can break a month of good work.

6. **Each trade needs a pre-defined SL.** Max SL as % of premium, DTE, or a specific technical level — before entry. No thinking during the trade. Just execution.

7. **Respect your timeframe.** You are not an intraday trader. Always set up trades that don't need intraday management. Emergency exit = stop-loss, not discretion.

8. **Small underlyings = best learning lab.** Small stocks with liquid options provide the fastest feedback loop for learning trade management. Use them intentionally for skill-building.

9. **Use your broker.** It has better information than Barchart or others at your fingertips when analysing stocks. Don't waste time on inferior tools.

10. **Don't trade underlyings that are too big or that you don't have conviction for.** Size relative to your account matters. Conviction relative to your research matters.

11. **Short premium is great until it isn't.** OK for larger indices. Not OK for everything else without defined risk. The tail risk is real — LTCM proved it.

### Y1 Learnings (2025)

*(Add retrospectively if desired)*

---

## Document Map

This file is the personal wrapper. The strategy details live in dedicated documents:

| Document | Content |
|----------|---------|
| `InvestingPlaybook.md` | LONG-TERM portfolio + DRIFT options income framework |
| `TradingPlaybook.md` | Swing trading system: profit mechanisms, setups, entries, exits |
| `DaytradingPlaybook.md` | Intraday strategies — research and automation candidates |
| `OptionsStrategyReference.md` | All options structures with management rules |
| `BreakoutStrategy.md` | Breakout-specific playbook (presentation format) |
| `_research/by-role/trader.md` | Research library sorted for trading |
| `_research/by-role/investor.md` | Research library sorted for investing |
