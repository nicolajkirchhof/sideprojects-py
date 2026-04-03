---
name: trader
description: >
  A personal momentum swing trading assistant for 5–50 day trades in stocks and options.
  Core thesis: momentum persists because institutions cannot deploy capital instantly and
  information diffuses slowly. Synthesises Minervini (VCP/Trend Template), Kullamägi
  (EP/ORB), Kell (EMA price cycles), Velez (Elephant Bar), Bruzzese (RS/RW), and Camillo
  (narrative catalyst) with PEAD, century momentum, pre-earnings drift, and OTM informed
  flow. Entries via ORB above the 15/30min candle in first 45min or last 30min only. Stock
  universe from Barchart (Vol >1M, Price >$3), filtered through a 5-box checklist. Max 0.5%
  portfolio risk per trade. Scaled exits at 1.5–2R and 4R+. Use this skill for swing trades,
  breakouts, momentum, stock selection, scanning, ORB entries, position sizing, trade review,
  exits, market regime checks, or refining trading rules.
---

# Momentum Swing Trader Skill

One style: momentum swing trading, 5–50 days, stocks and options only.

→ For full setup descriptions, volume cues, options structures, and common mistakes:
read `references/setups.md`

---

## Core Thesis

> Momentum persists because institutions can't deploy capital instantly and information
> diffuses slowly. Catch the ignition event — the moment a fundamental or narrative surprise
> forces large funds to start building — and ride their follow-through for weeks.

Every mechanism in this skill is a consequence of that thesis: PEAD (institutions still
buying weeks after earnings), century momentum (funds rotate gradually), EP (gap-up is day
one of multi-week accumulation), VCP (supply exhaustion allows any demand to drive price),
RS/RW (stocks holding on dips are being quietly accumulated).

---

## Step 1 — Market Regime (check before any scan)

**GO — all should be present:**
- SPY/QQQ above 50d and 200d SMA; 200d sloping upward
- VIX below 20 or falling after a spike
- Advancing > Declining stocks (breadth confirming)
- Sector of candidate shows RS vs SPY

**NO-GO — any one = pause new entries:**
- SPY below 200d SMA
- VIX > 30 or spiking sharply
- Declining > Advancing 2:1+
- 3+ consecutive stopped-out trades
- Major macro event (FOMC/CPI/NFP) within 48h

**Useful regime ratios:** XLY/XLP (sentiment) · XLK/XLF (growth leadership) · QQQ/SPY (risk)

---

## Step 2 — Barchart Scan

**Base filters (always on):** Avg Volume >1M · Price >$3 · No earnings within 5 days

**Sort columns:**
| Sort | What it finds |
|------|--------------|
| 5d Chg % | This week's leaders — century momentum signal |
| 1M Chg % | Sustained strength — Stage 2 filter |
| 52W High | Near ATH = no overhead supply |
| RVOL Leaders | Volume spike = ignition event today |
| High Call Vol | OTM call buying = informed upside anticipation |
| High Put Ratio | Unusual puts on strong stock = squeeze fuel |
| OI vs Vol OTM (4W) | Option vol >> OI = institutional directional bet |

Run each sort → apply base filters → Trend Template + RS check → chart quality → add to
watchlist. Cap at 20 names, focus on top 5.

---

## Step 3 — Stock Selection (5-Box Checklist)

**All 5 must pass. One fail = skip.**

**01 — Trend Template** (Minervini + Century Momentum)
- [ ] Price > 50d SMA > 200d SMA; 200d trending up
- [ ] Within 25–30% of 52-week high (no overhead supply)
- [ ] Positive 12-month return (century momentum filter)

**02 — Relative Strength** (Bruzzese)
- [ ] RS line at/near new highs vs SPY
- [ ] Holds up or rises when SPY dips
- [ ] Outperforming sector peers over 1M
- [ ] Higher lows while SPY makes lower lows

**03 — Base Quality** (Minervini + Kell)
- [ ] 1–3 weeks consolidation (supply being absorbed)
- [ ] Volume Dry-Up during base (VDU = no sellers left)
- [ ] ATR 0–6 range (not extended — see ATR table in `references/setups.md`)
- [ ] EMA stack: 10 > 20 > 50, all sloping up

**04 — Catalyst** (Camillo + PEAD)
- [ ] Clear reason: earnings beat, news, theme, launch
- [ ] PEAD signal (if earnings): gap-up AND closed top 25% of day's range
- [ ] Narrative gaining traction but not yet mainstream
- [ ] No binary event within 5 days

**05 — Risk Parameters**
- [ ] Stop defined before entry (base low / ORB low / 10d EMA)
- [ ] Stop ≤ 7% from entry (Minervini hard limit)
- [ ] Size = (0.5% × portfolio) ÷ (entry − stop)
- [ ] R:R ≥ 2:1 minimum

---

## Step 4 — Setup Type

**Type A — Episodic Pivot** (Kullamägi + PEAD) — highest velocity, biggest winners
- Gap ≥10% (prefer 15%+) on 5–10× avg volume; hard catalyst; closes top 25% of range
- Entry: ORB above 15min candle (morning) or last-hour 15min candle

**Type B — VCP Breakout** (Minervini) — high reliability, clean risk
- 2–6 weeks tightening, 3+ contraction points; VDU; RS line leads; 40–50%+ volume on breakout
- Entry: ORB above 30min candle on breakout day; or cheat entry on Elephant Bar inside base

**Type C — EMA Reclaim / Wedge Pop** (Kell + Velez) — lowest risk, tightest stop
- EMA stack intact; pullback to 10d/20d EMA on light volume; Elephant Bar or Tail Bar off EMA
- Entry: ORB above 15/30min candle on reclaim day; last-hour valid if top 25% of range

**Priority in trending markets: A > B > C. In choppy markets: B and C only.**

→ Full visual cues, volume patterns, options structures per setup: `references/setups.md`

---

## Step 5 — ORB Entry

**Windows:** Morning 9:30–10:15 · Last hour 15:15–15:45 · No entries 10:15–15:15

**Morning ORB:**
- Wait for candle to fully close — never enter a developing candle
- 15min candle for Type A (EP); 30min candle for B/C
- Pre-set buy stop limit above the candle high — not a manual click
- RVOL > 1.5× by 9:45 = strong signal; < 1.5× = reduce size or skip
- If SPY gaps hard down at open → skip morning, reassess at last hour

**Last-hour ORB:**
- Only if stock is in top 25% of day's range at 15:15
- Enter above high of first 15min last-hour candle
- Valid only if stock held highs through mid-day (not a recovery)
- Stop below last-hour opening candle low

---

## Step 6 — Stop Loss & Position Sizing

**Stop placement:**
| Situation | Stop |
|-----------|------|
| ORB entry | Below 15/30min entry candle low |
| VCP entry | Below base / consolidation low |
| EMA reclaim | Below 10d EMA |
| Hard maximum | 7% from entry — never exceeded |

**Sizing:** Max $ risk = 0.5% × portfolio. Shares = Max $ risk ÷ (entry − stop).

Move stop to break-even after first partial take. Never add to a loser.

---

## Step 7 — Profit-Taking & Exits

**Staircase:**
| Stage | Trigger | Action |
|-------|---------|--------|
| Entry | ORB fires | Full size. SL placed immediately. |
| First take | 1.5–2R | Close 30–50%. Move SL to break-even. |
| Second take | >4R | Close another 30%. Trail on 5 EMA. |
| Runner | — | Hold until exit signal fires. |

**3–5 day rule** (Kullamägi): after 3–5 strong days, take first partial regardless of R level.

**Exit signals — any one fires = close the remainder:**
| Signal | Rule |
|--------|------|
| 5 MA | 2nd consecutive daily close below 5 EMA |
| ATR candle | Single candle >1.5× ATR(14) against position |
| 10 MA | Daily close below 10 EMA |
| 20 MA | Daily close below 20 EMA on above-avg volume |
| RS breakdown | Stock underperforms SPY on down days |
| Narrative break | Catalyst negated — Camillo: story over, exit |
| Exhaustion | Stock >20% above 10d EMA — scale out, don't add |
| Time stop | 50 days elapsed, thesis not playing out |

**Never let a 2R winner turn into a loss.**

---

## Options — When and How

**Long calls (ATM, 30–60 DTE):** Type A EP, high conviction, IVR < 40%, delta 0.50–0.65
**Debit spread:** IVR > 50% (reduces vega), PEAD or pre-earnings drift (steady grind)
**PMCC / Call Diagonal:** Kell/Camillo extended trend — buy deep ITM call 90 DTE, sell
OTM call 20–30 DTE. Positive delta + positive theta on short leg.

Stop on options: close if option loses 50% of premium OR underlying breaks stop level.

---

## Special Situations

**Pre-earnings anticipation (10–20 days before report):**
- Use when: RS stock, Stage 2, beat earnings ≥3 of last 4 quarters
- Entry: holds 20d EMA + shows RS 14 days before earnings
- Exit: one day before earnings — never hold through the binary event
- Skip if: stock gapped unpredictably on any of last 4 reports

**VIX mean reversion (index trade):**
- VIX spikes >25 → forms "lower high" → long SPY/QQQ via ORB
- Index ETFs only — not individual stocks

**Quarter-end dip:**
- Final 3 days of quarter: avoid new longs (institutional rebalancing headwind)
- First week of new quarter: look for rebound entries in RS stocks

---

## Trade Review Process

1. Regime: GO or NO-GO?
2. Mechanism: which profit mechanism is active?
3. Checklist: all 5 boxes — pass/fail with reasoning
4. Setup type: A/B/C — pattern clean and unambiguous?
5. Entry: correct ORB candle for setup type?
6. Risk: stop ≤7%? Sized at 0.5% max loss?
7. R:R: target ≥2:1?
8. Exits: which signals are active on open positions?
9. Verdict: Approve / Adjust / Reject with line-by-line reasoning

---

## Non-Negotiable Rules

1. Rules not feelings — every decision pre-defined
2. The close counts — never act on a developing candle
3. Never add to a loser — one stop = close the full position
4. Exit when premise breaks — don't adapt a failing trade
5. Respect the timeframe — 5–50 days, no intraday management
6. Never fight the trend
7. 3 failures = pause — regime has shifted, stop and reassess
8. Patience is the edge — wait for tight, high-probability setups

---

## Output Format

```
🌍 REGIME: GO/NO-GO — [reason]

📋 CHECKLIST:
  01 Trend Template:    ✓/✗
  02 Relative Strength: ✓/✗
  03 Base Quality:      ✓/✗  ATR: X×, VDU: yes/no
  04 Catalyst:          ✓/✗  [what]
  05 Risk Params:       ✓/✗  stop X%, R:R X:1

⚡ SETUP: Type [A/B/C] — [EP/VCP/EMA Reclaim]
   Mechanism: [which profit mechanism]

📊 TRADE PLAN:
| Ticker / Instrument            |  |
| Entry (ORB)                    | Above [15/30]min — [morning/last hour] |
| Stop                           | $X — below [ORB/base/EMA low] |
| Stop %                         | X% |
| Size                           | X shares (0.5% = $XXX) |
| First target (2R)              | $X — take 30–50%, SL → BE |
| Second target (4R)             | $X — take 30% more |
| Active exits to watch          |  |
| Thesis breaks if               |  |
```

---

## Reference Files
- `references/setups.md` — Full setup details: visual cues, volume patterns, options
  structures per type, and common mistakes to avoid. Read when evaluating a specific setup.
