# Intraday Execution Engine — Backlog

Automated live execution of intraday PM strategies via IBKR Gateway.
Runs as a long-lived Python process; fires bracket orders at bar closes,
manages trailing stops natively via IBKR, and journals all actions to JSONL.

**Module:** `finance/execution/`
**Platform:** Python async process (ib_async + APScheduler)
**Dependencies:** IBKR Gateway running locally (paper: port 4002, live: port 4001)

---

## Architecture Decisions (settled)

- ib_async async API (`asyncio.run` + `APScheduler.AsyncIOScheduler`)
- Entry: OCA stop orders (BUY STOP + SELL STOP in same OCA group)
- Stop: native IBKR trailing stop (`orderType='TRAIL'`, `auxPrice=trail_pts`) placed on fill
- Fill callback cancels unfilled OCA leg, places trailing stop
- ATR trail: `atr * ATR_TRAIL_FACTOR` (SRS, OCO strategies); `bar_range + 2 * ENTRY_OFFSET_PTS` (ASRS strategy)
- Conflict resolution: same direction → skip; opposite direction → flip (cancel + reverse)
- Journal: append-only JSONL at `finance/execution/logs/YYYY-MM-DD.jsonl`
- Tradelog sync: Flex sync handles fills; JSONL holds automation metadata keyed by ExecutionId
- Paper/live: single config flag (`mode: paper | live`)
- Constants: class-level per strategy (no shared config for strategy tuning values)
- Daily loss: fill-based accumulation per strategy; pre-trade gate blocks order if `trail_pts × multiplier ≥ max_loss`; strategy halted (not engine) on breach
- Loss limits: EUR for FDXS (€1/pt), USD for MNQ ($2/pt); separate limits in config
- Initial loss limits: `max_daily_loss_eur: 100`, `max_daily_loss_usd: 100`
- Journal thread-safety: plain synchronous append — single asyncio event loop serializes naturally
- Fill routing: `PositionTracker` maps `order_id → strategy_id` at placement; global `execDetailsEvent` dispatches via lookup
- Partial fills: ignore `PartiallyFilled` events; log as `partial_fill_ignored`
- Reconnect: 3 attempts, 5s→15s→45s backoff; log `missed_signal` if signal fires during downtime; reconcile on success
- ATR fetch failure: abort engine start, send Telegram alert before exit
- Alerting: Telegram bot (`bot_token` + `chat_id` in config); fires on fill, halt, startup/shutdown, missed signal, ATR failure, engine error
- Integration tests: deferred to Phase 2

---

## Signal Timing

| Strategy | Instrument | Bar closes | Scheduler fires |
|----------|------------|------------|--------------------|
| Hougaard SRS | FDXS | 09:45 Frankfurt | 09:45:30 Frankfurt |
| Hougaard ASRS | MNQ | 09:50 ET | 09:50:30 ET |
| Hougaard SRS | MNQ | 10:00 ET | 10:00:30 ET |
| OCO Opening Bar 30m | MNQ | 10:00 ET | 10:00:30 ET |

---

## Epic 1 — Core Infrastructure

### E1-S1: Config module

Status: Pending

**As a** developer running the execution engine,
**I want** a YAML-based config with typed dataclasses,
**So that** switching between paper and live requires changing one line.

**Acceptance criteria:**
- [ ] `EngineConfig`, `IbkrConfig`, `RiskConfig`, `TelegramConfig` dataclasses
- [ ] `load_config(path)` validates required fields and raises on invalid mode
- [ ] Default `config.yaml`: paper mode, `client_id=10`, `max_daily_loss_eur=100`, `max_daily_loss_usd=100`
- [ ] `mode: paper` selects `paper_port`; `mode: live` selects `live_port`
- [ ] `TelegramConfig`: `bot_token`, `chat_id` fields
- [ ] `RiskConfig` includes both `max_daily_loss_eur` and `max_daily_loss_usd`

**Affected layers:** Config
**Dependencies:** None

---

### E1-S2: Journal

Status: Pending

**As a** trader reviewing automation performance,
**I want** every engine action logged to a JSONL file,
**So that** I can join it with Flex import data for strategy-level P&L analysis.

**Acceptance criteria:**
- [ ] Writes to `finance/execution/logs/YYYY-MM-DD.jsonl`, one JSON object per line
- [ ] Entry types: `signal`, `placed`, `fill`, `cancel`, `skip`, `flip`, `eod_flatten`, `partial_fill_ignored`, `missed_signal`, `strategy_halted`, `pre_trade_gate_blocked`
- [ ] Every entry includes: `ts` (ISO with TZ), `strategy_id`, `symbol`, `direction`
- [ ] Fill entries include: `fill_price`, `trail_pts`, `atr`, `ibkr_execution_id`
- [ ] `pre_trade_gate_blocked` entries include: `trail_pts`, `max_loss`, `multiplier`, `currency`
- [ ] File rotates daily (new file per trading day)
- [ ] Thread-safety: plain synchronous append within the asyncio event loop (cooperative scheduling eliminates contention)

**Affected layers:** Persistence
**Dependencies:** None

---

### E1-S3: Position tracker

Status: Pending

**As a** developer,
**I want** an in-memory record of open positions and pending orders per instrument,
**So that** conflict resolution and EOD flattening work correctly even after reconnects.

**Acceptance criteria:**
- [ ] `InstrumentState` dataclass: `long_order_id`, `short_order_id`, `direction`, `fill_price`, `trail_order_id`, `strategy_id`
- [ ] `PositionTracker.get(symbol)` → `InstrumentState | None`
- [ ] `set_pending`, `on_fill`, `on_close` transition methods
- [ ] `order_id_to_strategy(order_id)` → `str` — maps any order ID to its `strategy_id` for fill routing
- [ ] `reconcile(ibkr_positions, ibkr_orders)` rebuilds state from live IBKR data on reconnect
- [ ] All state changes logged to journal

**Affected layers:** State
**Dependencies:** E1-S2

---

### E1-S4: Broker wrapper

Status: Pending

**As a** developer,
**I want** a single `Broker` class that abstracts all ib_async calls,
**So that** strategies and the engine never interact with ib_async directly.

**Acceptance criteria:**
- [ ] `connect(config)` — async connect; selects port based on `mode`
- [ ] `qualify_contracts(specs)` → `dict[str, Contract]`
- [ ] `fetch_daily_bars(contract, n_days)` → list of BarData (for ATR)
- [ ] `fetch_intraday_bars(contract, bar_size, session_open, session_tz)` → session bars for today
- [ ] `place_oca_entry(contract, spec)` → `(Trade, Trade)` — OCA group, `transmit=True`
- [ ] `place_trailing_stop(contract, direction, trail_pts, qty)` → `Trade`
- [ ] `cancel_order(order_id)`, `flatten_position(contract)`
- [ ] `get_open_positions()`, `get_open_orders()` — for reconcile
- [ ] Raises `BrokerError` (not ib_async internals) on failure

**Affected layers:** Broker
**Dependencies:** E1-S1

---

## Epic 2 — Strategies

### E2-S1: Strategy base

Status: Pending

**Acceptance criteria:**
- [ ] `SignalBar` dataclass: `open, high, low, close, bar_time`
- [ ] `OrderSpec` dataclass: `symbol, direction, entry_long, entry_short, trail_pts, qty, strategy_id, signal_bar, atr`
- [ ] `Strategy` ABC: `strategy_id`, `symbol`, `session_tz`, `signal_fire_time`, `eod_time`, `currency` (`EUR|USD`), `multiplier` (points→currency), `max_daily_loss`, abstract `compute_signal(bars, atr) → OrderSpec | None`
- [ ] `pre_trade_gate(trail_pts) → bool` on base class: returns `False` (and journals `pre_trade_gate_blocked`) if `trail_pts × multiplier ≥ max_daily_loss`
- [ ] `DailyPnLTracker` on base class: `record_fill(entry, exit, direction)` accumulates realized PnL; `is_halted() → bool`
- [ ] `compute_14d_atr(daily_bars) → float` shared utility

**Affected layers:** Strategy
**Dependencies:** E1-S4

---

### E2-S2: Hougaard SRS strategy

Status: Pending

**Signal:** Nth 15-min bar (bar[2] = 09:30 Frankfurt for FDXS; bar[1] = 09:45 ET for MNQ)
**Stop:** `atr * ATR_TRAIL_FACTOR`

**Acceptance criteria:**
- [ ] `HougaardSrsStrategy(symbol, session_tz, session_open, bar_index, signal_fire_time, eod_time)`
- [ ] Class-level constants: `ENTRY_OFFSET_PTS`, `ATR_TRAIL_FACTOR` (set from backtest results — placeholder until calibrated)
- [ ] `compute_signal` returns `None` if bar index unavailable
- [ ] Entry: `bar.high + ENTRY_OFFSET_PTS` (long), `bar.low - ENTRY_OFFSET_PTS` (short)
- [ ] Instantiated for FDXS (`bar_index=2`, Frankfurt, `currency=EUR`, `multiplier=1`) and MNQ (`bar_index=1`, ET, `currency=USD`, `multiplier=2`)

**Dependencies:** E2-S1

---

### E2-S3: Hougaard ASRS strategy

Status: Pending

**Signal:** 4th 5-min bar (bar[3] = 09:45 ET for MNQ)
**Stop:** `bar_range + 2 * ENTRY_OFFSET_PTS`

**Acceptance criteria:**
- [ ] Class-level constants: `ENTRY_OFFSET_PTS`, `MIN_BAR_RANGE_PTS` (set from backtest results — placeholder until calibrated)
- [ ] `compute_signal` returns `None` if bar range < `MIN_BAR_RANGE_PTS`; journals `pre_trade_gate_blocked` reason
- [ ] `trail_pts = signal_bar.high - signal_bar.low + 2 * ENTRY_OFFSET_PTS`
- [ ] Instantiated for MNQ (`currency=USD`, `multiplier=2`)

**Dependencies:** E2-S1

---

### E2-S4: OCO Opening Bar 30m strategy

Status: Pending

**Signal:** 1st 30-min bar (bar[0] = 09:30 ET for MNQ, closes at 10:00)
**Stop:** `atr * ATR_TRAIL_FACTOR`

**Acceptance criteria:**
- [ ] Fires at 10:00:30 ET
- [ ] Class-level constants: `ENTRY_OFFSET_PTS`, `ATR_TRAIL_FACTOR` — consistent with E2-S2 (same values unless backtest says otherwise)
- [ ] `compute_signal` returns `None` if bar unavailable
- [ ] Instantiated for MNQ (`currency=USD`, `multiplier=2`)

**Dependencies:** E2-S1

---

## Epic 3 — Engine & Entry Point

### E3-S1: Engine

Status: Pending

**Acceptance criteria:**
- [ ] `Engine(config)` — holds broker, tracker, journal, alerter, strategies, daily ATR cache, daily PnL state
- [ ] `start()`: connect → qualify contracts → fetch ATR (abort + alert on failure) → reconcile → schedule jobs → run loop
- [ ] Pre-market job (08:45 Frankfurt / 09:00 ET): refresh ATR cache via `reqHistoricalDataAsync`
- [ ] Signal job per strategy:
  - Check `strategy.is_halted()` → skip if halted
  - Fetch bars → `compute_signal` → pre-trade gate → conflict check → place OCA → register fill callbacks
  - If disconnected at signal time: log `missed_signal`, alert
- [ ] Fill callback (via `execDetailsEvent` → `tracker.order_id_to_strategy(order_id)` dispatch):
  - Ignore `PartiallyFilled` (log `partial_fill_ignored`)
  - On `Filled`: cancel other OCA leg → place trailing stop → update tracker → journal `fill` → alert
  - On exit fill: `strategy.daily_pnl.record_fill(...)` → check `is_halted()` → alert if halted
- [ ] EOD job per instrument: `get_open_positions()` + `get_open_orders()` → cancel → flatten → journal `eod_flatten`
- [ ] `_on_reconnect()`: 3 attempts, 5s→15s→45s backoff; on success → `tracker.reconcile()`; on exhaustion → alert + shutdown
- [ ] Conflict logic: same direction → skip + journal `skip`; opposite → flatten + new bracket + journal `flip`
- [ ] `apscheduler>=3.10` added to `pyproject.toml`

**Affected layers:** Engine
**Dependencies:** E1-S1 → E1-S4, E2-S1 → E2-S4, E5-S1

---

### E3-S2: Entry point

Status: Pending

**Acceptance criteria:**
- [ ] `python -m finance.execution` starts the engine
- [ ] `--config PATH` overrides default config path
- [ ] `--mode paper|live` overrides config mode
- [ ] Logs startup banner with mode, instruments, and scheduled signal times

**Dependencies:** E3-S1

---

## Epic 4 — Tests

### E4-S1: Unit tests

Status: Pending

**Acceptance criteria:**
- [ ] `test_execution_config.py` — YAML loading, mode validation, missing fields raise, both loss limit fields present
- [ ] `test_execution_journal.py` — all entry types written correctly, required fields present, file rotates by day
- [ ] `test_execution_tracker.py` — state transitions: pending → fill → close; `order_id_to_strategy` lookup; reconcile rebuilds state
- [ ] `test_execution_strategies.py` — `compute_signal` with synthetic bars: correct entry/trail prices; `None` on insufficient bars; ASRS `None` on narrow bar; pre-trade gate blocks when `trail_pts × multiplier ≥ max_loss`; `DailyPnLTracker` halts after breach

**Dependencies:** E1-S1 → E1-S3, E2-S1 → E2-S4
**Note:** Integration/smoke tests and paper trading validation deferred to Phase 2.

---

## Epic 5 — Alerting

### E5-S1: Telegram alerter

Status: Pending

**As a** trader running the engine live,
**I want** real-time Telegram notifications for fills, halts, and engine errors,
**So that** I can monitor execution without watching logs.

**Acceptance criteria:**
- [ ] `TelegramAlerter(bot_token, chat_id)` with async `send(message: str)`
- [ ] Fires on: engine startup, engine shutdown, fill (entry + exit), strategy halted, pre-trade gate blocked, missed signal, ATR fetch failure (before abort), reconnect failure, unhandled engine error
- [ ] Message format: `[PAPER|LIVE] <emoji> <event>` — emoji: `filled=`, `halted=`, `error=`, `startup=`, `shutdown=`
- [ ] Graceful degradation: if Telegram send fails, log the error but do not crash the engine
- [ ] `bot_token` and `chat_id` loaded from `TelegramConfig`

**Affected layers:** Alerting
**Dependencies:** E1-S1
