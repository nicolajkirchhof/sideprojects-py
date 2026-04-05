# Tradelog v2 — Trade-Centric Data Model

## Context

The current data model has three disconnected entities (`TradeEntry`, `OptionPosition`, `Trade`) linked only by symbol string. This backlog restructures around a **Trade** as the parent concept — the decision to put on a position — which owns legs (option positions, stock positions) and lifecycle events.

### Decisions Made

| Decision | Choice |
|----------|--------|
| Trade concept | Merge `TradeEntry` into `Trade` (parent). Rename IBKR `Trade` → `StockPosition`. |
| Follow-ups | Chain via `parentTradeId` (self-referencing FK) |
| Assignment | Manual only — Flex sync imports unassigned, user links in UI |
| Trade events | Scale-in, partial profit take, roll, stop/forced close |
| Enums | Update Budget + Strategy to match investing/trading playbooks |

### Target Entity Model

```
Trade (was TradeEntry)
├── id, accountId, symbol, strategy, budget, typeOfTrade
├── directional, thesis (notes), date opened/closed
├── parentTradeId? → Trade (follow-up chain)
│
├─► OptionPosition (FK: tradeId, nullable)
│   ├── short put leg
│   └── kicker call leg
│
├─► StockPosition (was Trade, FK: tradeId, nullable)
│   └── stock fills, futures fills
│
└─► TradeEvent (new)
    ├── type: ScaleIn | ProfitTake | Roll | Stop
    ├── date, notes, pnlImpact?
    └── linked positionId? (which leg was affected)
```

### Naming Changes

| Current | New | Reason |
|---------|-----|--------|
| `Trade` (model) | `StockPosition` | Frees "Trade" for the parent concept |
| `TradeEntry` (model) | `Trade` | This IS the trade — thesis + legs + events |
| `Trades` (table) | `StockPositions` | Database table rename |
| `TradeEntries` (table) | `Trades` | Database table rename |
| `TradesController` | `StockPositionsController` | API route: `/api/stock-positions` |
| `TradeEntriesController` | `TradesController` | API route: `/api/trades` |
| `/trades` (frontend route) | `/stock-positions` | UI navigation — raw IBKR stock/ETF/futures positions |
| `/trade-entries` (frontend route) | `/trades` | UI navigation — this becomes the main trade view |

---

## Epic 1: Rename & Restructure Data Model

> Foundation — must be done first. All other epics depend on this.

### E1-S1: Rename Trade → StockPosition (backend)

**As a** developer,
**I want to** rename the `Trade` model/table to `StockPosition`,
**So that** the name "Trade" is freed for the parent concept.

**Acceptance criteria:**
- [ ] Model class renamed `Trade` → `StockPosition` in `Models/`
- [ ] DB table renamed `Trades` → `StockPositions` via EF migration
- [ ] `TradesController` renamed to `StockPositionsController`, route `/api/stock-positions`
- [ ] `TradesService` (if exists) renamed to `StockPositionsService`
- [ ] `FlexSyncService` updated — references to `Trade` model become `StockPosition`
- [ ] All existing data preserved (migration is a rename, not a recreate)
- [ ] Existing tests pass with updated names

**Affected layers:** Data model, API, Services
**Dependencies:** None
**Notes:** This is a rename-only change. No schema changes to columns.

---

### E1-S2: Rename TradeEntry → Trade (backend)

**As a** developer,
**I want to** rename the `TradeEntry` model/table to `Trade`,
**So that** the parent concept has the correct name.

**Acceptance criteria:**
- [ ] Model class renamed `TradeEntry` → `Trade` in `Models/`
- [ ] DB table renamed `TradeEntries` → `Trades` via EF migration (same migration as S1)
- [ ] `TradeEntriesController` renamed to `TradesController`, route `/api/trades`
- [ ] All existing data preserved
- [ ] Existing tests pass

**Affected layers:** Data model, API, Services
**Dependencies:** E1-S1 (must rename Trade→StockPosition first to avoid name collision)
**Notes:** Combined migration with E1-S1. Order matters: rename Trades→StockPositions first, then TradeEntries→Trades.

---

### E1-S3: Add tradeId FK to OptionPosition and StockPosition

**As a** developer,
**I want to** add a nullable `TradeId` foreign key on `OptionPosition` and `StockPosition`,
**So that** positions can be linked to a parent Trade.

**Acceptance criteria:**
- [ ] `OptionPosition.TradeId` added (int?, nullable FK → Trades.Id)
- [ ] `StockPosition.TradeId` added (int?, nullable FK → Trades.Id)
- [ ] Navigation properties: `Trade.OptionPositions` (collection), `Trade.StockPositions` (collection)
- [ ] EF migration adds columns with `NULL` default (existing rows remain unassigned)
- [ ] API: `GET /api/trades/{id}` includes linked OptionPositions and StockPositions in response
- [ ] API: `GET /api/option-positions?unassigned=true` returns positions with `tradeId == null`
- [ ] API: `GET /api/stock-positions?unassigned=true` returns stock positions with `tradeId == null`

**Affected layers:** Data model, API
**Dependencies:** E1-S1, E1-S2
**Notes:** Nullable FK — positions imported by Flex sync arrive unassigned. User links them manually (Epic 2).

---

### E1-S4: Add parentTradeId for follow-up chains

**As a** developer,
**I want to** add a self-referencing `ParentTradeId` FK on `Trade`,
**So that** DRIFT rolls and follow-up trades can be chained.

**Acceptance criteria:**
- [ ] `Trade.ParentTradeId` added (int?, nullable self-FK → Trades.Id)
- [ ] API: `GET /api/trades/{id}` includes `parentTradeId` and `children` (list of follow-up trade IDs)
- [ ] API: `GET /api/trades/{id}/chain` returns the full chain (root → all descendants)
- [ ] Deleting a trade with children is blocked (400) or cascades `ParentTradeId` to null

**Affected layers:** Data model, API
**Dependencies:** E1-S2

---

### E1-S5: Create TradeEvent entity

**As a** developer,
**I want to** create a `TradeEvent` entity for logging trade lifecycle events,
**So that** scale-ins, profit takes, rolls, and stops are tracked.

**Acceptance criteria:**
- [ ] New model `TradeEvent` with: Id, TradeId (FK), Type (enum), Date, Notes, PnlImpact (decimal?)
- [ ] `TradeEventType` enum: `ScaleIn`, `ProfitTake`, `Roll`, `Stop`
- [ ] Optional `PositionId` (int?) — which OptionPosition or StockPosition was affected
- [ ] API: `GET /api/trades/{id}/events` — list events for a trade
- [ ] API: `POST /api/trades/{id}/events` — add an event
- [ ] API: `DELETE /api/trade-events/{id}` — remove an event

**Affected layers:** Data model, API
**Dependencies:** E1-S2

---

### E1-S6: Update Strategy and Budget enums

**As a** developer,
**I want to** update the `Strategy` and `Budget` enums to match the investing/trading playbooks,
**So that** the dropdown values reflect the actual strategies in use.

**Acceptance criteria:**
- [ ] `Budget` enum updated: `LongTerm`, `Drift`, `Swing`, `Speculative`
- [ ] `Strategy` enum updated: `PositiveDrift`, `RangeBound`, `IVMeanReversion`, `BreakoutMomentum`, `PEAD`, `PreEarnings`, `SectorStrength`, `SectorWeakness`, `Slingshot`, `GreenLineBreakout`
- [ ] EF migration maps old enum values to new (e.g., `RangeBoundCommodities` → `RangeBound`, `Momentum` → `BreakoutMomentum`, `Core` → `Drift`)
- [ ] Frontend enum labels updated
- [ ] Existing data migrated without loss

**Affected layers:** Data model, API, Frontend
**Dependencies:** E1-S2
**Notes:** `Breakout` removed (merged into `BreakoutMomentum`). `PEADS` → `PEAD`. `Core` → `Drift` (most "Core" positions are DRIFT short puts). `Speculative` stays.

---

## Epic 2: Trade Management UI

> The main workflow — create trades, assign legs, log events.

### E2-S1: Rename frontend routes and components

**As a** user,
**I want to** see "Trades" in the nav for the journal view and "Stock Positions" for IBKR positions,
**So that** the naming is consistent with the new model.

**Acceptance criteria:**
- [ ] `/trade-entries` route → `/trades` (journal/thesis view, main trade management)
- [ ] `/trades` route → `/stock-positions` (IBKR stock/ETF/futures positions)
- [ ] Navigation sidebar labels updated
- [ ] All internal links and `router.navigate()` calls updated

**Affected layers:** Frontend
**Dependencies:** E1-S1, E1-S2 (API routes must exist)

---

### E2-S2: Assign positions to a Trade (UI)

**As a** trader,
**I want to** select unassigned OptionPositions and StockPositions and link them to a Trade,
**So that** I can see all legs of a multi-leg structure in one place.

**Acceptance criteria:**
- [ ] Trade detail sidebar shows "Legs" section with currently assigned positions
- [ ] "Add Leg" button opens a picker showing unassigned positions (filtered by symbol)
- [ ] User can select one or more positions and click "Assign"
- [ ] Assigned positions show in the Trade detail with key fields (strike, expiry, pos, P/L)
- [ ] User can unassign a position (sets `tradeId = null`)
- [ ] API calls: `PATCH /api/option-positions/{id}` with `{ tradeId: X }` and `PATCH /api/stock-positions/{id}` with `{ tradeId: X }`

**Affected layers:** Frontend, API (PATCH endpoint)
**Dependencies:** E1-S3, E2-S1

---

### E2-S3: Create follow-up trade

**As a** trader,
**I want to** create a follow-up trade linked to an existing trade,
**So that** I can track rolling DRIFT positions as a chain.

**Acceptance criteria:**
- [ ] Trade detail shows "Create Follow-up" button
- [ ] Follow-up pre-fills: symbol, strategy, budget from parent
- [ ] `parentTradeId` set automatically
- [ ] Trade list shows chain indicator (e.g., indent or "→" icon for follow-ups)
- [ ] Trade detail shows parent link and child links

**Affected layers:** Frontend
**Dependencies:** E1-S4, E2-S1

---

### E2-S4: Log trade events

**As a** trader,
**I want to** log scale-ins, profit takes, rolls, and stops on a Trade,
**So that** I have a management history for each position.

**Acceptance criteria:**
- [ ] Trade detail sidebar shows "Events" timeline
- [ ] "Add Event" button with: type dropdown, date, notes, optional P/L impact
- [ ] Events displayed chronologically with type badge
- [ ] Delete event (with confirm)
- [ ] Roll events can link to the follow-up Trade (if one exists)

**Affected layers:** Frontend
**Dependencies:** E1-S5, E2-S1

---

## Epic 3: Dashboard & Analytics Updates

### E3-S1: Update dashboard to show Trade-level summaries

**As a** trader,
**I want to** see trades (not raw positions) as the primary dashboard view,
**So that** I see my decision-level P/L, not individual leg P/L.

**Acceptance criteria:**
- [ ] Dashboard groups positions by Trade (where assigned)
- [ ] Unassigned positions still show individually
- [ ] Trade-level P/L = sum of all leg P/Ls
- [ ] Trade-level Greeks = sum of all leg Greeks
- [ ] Clicking a trade navigates to `/trades/{id}`

**Affected layers:** Frontend, API (new endpoint or computed fields)
**Dependencies:** E2-S2

---

### E3-S2: Analytics by strategy chain

**As a** trader,
**I want to** see analytics aggregated by strategy and by trade chain,
**So that** I can evaluate which strategies and which rolling campaigns are profitable.

**Acceptance criteria:**
- [ ] Analytics page groups by Strategy with win rate, avg P/L, total P/L
- [ ] New "Chains" view shows each parentTradeId chain with cumulative P/L
- [ ] DRIFT chains show total premium collected vs losses across all rolls

**Affected layers:** Frontend, API
**Dependencies:** E1-S4, E3-S1

---

## Implementation Order

```
Phase 1 — Rename (E1-S1, E1-S2)           ← Architect first
Phase 2 — FK links (E1-S3, E1-S4, E1-S5)  ← Data model
Phase 3 — Enums (E1-S6)                    ← Schema + migration
Phase 4 — Frontend rename (E2-S1)          ← Unblocks UI work
Phase 5 — Trade management (E2-S2, E2-S3, E2-S4) ← Core workflow
Phase 6 — Dashboard (E3-S1, E3-S2)         ← Polish
```

**Minimum viable workflow:** Phases 1–5. You can start using the trade-centric model after Phase 5 is complete. Phase 6 is polish.

---

## Open Questions for Architect

1. **Migration strategy:** Single large migration or one per story? (Recommend: one migration for all of Epic 1, since the renames and FK adds are interdependent)
2. **FlexSync integration:** When Flex sync creates new OptionPositions, should it attempt to find an existing open Trade on the same symbol and leave a hint? Or purely unassigned?
3. **Trade status:** Should Trade have an explicit `Status` enum (Open/Closed/Rolled) or derive it from whether all legs are closed?
