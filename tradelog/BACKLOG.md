# Tradelog — Product Backlog

## Completed

- ~~E1: Date Format Settings~~ — shipped
- ~~E2-S1: Opened/Closed columns~~ — shipped
- ~~E3: Configurable Enums (S1-S4)~~ — shipped (incl. Settings UI)
- ~~E4: Trade Creation from Positions (S0-S4)~~ — shipped

---

## E3-S3: Settings UI for managing lookup values

**As a** trader,
**I want** a section in the Settings page where I can add, rename, reorder, and deactivate enum values,
**So that** I can evolve my trading vocabulary without developer intervention.

**Acceptance criteria:**
- [ ] Tab or section per enum type (Strategy, TypeOfTrade, etc.) on the `/settings` page
- [ ] Each section shows a reorderable list (drag-to-reorder or up/down buttons)
- [ ] Inline edit for renaming (click name → text input → save on blur/enter)
- [ ] "Add" button appends a new value at the end
- [ ] "Deactivate" action with confirmation; deactivated values shown in a collapsed "Inactive" section with a "Reactivate" option
- [ ] Changes save immediately (no form-level submit)

**Affected layers:** Frontend (settings component, LookupService already exists)
**Dependencies:** None (API + service already shipped)

---

## E4: Trade Creation from Positions

Streamline the workflow of creating trade theses from automatically-synced IBKR positions. Currently positions arrive via Flex sync without a linked trade — the user must manually create a trade and assign positions separately.

### E4-S0: Fix view/edit mode for all trade form fields

**As a** trader,
**I want** all fields on the trade detail sidebar to be read-only in view mode and editable only after clicking Edit,
**So that** I don't accidentally modify trade data while reviewing.

**Acceptance criteria:**
- [ ] Quill rich text editors are read-only in view mode (`[readOnly]` or Quill's `enable()/disable()`)
- [ ] Checkboxes are visually disabled and not clickable in view mode
- [ ] Leg picker "Assign Positions" button is hidden in view mode
- [ ] "Add Event" button is hidden in view mode
- [ ] "Create Follow-Up" button is visible in view mode (it navigates, doesn't mutate)
- [ ] Verify the same pattern works on all other pages that use view/edit mode (accounts, option-positions, stock-positions, capital, weekly-prep, portfolio)

**Affected layers:** Frontend (trades.html primarily, quick audit of other pages)
**Dependencies:** None

---

### E4-S0b: Sortable columns on Trades table

**As a** trader,
**I want to** sort the trades table by any column,
**So that** I can find trades by date, strategy, symbol, or rating quickly.

**Acceptance criteria:**
- [ ] All columns in the Trades table have `mat-sort-header`
- [ ] Default sort: Date descending (current behavior preserved)
- [ ] Switch from raw signal array to `MatTableDataSource` (enables sort + client-side filter)
- [ ] Include the Status column from E4-S1 once implemented

**Affected layers:** Frontend (trades.ts, trades.html)
**Dependencies:** None

---

### E4-S0c: Symbol autocomplete filter on Trades table

**As a** trader,
**I want to** filter trades by typing a symbol with autocomplete suggestions,
**So that** I can quickly narrow the table to a specific underlying.

**Acceptance criteria:**
- [ ] New text input in the filter bar (alongside Budget/Strategy dropdowns) with `mat-autocomplete`
- [ ] Autocomplete suggests from distinct symbols in the current trades list, sorted alphabetically
- [ ] Partial matching: typing "SP" shows "SPY", "SPXL", etc.
- [ ] Selecting or typing a symbol filters the table client-side
- [ ] Clearing the input removes the filter
- [ ] Filter combines with Budget/Strategy filters (AND logic)

**Affected layers:** Frontend (trades.ts, trades.html)
**Dependencies:** E4-S0b (sorting needs `MatTableDataSource` which also supports filtering)

---

### E4-S1: Trade status field (Open/Closed)

**As a** trader,
**I want** each trade to show whether it's open or closed based on its linked positions,
**So that** I can see at a glance which trades are still active.

**Acceptance criteria:**
- [ ] `Trade` model gains a `Status` string field (nullable, values: `"Open"`, `"Closed"`, null for no linked positions)
- [ ] Status recomputed whenever: a position is assigned/unassigned, a position is closed/reopened, or during Flex sync
- [ ] Logic: trade has ≥1 linked position AND all linked positions are closed → `"Closed"`. Otherwise → `"Open"`. No positions → null
- [ ] For stock positions, "closed" = running total position (`TotalPos`) reaches zero
- [ ] Trades table shows Status as a sortable, filterable column
- [ ] EF migration adds the column; startup backfill computes status for existing trades

**Affected layers:** Data model, API, Frontend
**Dependencies:** None
**Notes:** Recompute logic should be a shared service called from multiple write paths. Needs `/architect` for the recompute trigger design.

---

### E4-S2: Position picker on New Trade

**As a** trader,
**I want to** select unlinked positions when creating a new trade,
**So that** the trade and position assignment happen in a single step.

**Acceptance criteria:**
- [ ] "New Trade" opens the sidebar in create mode with a position picker visible immediately
- [ ] Picker shows unlinked Option Positions and Stock Positions (`TradeId IS NULL`)
- [ ] Symbol filter text input at the top filters both sections
- [ ] Checkboxes for multi-select; mixed option + stock selection supported
- [ ] When positions are selected, form auto-fills: `Symbol` from first selection, `Date` from earliest opened date
- [ ] On save: Trade created AND selected positions linked in one API call
- [ ] If no positions selected, trade is still created (positions can be added later)

**Affected layers:** API (extend `POST /api/trades` to accept position IDs), Frontend
**Dependencies:** None (but benefits from E4-S0b for consistent table UX)
**Notes:** Needs `/architect` for the composite API endpoint design (trade + position IDs in one request).

---

### E4-S4: Auto-update trade status on position changes

**As a** trader,
**I want** my trade's status to update automatically when positions are synced or closed,
**So that** I don't have to manually track which trades are still active.

**Acceptance criteria:**
- [ ] After Flex sync closes/creates positions → recompute status for affected trades
- [ ] After option events (expiry/assignment) → recompute status for affected trades
- [ ] After manual position close in the UI → recompute the linked trade's status
- [ ] After position assign/unassign → recompute status for both old and new trade
- [ ] Recompute is idempotent

**Affected layers:** Backend (FlexSyncService, OptionPositionsController, StockPositionsController)
**Dependencies:** E4-S1

---

### E4-S3: Create Trade from Position pages (deferred)

**As a** trader,
**I want to** select positions on the Option/Stock Positions page and click "Create Trade",
**So that** I can start a trade from the positions I'm looking at.

**Acceptance criteria:**
- [ ] Row checkboxes on Option Positions and Stock Positions tables for multi-select
- [ ] "Create Trade" button appears when unlinked rows are selected
- [ ] Navigates to Trades page with position IDs as query params
- [ ] Trades page opens New Trade form pre-filled with those positions
- [ ] Warning if selected positions span multiple symbols

**Affected layers:** Frontend (option-positions, stock-positions, trades)
**Dependencies:** E4-S2
**Notes:** Lower priority — E4-S2 covers the core workflow from the Trades page. Defer unless E4-S2 feels insufficient.

---

---

## E5: View-Mode Consistency

Fix visual and state inconsistencies in the sidebar's read-only (view) mode.

### E5-S1: Consistent view-mode rendering for trade sidebar

**As a** trader,
**I want** all sidebar elements to look and behave as read-only when I haven't clicked Edit,
**So that** I can trust I'm not accidentally modifying data while reviewing.

**Acceptance criteria:**
- [ ] Quill editors render with a muted/disabled visual state in view mode (reduced opacity, no editable border, cursor: default) — matching how Material inputs look when disabled
- [ ] `pointer-events: none` retained as interaction blocker alongside the visual change
- [ ] `showEventForm` is reset to `false` in `onRowSelect` (prevents stale event form leaking between rows)
- [ ] Event form section guarded with `editMode()` so it cannot render in view mode even if the signal is stale
- [ ] Leg picker for existing trades guarded with `editMode()` for the same reason
- [ ] Same Quill disabled styling applied to weekly-prep page

**Affected layers:** Frontend (`styles.css`, `trades.html`, `trades.ts`, `weekly-prep.html`)
**Dependencies:** None

**Root causes addressed:**
1. `pointer-events-none` blocks interaction but doesn't change visual appearance — Quill editors look "active" (normal border, color) vs Material inputs that look grayed out
2. `showEventForm` not reset in `onRowSelect` — event form can leak from previous edit session into view mode of a new row
3. Leg picker and event form only guarded by their own show-signal, not by `editMode()` — stale state can make them visible in view mode

---

## Implementation Order

```
E5-S1  (view-mode consistency — bug fix)
```
