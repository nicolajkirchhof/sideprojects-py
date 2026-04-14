# Tradelog — Product Backlog

## Completed

- ~~E1: Date Format Settings~~ — shipped
- ~~E2-S1: Opened/Closed columns~~ — shipped
- ~~E3: Configurable Enums (S1-S4)~~ — shipped (incl. Settings UI)
- ~~E4: Trade Creation from Positions (S0-S4)~~ — shipped
- ~~E5-S1: View-Mode Consistency~~ — shipped
- ~~E6: Position Notes + Remove Trade Events (S1-S4)~~ — shipped
- ~~E7: Strategy Library (S1-S3)~~ — shipped

---

## E7-S4: Link strategies to documents

**As a** trader,
**I want** to navigate from a trade's strategy to its playbook document,
**So that** I can quickly check the rules.

**Acceptance criteria:**
- [ ] Trades table: Strategy column shows a link icon if the strategy has a document
- [ ] Clicking navigates to `/strategy-library?doc={id}`
- [ ] Strategy Library reads query param and auto-selects the document
- [ ] LookupValues API response includes `documentId` (nullable)

**Affected layers:** Frontend (trades, strategy-library), API (extend lookups)
**Dependencies:** E7-S2
**Status:** Pending

---

## E8: Trades Table Enhancements + View-Mode Fix

### E8-S1: Trade P&L column in trades table

**As a** trader,
**I want** to see the aggregated P&L for each trade in the trades table,
**So that** I can scan which trades are profitable without opening each one.

**Acceptance criteria:**
- [ ] New `pnl` column in the trades table between `managementRating` and `status`
- [ ] P&L = sum of all linked option position P&L (realized + unrealized) + stock P&L
- [ ] Green for positive, red for negative, muted for zero/null
- [ ] Trades with no linked positions show `—`
- [ ] Sortable via `mat-sort-header`
- [ ] Computed on backend in GetAll via `TradeListItemDto`

**Affected layers:** API (new DTO, TradesController), Frontend (trades table)
**Dependencies:** None
**Status:** Pending

**Implementation:**
- [x] Create `Dtos/TradeListItemDto.cs`
- [x] Update `TradesController.GetAll` to return `TradeListItemDto[]` with P&L computation
- [x] Update `trades.service.ts` Trade interface to include `pnl`
- [x] Add `'pnl'` to `displayedColumns` in `trades.ts`
- [x] Add P&L column template in `trades.html`

---

### E8-S2: Status filter on trades table

**As a** trader,
**I want** to filter the trades table by Open/Closed status,
**So that** I can focus on active trades or review closed ones.

**Acceptance criteria:**
- [ ] `mat-button-toggle-group` (Open / Closed / All) in the filter bar
- [ ] Default: Open
- [ ] Client-side filter on `MatTableDataSource`
- [ ] Combines with existing filters (AND logic)
- [ ] Null status shown only when "All" selected

**Affected layers:** Frontend (trades.ts, trades.html)
**Dependencies:** None
**Status:** Pending

**Implementation:**
- [x] Add `filterStatus` signal to `trades.ts`
- [x] Extend `filterPredicate` with status check
- [x] Add toggle group to `trades.html` filter bar

---

### E8-S3: Fix rich text field rendering in view mode

**As a** trader,
**I want** rich text content visible in view mode without clicking Edit,
**So that** I can read my trade thesis while reviewing.

**Acceptance criteria:**
- [ ] View mode: render HTML via `[innerHTML]` instead of Quill
- [ ] Edit mode: Quill renders as before
- [ ] Empty fields show nothing in view mode
- [ ] Applies to 5 trades fields + 8 weekly-prep fields

**Affected layers:** Frontend (trades.html, trades.ts, weekly-prep.html, weekly-prep.ts, styles.css)
**Dependencies:** None
**Status:** Pending

**Implementation:**
- [x] Add `.ql-view-content` CSS class for innerHTML rendering
- [x] Replace each Quill block with `@if (editMode()) { quill } @else { div [innerHTML] }`
- [x] Update trades.html (5 fields)
- [x] Update weekly-prep.html (8 fields)
- [x] Remove `.quill-readonly` class from styles.css
- [x] Remove `[readOnly]` bindings from Quill editors

---

### E8-S4: Unify disabled field label colors

**As a** trader,
**I want** consistent muted styling on all labels in view mode.

**Acceptance criteria:**
- [ ] Rich text labels use `opacity-60` in view mode, full opacity in edit mode
- [ ] All sidebar pages are consistent
- [ ] `.quill-readonly` CSS class removed

**Affected layers:** Frontend (trades.html, weekly-prep.html, styles.css)
**Dependencies:** E8-S3
**Status:** Pending

**Implementation:**
- [x] Add `[class.opacity-60]="!editMode()"` to all rich text labels in trades.html
- [x] Same for weekly-prep.html
- [x] Delete `.quill-readonly` block from styles.css

---

## Implementation Order

```
E8-S2  (status filter — quick, independent)
E8-S1  (trade P&L — backend + frontend)
E8-S3  (fix rich text rendering)
E8-S4  (label colors — cleanup after S3)
  ↓
E7-S4  (link strategies to documents — deferred)
```
