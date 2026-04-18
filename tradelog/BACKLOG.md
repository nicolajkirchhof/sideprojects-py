# Tradelog — Product Backlog

## Completed

- ~~E1: Date Format Settings~~ — shipped
- ~~E2-S1: Opened/Closed columns~~ — shipped
- ~~E3: Configurable Enums (S1-S4)~~ — shipped (incl. Settings UI)
- ~~E4: Trade Creation from Positions (S0-S4)~~ — shipped
- ~~E5-S1: View-Mode Consistency~~ — shipped
- ~~E6: Position Notes + Remove Trade Events (S1-S4)~~ — shipped
- ~~E7: Strategy Library (S1-S3)~~ — shipped
- ~~E8: Trades Table Enhancements + View-Mode Fix (S1-S4)~~ — shipped

---

## E7-S4: Link strategies to documents

**As a** trader,
**I want** to navigate from a trade's strategy to its playbook document,
**So that** I can quickly check the rules.

**Acceptance criteria:**
- [ ] Trades table: Strategy column shows a link icon if the strategy has a linked document
- [ ] Clicking the icon navigates to `/strategy-library?doc={id}`
- [ ] Strategy Library reads query param and auto-selects the document

**Affected layers:** Frontend only (trades, strategy-library)
**Dependencies:** None (Document model + API already exist)
**Status:** Pending

**Implementation:**
- [x] In `trades.ts`: inject `DocumentService`, load documents on init, build `strategyDocMap: Map<number, number>` (strategyId → first documentId)
- [x] In `trades.ts`: add `docForStrategy(strategyId)` method and `openDoc(docId)` navigation
- [x] In `trades.html`: add link icon next to strategy name in the Strategy column
- [x] In `strategy-library.ts`: read `doc` query param on init and auto-select the document

---

## E9: Copy Ticker Lists to Clipboard

### E9-S1: Copy option position tickers

**As a** trader,
**I want to** copy all visible option position symbols as a comma-separated list from the Option Positions page,
**So that** I can paste them into Barchart to track my option underlyings.

**Acceptance criteria:**
- [x] Icon button with tooltip in the filter toolbar on Option Positions page
- [x] Clicking extracts unique symbols from filtered rows
- [x] Symbols joined with `,` and written to clipboard
- [x] Snackbar confirms "Copied N tickers to clipboard"
- [x] Shows error snackbar when no rows
- [x] Respects all active filters (status, symbol, right, expiry)

**Affected layers:** Frontend only
**Status:** Done

### E9-S2: Copy stock/futures position tickers

**As a** trader,
**I want to** copy all visible stock position symbols as a comma-separated list from the Stock Positions page,
**So that** I can paste them into Barchart to track my stock underlyings.

**Acceptance criteria:**
- [x] Icon button with tooltip in the filter toolbar on Stock Positions page
- [x] Clicking extracts unique symbols from filtered rows
- [x] Symbols joined with `,` and written to clipboard
- [x] Snackbar confirms "Copied N tickers to clipboard"
- [x] Shows error snackbar when no rows
- [x] Respects active symbol filter

**Affected layers:** Frontend only
**Status:** Done

### E9-S3: Dashboard → Analytics, remove Instrument Summaries

**As a** trader,
**I want** the Dashboard to show Analytics (equity curves, strategy/budget performance),
**So that** I see the most important overview when I open the app.

**Acceptance criteria:**
- [x] `/dashboard` route points to `AnalyticsComponent`
- [x] Analytics nav link removed (Dashboard replaces it)
- [x] Instrument Summaries component deleted
- [x] No broken routes or imports

**Affected layers:** Frontend only (routing, nav, cleanup)
**Status:** Done
