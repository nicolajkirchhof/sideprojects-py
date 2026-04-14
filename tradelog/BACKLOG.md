# Tradelog — Product Backlog

## Completed

- ~~E1: Date Format Settings~~ — shipped
- ~~E2-S1: Opened/Closed columns~~ — shipped
- ~~E3: Configurable Enums (S1-S4)~~ — shipped (incl. Settings UI)
- ~~E4: Trade Creation from Positions (S0-S4)~~ — shipped
- ~~E5-S1: View-Mode Consistency~~ — shipped

---

## E6: Position Notes + Remove Trade Events

Replace trade events with per-position notes. Unify disabled styling.

### E6-S1: Add Notes field to OptionPosition and StockPosition

**As a** trader,
**I want** a plain text notes field on each option and stock position,
**So that** I can journal observations, adjustments, or reminders per position.

**Acceptance criteria:**
- [ ] `OptionPosition` model gains `Notes` string field (nullable, no length limit)
- [ ] `StockPosition` model gains `Notes` string field (nullable, no length limit)
- [ ] `OptionPositionDto` and `StockPositionDto` include `notes` in API responses
- [ ] EF migration adds the columns
- [ ] `PUT` endpoints for both position types persist the notes field
- [ ] Existing positions have `Notes = NULL` after migration

**Affected layers:** Data model, API, EF migration
**Dependencies:** None

---

### E6-S2: Notes input on position detail sidebars

**As a** trader,
**I want** to view and edit position notes in the option-positions and stock-positions sidebars,
**So that** I can add notes while reviewing a position.

**Acceptance criteria:**
- [ ] Option Positions sidebar: plain text `textarea` for Notes, respects view/edit mode
- [ ] Stock Positions sidebar: same
- [ ] Notes saved on form submit (existing Save flow)
- [ ] Empty notes field shows nothing in view mode

**Affected layers:** Frontend (option-positions, stock-positions)
**Dependencies:** E6-S1

---

### E6-S3: Show position notes in trade detail sidebar

**As a** trader,
**I want** to see notes from linked positions when viewing a trade,
**So that** I can review position-level observations in the context of the trade thesis.

**Acceptance criteria:**
- [ ] In the Legs section of the trade sidebar, if a position has notes, show them in a second row below the position's data row
- [ ] Notes row: smaller font, muted, indented
- [ ] Only renders when notes is non-null and non-empty
- [ ] Works for both option legs and stock legs
- [ ] Notes are read-only in the trade view (editing happens on the position pages)

**Affected layers:** Frontend (trades.html), API (OptionLegDto/StockLegDto must include notes)
**Dependencies:** E6-S1

---

### E6-S4: Remove trade events

**As a** trader,
**I want** trade events removed from the application,
**So that** the UI is simpler and I use position notes instead.

**Acceptance criteria:**
- [ ] `TradeEvents` table dropped via EF migration
- [ ] `TradeEvent` model, `TradeEventType` enum, `TRADE_EVENT_TYPE_LABELS` removed
- [ ] `TradeEventsController` (or endpoints) removed
- [ ] `TradeDetailDto.Events` field removed
- [ ] Events section removed from trades sidebar (HTML + TS)
- [ ] `Trade.TradeEvents` navigation property removed
- [ ] Existing trade event data deleted in migration
- [ ] Backend tests referencing TradeEvents updated or removed

**Affected layers:** Data model, API, Frontend, EF migration, Tests
**Dependencies:** None (cleanest to do last to avoid sidebar merge conflicts)

---

### E6-S5: Unify disabled/view-mode colors in sidebar

**As a** trader,
**I want** all read-only content in the detail sidebar to use the same muted color,
**So that** the visual language is consistent.

**Acceptance criteria:**
- [ ] In view mode, all field labels use `--mat-sys-on-surface-variant`
- [ ] All field values (text inputs, selects, checkboxes, Quill content) use the same muted color matching Material's disabled state
- [ ] Section headers use consistent opacity/color
- [ ] Applies to all pages with view/edit sidebars
- [ ] `.quill-readonly` text color matches Material disabled input color exactly

**Affected layers:** Frontend (`styles.css`, templates)
**Dependencies:** None

---

## E7: Strategy Library (replacing Portfolio)

Replace the flat Portfolio allocation page with a Strategy Library — a collection of named markdown documents editable in-app. Uses `ngx-markdown` for rendering + `<textarea>` for editing.

### E7-S1: Document model and API

**As a** trader,
**I want** named documents stored in the database that I can link to strategies,
**So that** my trading playbooks are accessible inside the app.

**Acceptance criteria:**
- [ ] New `Document` model: `Id`, `AccountId`, `Title` (required), `Content` (nvarchar(max), markdown), `UpdatedAt`, optional `LookupValueId` (FK → LookupValues, links to a strategy)
- [ ] EF migration creates the table
- [ ] `DocumentsController` with CRUD: `GET /api/documents`, `GET /{id}`, `POST`, `PUT`, `DELETE`
- [ ] Account-scoped via global query filter
- [ ] Seed: import existing `.md` files from `investing_framework/` as initial documents, linked to matching strategy names where applicable

**Affected layers:** Data model, API, EF migration
**Dependencies:** None

---

### E7-S2: Strategy Library page with markdown viewer and editor

**As a** trader,
**I want** a page where I see all my strategy documents and can view/edit them in markdown,
**So that** I can reference and update my playbooks without leaving the app.

**Acceptance criteria:**
- [ ] New `/strategy-library` route replacing `/portfolio` in the sidebar
- [ ] Left panel: list of documents (title + linked strategy name), click to select
- [ ] Right panel: markdown rendered view (ngx-markdown) in view mode
- [ ] Edit mode: `<textarea>` with raw markdown, optional live preview
- [ ] Save via `PUT /api/documents/{id}`
- [ ] "New Document" button
- [ ] Title editable inline
- [ ] Strategy link: dropdown to optionally link to a LookupValue (Strategy)
- [ ] Install `ngx-markdown` dependency

**Affected layers:** Frontend (new page, new dependency)
**Dependencies:** E7-S1

---

### E7-S3: Remove Portfolio model and page

**As a** trader,
**I want** the old Portfolio allocation page removed.

**Acceptance criteria:**
- [ ] `Portfolios` table dropped via EF migration
- [ ] `Portfolio` model, `PortfolioDto`, `PortfolioController` removed
- [ ] `PortfolioComponent` + route removed from frontend
- [ ] Sidebar link changed from "Portfolio" to "Strategy Library"
- [ ] `PortfolioAggregationService` removed if only used by portfolio
- [ ] Tests referencing Portfolio updated or removed

**Affected layers:** Data model, API, Frontend, EF migration, Tests
**Dependencies:** E7-S2 (replacement must exist first)

---

### E7-S4: Link strategies to documents

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

---

## Implementation Order

```
E6 and E7 are independent — can be interleaved or done in either order.

E6:  S1 → S2 → S3 → S4 → S5
E7:  S1 → S2 → S3 → S4
```
