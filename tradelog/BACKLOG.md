# Tradelog — Product Backlog

## E1: Date Format Settings

Global application setting controlling how dates display and how date inputs behave. Persisted as app-wide config (not per-user). Default: `yyyy-MM-dd` (ISO).

### E1-S1: Date format configuration service

**As a** trader,
**I want to** choose my preferred date format (ISO / US / German),
**So that** dates across the application match my locale preference.

**Acceptance criteria:**
- [ ] Three supported formats: `yyyy-MM-dd` (ISO, default), `MM/dd/yyyy` (US), `dd.MM.yyyy` (German)
- [ ] Setting persisted in `localStorage` and exposed via an injectable `DateFormatService` signal
- [ ] Changing the format applies immediately without page reload

**Affected layers:** Frontend (service)
**Dependencies:** None
**Notes:** This is a frontend-only setting for now. If multi-user support is added later, migrate to DB-backed per-user preference.

---

### E1-S2: Apply date format to all table columns

**As a** trader,
**I want** all date columns in every table to respect my chosen format,
**So that** I see consistent dates across the application.

**Acceptance criteria:**
- [ ] All `| date:` pipes across option-positions, stock-positions, trades, capital, weekly-prep, accounts use the format from `DateFormatService`
- [ ] Existing `shortDate` pipes replaced with a custom pipe or directive that reads the service
- [ ] Sorting by date columns still works correctly (sorts by underlying Date, not display string)

**Affected layers:** Frontend (pipe + all table templates)
**Dependencies:** E1-S1
**Notes:** Consider a shared `AppDatePipe` that injects `DateFormatService` to avoid passing the format everywhere.

---

### E1-S3: Apply date format to all date inputs

**As a** trader,
**I want** date pickers and text inputs for dates to match my chosen format,
**So that** I enter dates in a consistent format.

**Acceptance criteria:**
- [ ] All `mat-datepicker` inputs display the selected date in the configured format
- [ ] Manual text entry in date fields parses the configured format
- [ ] Invalid date input shows a validation error, does not silently accept the wrong format

**Affected layers:** Frontend (MatDatepicker custom adapter + all form templates)
**Dependencies:** E1-S1
**Notes:** Angular Material supports custom `DateAdapter` — implement a format-aware adapter that reads from `DateFormatService`.

---

### E1-S4: Settings page with date format selector

**As a** trader,
**I want** a Settings page accessible from the sidebar,
**So that** I can change the date format and any future app-wide settings.

**Acceptance criteria:**
- [ ] New `/settings` route with a standalone component in the sidebar navigation
- [ ] Radio group or dropdown for date format selection with live preview of a sample date
- [ ] Persists on selection (auto-save, no submit button needed)
- [ ] Page is extensible for future settings (layout should use sections/cards)

**Affected layers:** Frontend (new route, component, sidebar nav)
**Dependencies:** E1-S1

---

## E2: Option Position Table — Opened & Closed Columns

Currently the option-positions table only shows Expiry. Add Opened and Closed as visible, sortable columns.

### E2-S1: Add Opened and Closed columns to option-positions table

**As a** trader,
**I want to** see when each option position was opened and closed directly in the table,
**So that** I can sort and scan positions by entry/exit timing without opening the detail sidebar.

**Acceptance criteria:**
- [ ] `Opened` column added before the existing `Expiry` column, with `mat-sort-header`
- [ ] `Closed` column added after `Expiry`, with `mat-sort-header`
- [ ] Closed shows `—` (muted) for open positions (null value)
- [ ] Both columns use the date format from `DateFormatService` (or the existing pipe if E1 is not yet implemented)
- [ ] Column order in table: Symbol, Contract, R, Strike, Opened, Expiry, Closed, Pos, Cost, ...

**Affected layers:** Frontend (option-positions.html, option-positions.ts — displayedColumns array)
**Dependencies:** None (E1-S2 will update the date pipe later if implemented first)

---

## E3: Configurable Enums (DB-backed Lookups)

Replace hardcoded C# enums with database-backed lookup tables so users can add, rename, and remove values through the UI. Renaming updates all historical references.

> **Architectural decision required.** This epic changes the data model fundamentally (int enums → FK references or string-based lookups). Switch to `/architect` before implementation to decide: generic lookup table vs per-enum table, migration strategy for existing int-valued rows, and cascading rename semantics.

### E3-S1: Data model for configurable lookups

**As a** developer,
**I want** enum values stored in the database as lookup rows,
**So that** users can manage them without code changes or redeployment.

**Acceptance criteria:**
- [ ] Each configurable enum has a corresponding lookup table (or rows in a generic table) with at minimum: `Id`, `AccountId`, `Name`, `SortOrder`, `IsActive`
- [ ] Existing integer enum values are migrated to FK references in a data migration
- [ ] Historical trades/positions referencing a renamed value automatically reflect the new name
- [ ] Deleting a lookup value is soft-delete (`IsActive = false`) — prevents orphaned FK references
- [ ] Enums to convert: `Strategy`, `TypeOfTrade`, `Budget`, `Timeframe`, `DirectionalBias`, `ManagementRating`

**Affected layers:** Data model, EF migration, all backend services that read/write these fields
**Dependencies:** Architectural decision (see note above)
**Notes:** Start with `Strategy` and `TypeOfTrade` as the highest-priority pair. The remaining four can follow in a second pass once the pattern is proven.

---

### E3-S2: CRUD API for lookup values

**As a** trader,
**I want** API endpoints to list, create, rename, reorder, and deactivate lookup values,
**So that** the settings UI can manage them.

**Acceptance criteria:**
- [ ] `GET /api/lookups/{enumType}` — returns ordered list of active values for an enum type
- [ ] `POST /api/lookups/{enumType}` — creates a new value (name + sort order)
- [ ] `PUT /api/lookups/{enumType}/{id}` — renames a value; cascades rename to all referencing rows
- [ ] `PATCH /api/lookups/{enumType}/{id}/deactivate` — soft-deletes; value no longer appears in dropdowns but historical data is preserved
- [ ] `PATCH /api/lookups/{enumType}/{id}/reorder` — updates sort order
- [ ] Validation: duplicate names within the same enum type are rejected

**Affected layers:** API (new controller), backend services
**Dependencies:** E3-S1

---

### E3-S3: Settings UI for managing lookup values

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

**Affected layers:** Frontend (settings component, new API service)
**Dependencies:** E3-S2, E1-S4 (shares the Settings page shell)

---

### E3-S4: Replace hardcoded enum dropdowns with dynamic lookups

**As a** trader,
**I want** all form dropdowns (Strategy, TypeOfTrade, etc.) to pull their options from the database,
**So that** newly added values appear immediately in trade entry forms.

**Acceptance criteria:**
- [ ] All `mat-select` dropdowns for configurable enums fetch values from the lookup API
- [ ] Deactivated values do NOT appear in dropdowns for new entries
- [ ] Deactivated values DO still display correctly on historical trades (read-only rendering)
- [ ] Forms affected: Trades, Option Positions (if applicable), Portfolio, Analytics filters
- [ ] Enum labels used in table cells also resolve from the lookup cache (not hardcoded label maps)

**Affected layers:** Frontend (all form components, shared lookup service with caching)
**Dependencies:** E3-S2

---

## Suggested Implementation Order

```
E2-S1  (quick win, no dependencies, ~30 min)
  ↓
E1-S1 → E1-S2 → E1-S3 → E1-S4  (date format, incremental)
  ↓
E3-S1 → E3-S2 → E3-S3 → E3-S4  (configurable enums, needs /architect first)
```

**Rationale:**
- E2-S1 is a single-file frontend change — ship it immediately.
- E1 is self-contained and builds incrementally (service → pipes → inputs → settings page).
- E3 is the largest and requires an architectural decision before any code. Start it after E1 ships so the Settings page (E1-S4) already exists as a shell for E3-S3.
