---
paths:
  - "tradelog/**"
---

# Tradelog Application Context

## Architecture

Angular 21 frontend + .NET 10 backend REST API with SQL Server.

- **Backend:** `tradelog/backend/` — C# controllers, services, EF Core models, migrations
- **Frontend:** `tradelog/frontend/` — Angular + Tailwind CSS + PrimeNG
- **Tests:** `tradelog/backend.Tests/` — xUnit + SQLite in-memory

## Stack & Conventions

**Backend (.NET):**
- Entity Framework Core with SQL Server
- Account-scoped data via `IAccountContext` global query filters
- Services layer between controllers and data (no business logic in controllers)
- DTOs for API responses, Models for DB entities
- Decimal precision: 18,6 for all financial values

**Frontend (Angular):**
- Standalone components, lazy-loaded routes
- PrimeNG component library + Tailwind for layout
- Services call backend via HttpClient with `X-Account-Id` header

**Testing:**
- TDD: write tests first, then implement
- xUnit + `TestDbFixture` (SQLite in-memory) for integration tests
- Test behavior, not implementation

## Key Services

- `FlexQueryClient` — HTTP client for IBKR Flex Web Service API
- `FlexReportParser` — Parses Flex XML into typed DTOs
- `FlexSyncService` — Imports trades, positions, capital from Flex data
- `TwsLiveSyncService` — TWS connection for live Greeks + stock prices
- `IbkrConnectionManager` — TWS socket wrapper (DefaultEWrapper)

## API Endpoints

- `POST /api/ibkr/flex-sync` — Fetch Flex report, import trades/positions/capital
- `POST /api/ibkr/live-sync` — TWS Greeks + stock prices (1h cooldown)
- `GET /api/ibkr/sync/status` — Both sync timestamps
- CRUD: `/api/trades`, `/api/option-positions`, `/api/capital`, `/api/trade-entries`, `/api/accounts`

## Build & Run

```bash
# Backend
cd tradelog/backend && dotnet build
cd tradelog/backend && dotnet run
cd tradelog/backend.Tests && dotnet test

# Frontend
cd tradelog/frontend && npm install
cd tradelog/frontend && npm start        # dev server on :4200
cd tradelog/frontend && npm run build    # production build
```

## Available Roles

Use `/architect` for design, `/developer` for implementation, `/reviewer` for code review, `/reqeng` for requirements.
