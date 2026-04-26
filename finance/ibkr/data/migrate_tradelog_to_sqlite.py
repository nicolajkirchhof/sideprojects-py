"""One-time migration: copy all data from Azure SQL (staging) to local SQLite.

Source: tradelog-staging on ocin.database.windows.net (Azure SQL)
Dest:   tradelog/_data/tradelog.db (SQLite, created by dotnet run EnsureCreated)

Usage:
  uv run python -m finance.ibkr.data.migrate_tradelog_to_sqlite
  uv run python -m finance.ibkr.data.migrate_tradelog_to_sqlite --dry-run
  uv run python -m finance.ibkr.data.migrate_tradelog_to_sqlite --db path/to/tradelog.db
"""
from __future__ import annotations

import argparse
import sqlite3
from decimal import Decimal
from pathlib import Path

import pyodbc

# ── Connection details ────────────────────────────────────────────────────────

SQL_SERVER = "ocin.database.windows.net,1433"
SQL_DB = "tradelog_stage"
SQL_USER = "ocin"
# Development DB password (DbPassword:Development from dotnet user-secrets)
SQL_PASSWORD = "oMPbvsg6MbgcFfxmH39pWa6HtMCJjdyv"

DEFAULT_SQLITE_PATH = Path("tradelog/_data/tradelog.db")

# ── Tables present in both prod SQL Server and the new SQLite schema ──────────
# Newer tables (LookupValues, Documents, DocumentStrategyLinks, DailyPreps,
# TradeAnalyses) are not yet in prod — they start empty in SQLite.
# Removed tables (Portfolios, TradeEvents) are skipped.

TABLES: list[str] = [
    "Accounts",
    "StockPriceCaches",
    "Trades",           # self-FK (ParentTradeId), but SQLite won't enforce it
    "OptionPositions",
    "StockPositions",
    "OptionPositionsLogs",
    "Capitals",
    "WeeklyPreps",
]

# ── Helpers ───────────────────────────────────────────────────────────────────


def connect_sqlserver() -> pyodbc.Connection:
    conn_str = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        f"SERVER={SQL_SERVER};"
        f"DATABASE={SQL_DB};"
        f"UID={SQL_USER};"
        f"PWD={SQL_PASSWORD};"
        "Encrypt=yes;TrustServerCertificate=no;Connection Timeout=60;"
    )
    return pyodbc.connect(conn_str)


def get_columns(src_cur: pyodbc.Cursor, table: str) -> list[str]:
    src_cur.execute(f"SELECT TOP 0 * FROM [{table}]")
    return [col[0] for col in src_cur.description]


def migrate_table(
    src_cur: pyodbc.Cursor,
    dst: sqlite3.Connection,
    table: str,
    dry_run: bool,
) -> int:
    cols = get_columns(src_cur, table)
    col_list = ", ".join(f'"{c}"' for c in cols)
    placeholders = ", ".join("?" * len(cols))

    src_cur.execute(f"SELECT {col_list} FROM [{table}]")
    rows = src_cur.fetchall()
    if not rows:
        return 0

    if dry_run:
        return len(rows)

    # sqlite3 doesn't support decimal.Decimal — convert to float
    def coerce(v: object) -> object:
        return float(v) if isinstance(v, Decimal) else v

    coerced = [tuple(coerce(v) for v in row) for row in rows]

    # Disable FK enforcement during bulk insert (SQLite default is off anyway)
    dst.execute("PRAGMA foreign_keys = OFF")
    dst.executemany(
        f'INSERT OR REPLACE INTO "{table}" ({col_list}) VALUES ({placeholders})',
        coerced,
    )
    dst.commit()
    dst.execute("PRAGMA foreign_keys = ON")
    return len(rows)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate tradelog data from Azure SQL to SQLite")
    parser.add_argument("--db", default=str(DEFAULT_SQLITE_PATH), help="Path to SQLite DB")
    parser.add_argument("--dry-run", action="store_true", help="Count rows without writing")
    args = parser.parse_args()

    sqlite_path = Path(args.db)
    if not sqlite_path.exists():
        raise FileNotFoundError(
            f"SQLite DB not found at {sqlite_path}. "
            "Start the backend first so EnsureCreated() creates the schema."
        )

    print(f"Source: {SQL_DB} @ {SQL_SERVER}")
    print(f"Dest:   {sqlite_path}")
    if args.dry_run:
        print("[DRY RUN — no data will be written]")
    print()

    print("Connecting to SQL Server...")
    src_con = connect_sqlserver()
    src_cur = src_con.cursor()

    dst = sqlite3.connect(sqlite_path)

    total = 0
    for table in TABLES:
        count = migrate_table(src_cur, dst, table, args.dry_run)
        print(f"  {table:<30} {count:>6} rows")
        total += count

    src_con.close()
    dst.close()

    print()
    print(f"Total: {total} rows {'counted' if args.dry_run else 'migrated'}")


if __name__ == "__main__":
    main()
