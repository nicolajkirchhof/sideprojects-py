from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _read_ibkr_csv(csv_path: Path) -> pd.DataFrame:
    """
    Read an IBKR CSV that was likely written via DataFrame.to_csv(index=True),
    where the index name is often 'date'. Falls back gracefully.
    """
    # Try the common case first: index column is 'date'
    try:
        df = pd.read_csv(csv_path, index_col="date", parse_dates=["date"])
        df.index.name = "date"
        return df
    except Exception:
        pass

    # Try: first column is the index (often unnamed)
    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if df.index.name is None:
            df.index.name = "date"
        return df
    except Exception:
        pass

    # Fallback: plain read, then try to promote a 'date' column to index
    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.set_index("date")
        df.index.name = "date"
    return df


def convert_tree(root_dir: Path, force: bool, delete_csv: bool) -> tuple[int, int, int]:
    csv_files = sorted(root_dir.rglob("*.csv"))
    total = len(csv_files)
    converted = 0
    skipped = 0

    for csv_path in csv_files:
        pkl_path = csv_path.with_suffix(".pkl")

        if pkl_path.exists() and not force:
            csv_mtime = csv_path.stat().st_mtime
            pkl_mtime = pkl_path.stat().st_mtime
            if pkl_mtime >= csv_mtime:
                skipped += 1
                continue

        try:
            df = _read_ibkr_csv(csv_path)

            # Normalize for faster downstream loads
            if isinstance(df.index, pd.DatetimeIndex):
                df = df[~df.index.isna()]
                df = df.sort_index()

            df.to_pickle(pkl_path)
            converted += 1

            if delete_csv:
                csv_path.unlink(missing_ok=True)

        except Exception as e:
            print(f"[ERROR] {csv_path} -> {pkl_path}: {e}")

    return total, converted, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert IBKR CSV cache files to PKL for faster loads.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("finance/_data/ibkr"),
        help="Root directory to scan (default: finance/_data/ibkr)",
    )
    parser.add_argument("--force", action="store_true", help="Rewrite PKL even if it is newer than the CSV.")
    parser.add_argument(
        "--delete-csv",
        action="store_true",
        help="Delete CSV files after successful conversion (use with care).",
    )
    args = parser.parse_args()

    root_dir = args.root
    if not root_dir.exists():
        raise SystemExit(f"Root directory not found: {root_dir}")

    total, converted, skipped = convert_tree(root_dir, force=args.force, delete_csv=args.delete_csv)

    print(f"Scanned:   {total}")
    print(f"Converted:{converted}")
    print(f"Skipped:  {skipped}")
    if args.delete_csv:
        print("CSV deletion: enabled (converted files only).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
