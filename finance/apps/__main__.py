"""
finance.apps.__main__
======================
CLI entry point: ``python -m finance.apps [app-name] [options]``
"""
from __future__ import annotations

import argparse
import sys

from finance.apps import APPS, get_app, list_apps


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m finance.apps",
        description="Launch a finance application.",
    )
    parser.add_argument(
        "app",
        nargs="?",
        default=None,
        help="Application name to launch. Omit to list available apps.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="Start year for data loading (app-specific).",
    )

    args, remaining = parser.parse_known_args()

    if args.app is None:
        print("Available applications:\n")
        for name, desc in list_apps().items():
            print(f"  {name:20s}  {desc}")
        print(f"\nUsage: python -m finance.apps <app-name>")
        sys.exit(0)

    app_module = get_app(args.app)

    kwargs = {}
    if args.start_year is not None:
        kwargs["start_year"] = args.start_year

    app_module.launch(**kwargs)


if __name__ == "__main__":
    main()
