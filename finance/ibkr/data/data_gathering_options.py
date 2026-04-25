"""DEPRECATED: The daily/hourly options data gathering used TimescaleDB which has been removed.

To gather intraday data, use update_intraday.py instead:
  uv run python finance/ibkr/data/update_intraday.py
"""
raise ImportError("This script is deprecated. TimescaleDB has been removed.")
