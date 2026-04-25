# %% Imports
import pandas as pd
from pathlib import Path
from datetime import date

# %% Load archive
DATA_DIR = Path("finance/_data/assistant")

today = date.today().isoformat()
path = DATA_DIR / f"{today}.parquet"
if not path.exists():
    # Fall back to latest available file
    path = sorted(DATA_DIR.glob("*.parquet"))[-1]
    print(f"No file for today — loaded {path.name}")

df = pd.read_parquet(path)
print(f"Loaded {path.name}  →  {len(df)} rows")

# %% Split market vs candidates
mkt = df[df["row_type"] == "market"].copy()
cands = df[df["row_type"] != "market"].copy()

print(f"Market instruments : {len(mkt)}")
print(f"Candidates         : {len(cands)}")

# %% Market overview
MARKET_COLS = [
    "symbol", "category",
    "price", "change_pct", "change_5d_pct",
    "change_1m_pct", "change_3m_pct", "change_52w_pct",
    "atr_pct_20d", "iv_percentile", "hv20",
    "pct_from_50d_sma", "slope_50d_sma", "slope_200d_sma",
    "bb_pct", "ttm_squeeze",
]

overview = mkt[MARKET_COLS].set_index("symbol")
overview

# %% By category
for cat, grp in mkt.groupby("category"):
    print(f"\n── {cat} ──")
    print(grp[["symbol", "price", "change_pct", "change_1m_pct", "change_52w_pct"]].to_string(index=False))

# %% Forex
forex = mkt[mkt["category"] == "Forex"][MARKET_COLS]
forex

# %% Indices
indices = mkt[mkt["category"] == "Indices"][MARKET_COLS]
indices

# %% Volatility + Bonds
vol_bonds = mkt[mkt["category"].isin(["Volatility", "Bonds"])][MARKET_COLS]
vol_bonds

# %% Commodities
commodities = mkt[mkt["category"].str.startswith("Commodities")][MARKET_COLS]
commodities

# %% Full raw market rows
mkt
