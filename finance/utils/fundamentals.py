"""
finance.utils.fundamentals
===========================
Market cap classification, accruals anomaly computation, and Piotroski F-Score.

Accruals and F-Score are pure functions that operate on DataFrames — no DB
dependency. The Dolt query functions are separate for testability.
"""
import numpy as np
import pandas as pd

df_market_cap_thresholds = pd.read_csv('finance/_data/ref/MarketCapThresholds.csv')

def classify_market_cap(mcap_value, year):
  """Classifies market cap based on historical thresholds (from reprocess.py)."""
  if mcap_value is None or np.isnan(mcap_value):
    return "Unknown"

  # Get thresholds for the closest available year
  year_idx = df_market_cap_thresholds['Year'].sub(year).abs().idxmin()
  row = df_market_cap_thresholds.loc[year_idx]

  if mcap_value >= row['Large-Cap Entry (S&P 500)']:
    return "Large"
  elif mcap_value >= row['Mid-Cap Entry (S&P 400)']:
    return "Mid"
  elif mcap_value >= row['Small-Cap Entry (Russell 2000)']:
    return "Small"
  else:
    return "Micro"


# ---------------------------------------------------------------------------
# Accruals Anomaly (Sloan 1996)
# ---------------------------------------------------------------------------

def compute_accruals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute accruals ratio: (net_income - operating_cash_flow) / total_assets.

    High positive accruals = earnings inflated by accounting (short signal).
    Negative accruals = cash-backed earnings (long signal).

    Parameters
    ----------
    df : DataFrame with columns: net_income, net_cash_from_operating_activities,
         total_assets. Preserves all existing columns.

    Returns DataFrame with 'accruals_ratio' added.
    """
    df = df.copy()

    ni = pd.to_numeric(df.get("net_income"), errors="coerce") if "net_income" in df.columns else pd.Series(np.nan, index=df.index)
    cfo = pd.to_numeric(df.get("net_cash_from_operating_activities"), errors="coerce") if "net_cash_from_operating_activities" in df.columns else pd.Series(np.nan, index=df.index)
    ta = pd.to_numeric(df.get("total_assets"), errors="coerce") if "total_assets" in df.columns else pd.Series(np.nan, index=df.index)

    df["accruals_ratio"] = np.where(
        ta.notna() & (ta != 0) & ni.notna() & cfo.notna(),
        (ni - cfo) / ta.abs(),
        np.nan,
    )

    return df


# ---------------------------------------------------------------------------
# Piotroski F-Score (Piotroski 2000)
# ---------------------------------------------------------------------------

def _safe_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    """Extract column as float, returning NaN series if missing."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def compute_fscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 9-point Piotroski F-Score.

    Expects a DataFrame with current AND prior-quarter values. Prior-quarter
    columns are prefixed with 'prev_'. All 9 signals are computed as binary
    (0 or 1) columns prefixed with 'f_', and summed into 'fscore'.

    Profitability (4):
      1. f_roa_positive:  ROA > 0 (net_income / total_assets)
      2. f_cfo_positive:  CFO > 0
      3. f_delta_roa:     ROA improved vs prior quarter
      4. f_accruals:      CFO > net_income (accruals < 0)

    Leverage / Liquidity (3):
      5. f_delta_ltd:     Long-term debt decreased
      6. f_delta_cr:      Current ratio improved
      7. f_no_dilution:   No share dilution

    Operating Efficiency (2):
      8. f_delta_margin:  Gross margin improved
      9. f_delta_turnover: Asset turnover improved

    Returns DataFrame with f_* columns and 'fscore' (0-9).
    """
    if df.empty:
        out = df.copy()
        for col in ["fscore", "f_roa_positive", "f_cfo_positive", "f_delta_roa",
                     "f_accruals", "f_delta_ltd", "f_delta_cr", "f_no_dilution",
                     "f_delta_margin", "f_delta_turnover"]:
            out[col] = pd.Series(dtype=float)
        return out

    df = df.copy()

    # Current quarter
    ni = _safe_numeric(df, "net_income")
    cfo = _safe_numeric(df, "net_cash_from_operating_activities")
    ta = _safe_numeric(df, "total_assets")
    tca = _safe_numeric(df, "total_current_assets")
    tcl = _safe_numeric(df, "total_current_liabilities")
    ltd = _safe_numeric(df, "long_term_debt")
    gp = _safe_numeric(df, "gross_profit")
    sales = _safe_numeric(df, "sales")
    shares = _safe_numeric(df, "average_shares")

    # Prior quarter
    prev_ni = _safe_numeric(df, "prev_net_income")
    prev_ta = _safe_numeric(df, "prev_total_assets")
    prev_cfo = _safe_numeric(df, "prev_net_cash_from_operating_activities")
    prev_ltd = _safe_numeric(df, "prev_long_term_debt")
    prev_tca = _safe_numeric(df, "prev_total_current_assets")
    prev_tcl = _safe_numeric(df, "prev_total_current_liabilities")
    prev_gp = _safe_numeric(df, "prev_gross_profit")
    prev_sales = _safe_numeric(df, "prev_sales")
    prev_shares = _safe_numeric(df, "prev_average_shares")

    # ROA
    roa = ni / ta.where(ta != 0)
    prev_roa = prev_ni / prev_ta.where(prev_ta != 0)

    # Current ratio
    cr = tca / tcl.where(tcl != 0)
    prev_cr = prev_tca / prev_tcl.where(prev_tcl != 0)

    # Gross margin
    margin = gp / sales.where(sales != 0)
    prev_margin = prev_gp / prev_sales.where(prev_sales != 0)

    # Asset turnover
    turnover = sales / ta.where(ta != 0)
    prev_turnover = prev_sales / prev_ta.where(prev_ta != 0)

    # 9 binary signals
    df["f_roa_positive"] = (roa > 0).astype(int)
    df["f_cfo_positive"] = (cfo > 0).astype(int)
    df["f_delta_roa"] = (roa > prev_roa).astype(int)
    df["f_accruals"] = (cfo > ni).astype(int)
    df["f_delta_ltd"] = (ltd <= prev_ltd).astype(int)
    df["f_delta_cr"] = (cr > prev_cr).astype(int)
    df["f_no_dilution"] = (shares <= prev_shares).astype(int)
    df["f_delta_margin"] = (margin > prev_margin).astype(int)
    df["f_delta_turnover"] = (turnover > prev_turnover).astype(int)

    f_cols = [c for c in df.columns if c.startswith("f_")]
    df["fscore"] = df[f_cols].sum(axis=1)

    return df
