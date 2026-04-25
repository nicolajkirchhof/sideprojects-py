"""
Scoring Backtest — interactive exploration
==========================================
Replay historical closed trades through the weighted 0-100 scoring engine
and validate that high-scoring entries correlate with better outcomes.

Run with:
    ipython -i finance/playground/scoring_backtest.py

Sections:
  1. Load trades + run backtest → df
  2. Score distribution (histogram)
  3. Win rate by score bucket
  4. Dimension breakdown — which D1–D5 are most predictive
  5. Scatter: score_total vs pnl
"""
from __future__ import annotations

from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from finance.utils.swing_backtest import load_trade_entries, run_scoring_backtest

# ---------------------------------------------------------------------------
# Section 1 — Load + backtest
# ---------------------------------------------------------------------------

print("Loading trade entries from Tradelog API...")
entries = load_trade_entries()
print(f"  {len(entries)} closed trades found")

print("Running scoring backtest (reconstructing indicators from IBKR parquets)...")
df = run_scoring_backtest(entries)
print(f"  {len(df)} trades successfully scored")
print()
print(df[["symbol", "entry_date", "direction", "score_total", "pnl", "win"]].head(20).to_string())
print()
print(df.describe())

# ---------------------------------------------------------------------------
# Section 2 — Score distribution
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df["score_total"].dropna(), bins=20, color="steelblue", edgecolor="white")
axes[0].set_title("Score Distribution")
axes[0].set_xlabel("score_total")
axes[0].set_ylabel("count")

df_wins = df[df["win"] == True]["score_total"].dropna()
df_losses = df[df["win"] == False]["score_total"].dropna()
axes[1].hist(df_wins, bins=20, alpha=0.6, color="green", label="Win", edgecolor="white")
axes[1].hist(df_losses, bins=20, alpha=0.6, color="red", label="Loss", edgecolor="white")
axes[1].set_title("Score Distribution: Wins vs Losses")
axes[1].set_xlabel("score_total")
axes[1].legend()
plt.tight_layout()
plt.savefig("finance/playground/scoring_backtest_dist.png", dpi=120)
plt.show()

# ---------------------------------------------------------------------------
# Section 3 — Win rate by score bucket
# ---------------------------------------------------------------------------

buckets = [(0, 40), (40, 55), (55, 70), (70, 112)]
bucket_labels = ["0–40", "40–55", "55–70", "70+"]

win_rates: list[float] = []
counts: list[int] = []
for lo, hi in buckets:
    mask = (df["score_total"] >= lo) & (df["score_total"] < hi) & df["win"].notna()
    sub = df[mask]
    wr = sub["win"].mean() * 100 if len(sub) > 0 else 0.0
    win_rates.append(wr)
    counts.append(len(sub))

print("\nWin Rate by Score Bucket:")
for label, wr, n in zip(bucket_labels, win_rates, counts):
    print(f"  {label:>6}: {wr:.1f}% win rate  (n={n})")

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(bucket_labels, win_rates, color="steelblue", edgecolor="white")
for bar, n in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"n={n}", ha="center", va="bottom", fontsize=9)
ax.axhline(50, color="gray", linestyle="--", alpha=0.5)
ax.set_title("Win Rate by Score Bucket")
ax.set_ylabel("Win rate (%)")
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig("finance/playground/scoring_backtest_winrate.png", dpi=120)
plt.show()

# ---------------------------------------------------------------------------
# Section 4 — Dimension breakdown: correlation with outcome
# ---------------------------------------------------------------------------

dim_cols = ["score_d1", "score_d2", "score_d3", "score_d4", "score_d5"]
dim_labels = ["D1 Trend", "D2 RS", "D3 Base", "D4 Catalyst", "D5 Risk"]

df_valid = df[df["win"].notna()].copy()
df_valid["win_int"] = df_valid["win"].astype(int)

correlations = [df_valid[col].corr(df_valid["win_int"]) for col in dim_cols]

print("\nDimension Correlation with Win:")
for label, corr in zip(dim_labels, correlations):
    bar = "#" * int(abs(corr) * 30)
    sign = "+" if corr >= 0 else "-"
    print(f"  {label}: {sign}{abs(corr):.3f}  {bar}")

fig, ax = plt.subplots(figsize=(8, 4))
colors = ["green" if c >= 0 else "red" for c in correlations]
ax.bar(dim_labels, correlations, color=colors, edgecolor="white")
ax.axhline(0, color="white", linewidth=0.5)
ax.set_title("Dimension Correlation with Win (Pearson r)")
ax.set_ylabel("Correlation")
plt.tight_layout()
plt.savefig("finance/playground/scoring_backtest_dimensions.png", dpi=120)
plt.show()

# ---------------------------------------------------------------------------
# Section 5 — Scatter: score_total vs pnl
# ---------------------------------------------------------------------------

df_pnl = df[df["pnl"].notna() & df["score_total"].notna()].copy()

fig, ax = plt.subplots(figsize=(10, 6))
colors_scatter = df_pnl["win"].map({True: "green", False: "red", None: "gray"})
ax.scatter(df_pnl["score_total"], df_pnl["pnl"], c=colors_scatter, alpha=0.6, s=40)
ax.axhline(0, color="gray", linestyle="--", alpha=0.4)

# Trend line
m, b = np.polyfit(df_pnl["score_total"], df_pnl["pnl"], 1)
xs = np.linspace(df_pnl["score_total"].min(), df_pnl["score_total"].max(), 100)
ax.plot(xs, m * xs + b, color="steelblue", linewidth=1.5, label=f"trend (slope={m:.1f})")
ax.set_title("Score vs PnL")
ax.set_xlabel("score_total")
ax.set_ylabel("PnL ($)")
ax.legend()
plt.tight_layout()
plt.savefig("finance/playground/scoring_backtest_scatter.png", dpi=120)
plt.show()

print("\nDone. Figures saved to finance/playground/.")
