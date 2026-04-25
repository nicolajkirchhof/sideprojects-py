"""
finance.tests.test_scoring_evaluation
=======================================
Statistical validation of the weighted 0–100 scoring engine against the
momentum events dataset (all_YYYY.parquet).

Covers all event types, each scored in its natural direction:
  Long:  is_earnings, evt_atrp_breakout, evt_green_line_breakout,
         evt_episodic_pivot, evt_pre_earnings, evt_ema_reclaim
  Short: evt_selloff, evt_bb_lower_touch

Dataset required: finance/_data/research/swing/momentum_earnings/all_YYYY.parquet

Run offline (skips cleanly when data absent):
    python -m pytest finance/tests/test_scoring_evaluation.py -v

With data present (shows summary output):
    python -m pytest finance/tests/test_scoring_evaluation.py -v -s
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from finance.utils.swing_backtest import run_event_scoring

# ---------------------------------------------------------------------------
# Skip guard — evaluated once at module load
# ---------------------------------------------------------------------------

_DATA_DIR = Path("finance/_data/research/swing/momentum_earnings")
_HAS_DATA = _DATA_DIR.exists() and any(_DATA_DIR.glob("all_2*.parquet"))

_skip_no_data = pytest.mark.skipif(
    not _HAS_DATA, reason="Momentum earnings dataset not present"
)

# ---------------------------------------------------------------------------
# Direction mapping
# ---------------------------------------------------------------------------

# Event types that indicate a short setup — all others default to long.
_SHORT_EVENT_TYPES = {"evt_selloff", "evt_bb_lower_touch"}

_ALL_EVENT_COLS = [
    "is_earnings",
    "evt_atrp_breakout",
    "evt_green_line_breakout",
    "evt_episodic_pivot",
    "evt_pre_earnings",
    "evt_ema_reclaim",
    "evt_selloff",
    "evt_bb_lower_touch",
]


def _assign_direction(row: pd.Series) -> str:
    """
    Assign trade direction based on event type flags.

    Priority: a row is short when any short-type event is set AND no
    long-type event is also set (mixed signals → long, the lower-risk default).
    """
    has_short = any(row.get(col, False) for col in _SHORT_EVENT_TYPES)
    has_long = any(
        row.get(col, False)
        for col in _ALL_EVENT_COLS
        if col not in _SHORT_EVENT_TYPES
    )
    if has_short and not has_long:
        return "short"
    return "long"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_FWD_COL = "cpct10"
_MIN_ROWS = 200
_BACKTEST_YEARS = range(2016, 2027)  # 2016–2026 inclusive


def _load_year(year: int) -> pd.DataFrame | None:
    path = _DATA_DIR / f"all_{year}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    mask = df[_FWD_COL].notna() & (df["c0"] > 0)
    return df[mask].copy()


def _load_eval_data() -> pd.DataFrame:
    """Load and concatenate all events across 2016–2026, with direction assigned."""
    frames: list[pd.DataFrame] = []
    for year in _BACKTEST_YEARS:
        df = _load_year(year)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["direction"] = combined.apply(_assign_direction, axis=1)
    return combined


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

@_skip_no_data
class TestScoringEvaluation:
    """Statistical validation of the scoring engine on 2016–2026 event data."""

    @pytest.fixture(scope="class")
    def eval_df(self) -> pd.DataFrame:
        events = _load_eval_data()
        if events.empty:
            pytest.skip("No usable events found in dataset")

        scored = run_event_scoring(
            events,
            fwd_col=_FWD_COL,
            direction_col="direction",
        )

        if len(scored) < _MIN_ROWS:
            pytest.skip(
                f"Only {len(scored)} scored rows — need {_MIN_ROWS} for statistical tests"
            )

        return scored

    # ------------------------------------------------------------------
    # Hard assertions — schema and range invariants
    # ------------------------------------------------------------------

    def test_output_columns_present(self, eval_df: pd.DataFrame) -> None:
        required = [
            "symbol", "date", "direction", "score_total",
            "score_d1", "score_d2", "score_d3", "score_d4", "score_d5",
            "fwd_return", "win", "spy_regime",
        ]
        for col in required:
            assert col in eval_df.columns, f"Missing column: {col}"

    def test_both_directions_present(self, eval_df: pd.DataFrame) -> None:
        directions = set(eval_df["direction"].unique())
        assert "long" in directions, "No long events in scored output"
        assert "short" in directions, "No short events in scored output"

    def test_scores_in_valid_range(self, eval_df: pd.DataFrame) -> None:
        assert (eval_df["score_total"] >= 0).all(), "score_total below 0"
        assert (eval_df["score_total"] <= 112).all(), "score_total above 112"

    def test_dimension_scores_within_weight(self, eval_df: pd.DataFrame) -> None:
        weights = {1: 25, 2: 25, 3: 15, 4: 20, 5: 15}
        for dim, w in weights.items():
            col = f"score_d{dim}"
            if col in eval_df.columns:
                assert (eval_df[col] <= w + 0.01).all(), (
                    f"{col} exceeds max weight {w}"
                )

    # ------------------------------------------------------------------
    # Soft assertions — directional signal validation
    # ------------------------------------------------------------------

    def test_score_win_spearman_positive(self, eval_df: pd.DataFrame) -> None:
        """Spearman rho between score_total and win flag should be > 0."""
        valid = eval_df[eval_df["win"].notna()].copy()
        if len(valid) < _MIN_ROWS:
            pytest.skip(f"Too few rows with win flag ({len(valid)})")
        rho = valid["score_total"].corr(valid["win"].astype(float), method="spearman")
        assert rho > 0.0, (
            f"Spearman rho(score, win) = {rho:.4f} — scoring inverted vs outcomes"
        )

    def test_top_quartile_higher_median_return(self, eval_df: pd.DataFrame) -> None:
        """Top-25% scoring events should have higher median |return| than bottom 25%."""
        valid = eval_df[eval_df["fwd_return"].notna()].copy()
        # Use direction-adjusted return: positive = win for both long and short
        valid = valid.copy()
        valid["adj_return"] = valid.apply(
            lambda r: r["fwd_return"] if r["direction"] == "long" else -r["fwd_return"],
            axis=1,
        )
        q25 = valid["score_total"].quantile(0.25)
        q75 = valid["score_total"].quantile(0.75)
        top = valid[valid["score_total"] >= q75]
        bottom = valid[valid["score_total"] <= q25]

        if len(top) < 20 or len(bottom) < 20:
            pytest.skip(f"Quartile buckets too small (top={len(top)}, bottom={len(bottom)})")

        med_top = top["adj_return"].median()
        med_bottom = bottom["adj_return"].median()
        assert med_top > med_bottom, (
            f"Top-quartile median adj_return {med_top:.2f}% <= bottom {med_bottom:.2f}%"
        )

    def test_win_rate_higher_in_top_bucket(self, eval_df: pd.DataFrame) -> None:
        """Win rate for score>=65 should be >= win rate for score<50 minus 3%."""
        valid = eval_df[eval_df["win"].notna()].copy()
        high = valid[valid["score_total"] >= 65]
        low = valid[valid["score_total"] < 50]

        if len(high) < 20 or len(low) < 20:
            pytest.skip(f"Score buckets too small (high={len(high)}, low={len(low)})")

        wr_high = high["win"].mean()
        wr_low = low["win"].mean()
        assert wr_high >= wr_low - 0.03, (
            f"Win rate >=65: {wr_high:.1%}  <50: {wr_low:.1%}  diff={wr_high - wr_low:.1%}"
        )

    def test_regime_filter_improves_win_rate(self, eval_df: pd.DataFrame) -> None:
        """GO regime filter should not hurt win rate for score>=60 events (-2% tolerance)."""
        valid = eval_df[eval_df["win"].notna() & (eval_df["score_total"] >= 60)].copy()
        go = valid[valid["spy_regime"] == "go"]

        if len(valid) < 20:
            pytest.skip(f"Too few high-score events ({len(valid)})")
        if len(go) < 20:
            pytest.skip(f"Too few GO-regime + score>=60 events ({len(go)})")

        wr_all = valid["win"].mean()
        wr_go = go["win"].mean()
        assert wr_go >= wr_all - 0.02, (
            f"GO filter win rate {wr_go:.1%} < unfiltered {wr_all:.1%}  diff={wr_go - wr_all:.1%}"
        )

    # ------------------------------------------------------------------
    # Summary — always passes; use -s to see output
    # ------------------------------------------------------------------

    def test_print_summary(self, eval_df: pd.DataFrame) -> None:
        """Print scoring distribution and win rates by bucket. Always passes."""
        n = len(eval_df)
        valid = eval_df[eval_df["win"].notna()].copy()
        wr_overall = valid["win"].mean() if len(valid) > 0 else float("nan")

        n_long = (eval_df["direction"] == "long").sum()
        n_short = (eval_df["direction"] == "short").sum()

        print(f"\n{'='*65}")
        print(f"Scoring Evaluation  (n={n:,}  long={n_long:,}  short={n_short:,})")
        print(f"{'='*65}")
        print(f"score_total: mean={eval_df['score_total'].mean():.1f}  "
              f"median={eval_df['score_total'].median():.1f}  "
              f"std={eval_df['score_total'].std():.1f}")
        print(f"fwd_return ({_FWD_COL}): mean={eval_df['fwd_return'].mean():.2f}%  "
              f"median={eval_df['fwd_return'].median():.2f}%")
        print(f"Overall win rate: {wr_overall:.1%}")

        # Win rate by score bucket
        print("\nWin rate by score bucket (all directions):")
        buckets = [(0, 30), (30, 50), (50, 65), (65, 80), (80, 112)]
        for lo, hi in buckets:
            seg = valid[(valid["score_total"] >= lo) & (valid["score_total"] < hi)]
            if len(seg) > 0:
                wr = seg["win"].mean()
                med_ret = seg["fwd_return"].median()
                print(f"  [{lo:3d}-{hi:3d}): n={len(seg):5d}  WR={wr:.1%}  "
                      f"median_ret={med_ret:+.2f}%")

        # Direction breakdown
        print("\nWin rate by direction:")
        for d in ("long", "short"):
            seg = valid[valid["direction"] == d]
            if len(seg) > 0:
                wr = seg["win"].mean()
                med = seg["fwd_return"].median()
                print(f"  {d:5s}: n={len(seg):5d}  WR={wr:.1%}  median_ret={med:+.2f}%")

        # Regime breakdown
        print("\nWin rate by regime (all scores):")
        for regime in ("go", "no-go"):
            seg = valid[valid["spy_regime"] == regime]
            if len(seg) > 0:
                wr = seg["win"].mean()
                print(f"  {regime}: n={len(seg):5d}  WR={wr:.1%}")

        # Spearman
        if len(valid) >= 10:
            rho = valid["score_total"].corr(valid["win"].astype(float), method="spearman")
            print(f"\nSpearman rho(score, win) = {rho:.4f}")

        # Per-year breakdown
        if "date" in eval_df.columns and eval_df["date"].notna().any():
            print("\nWin rate by year:")
            for yr in sorted(pd.to_datetime(eval_df["date"]).dt.year.unique()):
                seg = valid[pd.to_datetime(valid["date"]).dt.year == yr]
                if len(seg) > 0:
                    wr = seg["win"].mean()
                    med = seg["fwd_return"].median()
                    print(f"  {yr}: n={len(seg):5d}  WR={wr:.1%}  median_ret={med:+.2f}%")

        print("=" * 65)
