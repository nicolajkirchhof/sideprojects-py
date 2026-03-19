"""
finance.utils.move_character
============================
Instrument-agnostic analysis functions for understanding the movement character
of an underlying asset.  All functions are pure computation (no plotting) and
accept a df_day DataFrame produced by SwingTradingData.

Functions
---------
calculate_regime_filter_stats       — MA20/MA50 regime episodes + forward returns
calculate_move_magnitude_stats      — ATR-normalised move distribution + tail frequencies
calculate_intratrend_retracement    — Max retracement depth during ordered-MA trends
calculate_gap_intraday_decomposition— Overnight gap vs intraday range decomposition
calculate_hv_regime_stats           — HV regime classification + transition matrix
calculate_impulse_forward_returns   — Forward returns after ATR-impulse sessions (Block 5)
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Regime Filter
# ---------------------------------------------------------------------------

def calculate_regime_filter_stats(df, forward_windows=(5, 10, 20), recovery_window=10):
    """
    Classify daily bars into Uptrend / Pullback / Breakdown states relative to
    MA20 and MA50, then compute episode statistics and forward returns.

    Returns
    -------
    episodes : pd.DataFrame
        One row per episode with columns:
        state, start, end, duration, max_depth_pct, fwd_5, fwd_10, fwd_20, recovered
    summary : dict
        Aggregated stats per state for quick printing.
    """
    required = {'c', 'ma20', 'ma50'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")

    df = df.dropna(subset=list(required)).copy()

    above_ma20 = df['c'] >= df['ma20']
    above_ma50 = df['c'] >= df['ma50']

    conditions = [
        above_ma20,
        ~above_ma20 & above_ma50,
        ~above_ma20 & ~above_ma50,
    ]
    state_labels = ['Uptrend', 'Pullback', 'Breakdown']
    df['state'] = np.select(conditions, state_labels, default='Unknown')
    df['episode_id'] = (df['state'] != df['state'].shift()).cumsum()

    episodes = []
    for ep_id, group in df.groupby('episode_id'):
        state = group['state'].iloc[0]
        start = group.index[0]
        end   = group.index[-1]
        duration = len(group)

        depth_pct = ((group['c'] - group['ma20']) / group['ma20'] * 100).min()

        end_pos = df.index.get_loc(end)
        fwd = {}
        for w in forward_windows:
            future_pos = end_pos + w
            if future_pos < len(df):
                fwd[f'fwd_{w}'] = (df['c'].iloc[future_pos] / df['c'].iloc[end_pos] - 1) * 100
            else:
                fwd[f'fwd_{w}'] = np.nan

        recovered = False
        if state == 'Pullback':
            future_slice = df.iloc[end_pos + 1: end_pos + 1 + recovery_window]
            if not future_slice.empty:
                recovered = bool((future_slice['c'] >= future_slice['ma20']).any())

        episodes.append({
            'state': state, 'start': start, 'end': end,
            'duration': duration, 'max_depth_pct': depth_pct,
            **fwd, 'recovered': recovered,
        })

    episodes_df = pd.DataFrame(episodes)

    summary = {}
    for state in state_labels:
        sub = episodes_df[episodes_df['state'] == state]
        if sub.empty:
            summary[state] = {}
            continue
        entry = {
            'n_episodes':   len(sub),
            'med_duration': sub['duration'].median(),
            'med_depth_pct': sub['max_depth_pct'].median(),
        }
        for w in forward_windows:
            col = f'fwd_{w}'
            if col in sub.columns:
                entry[f'med_{col}']        = sub[col].median()
                entry[f'pct_positive_{col}'] = (sub[col] > 0).mean() * 100
        if state == 'Pullback':
            entry['recovery_rate_pct'] = sub['recovered'].mean() * 100
        summary[state] = entry

    return episodes_df, summary


# ---------------------------------------------------------------------------
# Block 1 — Move Magnitude Distribution
# ---------------------------------------------------------------------------

def calculate_move_magnitude_stats(df, atr_col='atrp20'):
    """
    Compute ATR-normalised move distributions and tail frequencies.

    Parameters
    ----------
    df      : pd.DataFrame — df_day with columns 'pct' and atr_col
    atr_col : str          — ATR column used as normaliser (default: atrp20)

    Returns
    -------
    norm_moves      : pd.Series  — signed ATR-normalised daily moves
    tail_freq       : dict       — {threshold: {'up': %, 'down': %, 'total': %}}
    excess_kurtosis : float      — excess kurtosis of raw pct returns (normal = 0)
    skewness        : float      — skewness of raw pct returns (normal = 0)
    """
    required = {'pct', atr_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for move magnitude analysis: {missing}")

    df = df.dropna(subset=list(required)).copy()
    norm_moves = df['pct'] / df[atr_col]

    thresholds = [1.0, 1.5, 2.0, 3.0]
    tail_freq = {}
    for t in thresholds:
        up   = float((norm_moves > t).mean() * 100)
        down = float((norm_moves < -t).mean() * 100)
        tail_freq[t] = {'up': up, 'down': down, 'total': up + down}

    pct = df['pct'].values
    mean_r, std_r = pct.mean(), pct.std(ddof=1)
    if std_r > 0:
        z = (pct - mean_r) / std_r
        excess_kurtosis = float(np.mean(z ** 4) - 3.0)
        skewness        = float(np.mean(z ** 3))
    else:
        excess_kurtosis = skewness = 0.0

    return norm_moves, tail_freq, excess_kurtosis, skewness


# ---------------------------------------------------------------------------
# Block 2 — Intra-Trend Retracement Depth
# ---------------------------------------------------------------------------

def calculate_intratrend_retracement(df, periods=(5, 10, 20)):
    """
    For each ordered-MA trend episode, compute the maximum intra-trend
    retracement from the running peak (longs) or trough (shorts).

    Parameters
    ----------
    df      : pd.DataFrame — df_day
    periods : tuple        — MA periods defining ordered trend, e.g. (5, 10, 20)

    Returns
    -------
    long_results  : pd.DataFrame — columns: duration, max_retracement_pct
    short_results : pd.DataFrame — columns: duration, max_retracement_pct
    """
    required = {f'ma{p}' for p in periods} | {'c'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for retracement analysis: {missing}")

    df = df.dropna(subset=list(required)).copy()

    long_cond  = pd.Series(True, index=df.index)
    short_cond = pd.Series(True, index=df.index)
    for i in range(len(periods) - 1):
        hi = df[f'ma{periods[i]}']
        lo = df[f'ma{periods[i + 1]}']
        long_cond  &= (hi > lo)
        short_cond &= (hi < lo)

    def _retracements(condition, direction):
        results = []
        ep_id = (condition != condition.shift()).cumsum()
        for gid, group in df[condition].groupby(ep_id[condition]):
            if len(group) < 2:
                continue
            close = group['c'].values
            if direction == 'long':
                peak = np.maximum.accumulate(close)
                max_retr = float(((close - peak) / peak * 100).min())
            else:
                # Short: max_retr is the maximum adverse *upward* move (positive = against trade)
                trough = np.minimum.accumulate(close)
                max_retr = float(((close - trough) / trough * 100).max())
            results.append({'duration': len(group), 'max_retracement_pct': max_retr})
        return pd.DataFrame(results) if results else pd.DataFrame(
            columns=['duration', 'max_retracement_pct'])

    return _retracements(long_cond, 'long'), _retracements(short_cond, 'short')


# ---------------------------------------------------------------------------
# Block 3 — Gap vs Intraday Range Decomposition
# ---------------------------------------------------------------------------

def calculate_gap_intraday_decomposition(df):
    """
    Decompose daily true range into overnight gap and intraday range components.
    Uses atrp1 (single-day ATR %) as true range reference.

    Returns
    -------
    result              : pd.DataFrame — per-bar metrics
    rolling_gap_contrib : pd.Series   — 63-day rolling mean gap contribution
    """
    required = {'o', 'h', 'l', 'c', 'atrp1'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for gap decomposition: {missing}")

    df = df.dropna(subset=list(required)).copy()
    prev_close = df['c'].shift(1)

    overnight_gap_pct  = (df['o'] - prev_close) / prev_close * 100
    intraday_range_pct = (df['h'] - df['l'])    / df['o']    * 100
    true_range_pct     = df['atrp1']

    gap_contrib = (overnight_gap_pct.abs() / true_range_pct.replace(0, np.nan)).clip(0, 1)
    gap_dir     = np.sign(overnight_gap_pct)
    close_dir   = np.sign(df['c'] - df['o'])
    gap_filled  = (gap_dir != 0) & (close_dir == -gap_dir)

    result = pd.DataFrame({
        'overnight_gap_pct':  overnight_gap_pct,
        'intraday_range_pct': intraday_range_pct,
        'true_range_pct':     true_range_pct,
        'gap_contrib':        gap_contrib,
        'gap_filled':         gap_filled,
        'gap_dir':            gap_dir,
    }).dropna()

    # Compute rolling window after dropna so index is aligned with result
    rolling_gap_contrib = result['gap_contrib'].rolling(63, min_periods=21).mean()

    return result, rolling_gap_contrib


# ---------------------------------------------------------------------------
# Block 4 — Realized Volatility Regime Analysis
# ---------------------------------------------------------------------------

def calculate_hv_regime_stats(df, hv_col='hv20'):
    """
    Classify realized volatility into Low / Medium / High regimes using
    adaptive rolling percentile thresholds.  Compute episode durations and
    transition probability matrix.

    Returns
    -------
    hv           : pd.Series     — HV series used (or None if unavailable)
    regime       : pd.Series     — 'Low' / 'Medium' / 'High' per bar
    episodes     : pd.DataFrame  — columns: regime, start, end, duration
    trans_pct    : pd.DataFrame  — transition probability matrix (%)
    hv_col_used  : str           — column actually used
    """
    for candidate in [hv_col, 'hvc', 'hv9']:
        if candidate in df.columns:
            hv_col = candidate
            break
    else:
        print('Warning: no HV column found in df — skipping HV regime analysis')
        return None, None, None, None, None

    hv = df[hv_col].dropna()
    if len(hv) < 63:
        print(f'Warning: insufficient HV data ({len(hv)} bars) — need at least 63')
        return None, None, None, None, hv_col

    low_thresh  = hv.rolling(252, min_periods=63).quantile(0.33)
    high_thresh = hv.rolling(252, min_periods=63).quantile(0.67)

    regime = pd.Series('Medium', index=hv.index, dtype=object)
    regime[hv <= low_thresh]  = 'Low'
    regime[hv >= high_thresh] = 'High'

    ep_id = (regime != regime.shift()).cumsum()
    episodes = []
    for _, group in regime.groupby(ep_id):
        episodes.append({
            'regime':   group.iloc[0],
            'start':    group.index[0],
            'end':      group.index[-1],
            'duration': len(group),
        })
    episodes_df = pd.DataFrame(episodes)

    from_r = pd.Series(regime.values[:-1], name='From')
    to_r   = pd.Series(regime.values[1:],  name='To')
    trans_pct = pd.crosstab(from_r, to_r, normalize='index') * 100
    for state in ['Low', 'Medium', 'High']:
        if state not in trans_pct.index:
            trans_pct.loc[state] = 0.0
        if state not in trans_pct.columns:
            trans_pct[state] = 0.0
    trans_pct = trans_pct.loc[['Low', 'Medium', 'High'], ['Low', 'Medium', 'High']]

    return hv, regime, episodes_df, trans_pct, hv_col


# ---------------------------------------------------------------------------
# Block 5 — Forward Returns after ATR-Impulse Sessions
# ---------------------------------------------------------------------------

def calculate_impulse_forward_returns(df, threshold=1.75, horizons=(5, 10, 20)):
    """
    Identify sessions where abs(daily return / ATR20) exceeds *threshold*
    (ATR-impulse sessions) and compute forward returns at each horizon.

    This directly validates the 1.75× ATR20 impulse exit rule against the
    specific instrument's history: were these sessions continuation events
    (exit correct) or exhaustion events (exit premature)?

    Parameters
    ----------
    df        : pd.DataFrame — df_day with 'pct' and 'atrp20' columns
    threshold : float        — ATR multiple defining an impulse (default: 1.75)
    horizons  : tuple        — forward look-ahead windows in trading days

    Returns
    -------
    result : pd.DataFrame — one row per impulse session
        Columns: direction ('up'/'down'), atr_ratio, fwd_5, fwd_10, fwd_20
    """
    required = {'pct', 'atrp20', 'c'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for impulse forward returns: {missing}")

    df = df.dropna(subset=list(required)).copy()
    atr_ratio = df['pct'] / df['atrp20']

    impulse_mask = atr_ratio.abs() > threshold

    rows = []
    for pos, (idx, ratio) in enumerate(atr_ratio[impulse_mask].items()):
        pos_in_df = df.index.get_loc(idx)
        direction = 'up' if ratio > 0 else 'down'
        row = {'direction': direction, 'atr_ratio': float(ratio)}
        for h in horizons:
            fpos = pos_in_df + h
            if fpos < len(df):
                row[f'fwd_{h}'] = float((df['c'].iloc[fpos] / df['c'].iloc[pos_in_df] - 1) * 100)
            else:
                row[f'fwd_{h}'] = np.nan
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=['direction', 'atr_ratio'] + [f'fwd_{h}' for h in horizons])
    return pd.DataFrame(rows)
