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
calculate_ivp                       — Implied-volatility percentile (rolling)
calculate_vrp                       — Variance risk premium (IV − HV) + rolling percentile
calculate_time_underwater           — Days between ATH and next ATH (drawdown recovery)
calculate_otm_viability             — Kurtosis / upper-tail frequency vs Gaussian → OTM-option flag
calculate_rs_vs_spy                 — Relative-strength line and 63d RS percentile vs SPY
calculate_vcp_tightness             — Rolling range% tightness + volume dry-up + conditional fwd returns
calculate_overnight_reversal        — PM-08: overnight return bias after large intraday drops
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

        # Forward return measured from the START of the episode — "if you
        # observe this state today, what typically happens over the next N days".
        # Measuring from the end is biased: end-of-Uptrend is by construction
        # the last day before price fell below MA20.
        start_pos = df.index.get_loc(start)
        end_pos   = df.index.get_loc(end)
        fwd = {}
        for w in forward_windows:
            future_pos = start_pos + w
            if future_pos < len(df):
                fwd[f'fwd_{w}'] = (df['c'].iloc[future_pos] / df['c'].iloc[start_pos] - 1) * 100
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


# ---------------------------------------------------------------------------
# IV Percentile (IVP)
# ---------------------------------------------------------------------------

def calculate_ivp(iv_series: pd.Series, window: int = 252) -> pd.Series:
    """
    Rolling IV percentile: % of prior `window` observations with IV ≤ current IV.
    Returns values in [0, 100].
    """
    iv = iv_series.dropna()
    if iv.empty:
        return pd.Series(dtype=float)
    # rank(pct=True) inside a rolling window — last value is current
    def _pct_rank(x):
        return (x <= x.iloc[-1]).mean() * 100.0
    return iv.rolling(window=window, min_periods=max(20, window // 5)).apply(_pct_rank, raw=False)


# ---------------------------------------------------------------------------
# Variance Risk Premium (VRP)
# ---------------------------------------------------------------------------

def calculate_vrp(iv_series: pd.Series, hv_series: pd.Series, window: int = 252) -> pd.DataFrame:
    """
    VRP spread (IV − HV) plus its rolling percentile.
    Returns DataFrame with columns ['vrp', 'vrp_pct'].
    """
    df = pd.concat([iv_series.rename('iv'), hv_series.rename('hv')], axis=1).dropna()
    if df.empty:
        return pd.DataFrame(columns=['vrp', 'vrp_pct'])
    df['vrp'] = df['iv'] - df['hv']
    def _pct_rank(x):
        return (x <= x.iloc[-1]).mean() * 100.0
    df['vrp_pct'] = df['vrp'].rolling(window=window, min_periods=max(20, window // 5)).apply(_pct_rank, raw=False)
    return df[['vrp', 'vrp_pct']]


# ---------------------------------------------------------------------------
# Time Underwater (drawdown recovery)
# ---------------------------------------------------------------------------

def calculate_time_underwater(close: pd.Series) -> pd.Series:
    """
    For each drawdown episode (close < prior ATH until the close reclaims it),
    return the episode length in trading days. A never-recovered final episode
    is included with its partial length.
    """
    c = close.dropna()
    if c.empty:
        return pd.Series(dtype=float)
    ath = c.cummax()
    underwater = c < ath
    episodes = []
    run = 0
    for flag in underwater.values:
        if flag:
            run += 1
        elif run > 0:
            episodes.append(run)
            run = 0
    if run > 0:
        episodes.append(run)
    return pd.Series(episodes, dtype=float, name='time_underwater')


# ---------------------------------------------------------------------------
# OTM Long-Option Viability
# ---------------------------------------------------------------------------

def calculate_otm_viability(returns: pd.Series) -> dict:
    """
    Evaluate whether the return distribution supports long OTM options.

    Fat tails (upside + excess kurtosis) are required for OTM longs to be
    profitable in expectation vs a Gaussian baseline.

    Returns dict:
      kurtosis        : pandas excess kurtosis
      skew            : pandas skew
      upper_tail_2    : % of returns > 2σ
      upper_tail_3    : % of returns > 3σ
      gauss_tail_2    : Gaussian baseline % > 2σ (=2.275)
      gauss_tail_3    : Gaussian baseline % > 3σ (=0.135)
      flag            : 'VIABLE' | 'MARGINAL' | 'POOR'
    """
    r = returns.dropna()
    if len(r) < 50:
        return {
            'kurtosis': np.nan, 'skew': np.nan,
            'upper_tail_2': np.nan, 'upper_tail_3': np.nan,
            'gauss_tail_2': 2.275, 'gauss_tail_3': 0.135,
            'flag': 'POOR',
        }
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    z = (r - mu) / sd if sd > 0 else r * 0.0
    upper_2 = float((z > 2).mean() * 100)
    upper_3 = float((z > 3).mean() * 100)
    kurt = float(r.kurtosis())  # pandas: excess kurtosis
    skew = float(r.skew())

    # VIABLE: meaningfully fatter than Gaussian on both tails + non-negative skew
    # POOR: both tails at or below Gaussian
    viable = (upper_2 >= 3.5) and (upper_3 >= 0.4) and (kurt >= 2.0) and (skew >= -0.3)
    poor   = (upper_2 < 2.5) and (upper_3 < 0.2) and (kurt < 1.0)
    flag = 'VIABLE' if viable else ('POOR' if poor else 'MARGINAL')

    return {
        'kurtosis': kurt, 'skew': skew,
        'upper_tail_2': upper_2, 'upper_tail_3': upper_3,
        'gauss_tail_2': 2.275, 'gauss_tail_3': 0.135,
        'flag': flag,
    }


# ---------------------------------------------------------------------------
# Relative Strength vs SPY
# ---------------------------------------------------------------------------

def calculate_rs_vs_spy(close: pd.Series, spy_close: pd.Series, window: int = 63) -> pd.DataFrame:
    """
    RS line = close / spy_close (normalised to 1.0 at start) and a rolling
    `window`-day percentile of that line.

    Returns DataFrame indexed by date with columns ['rs', 'rs_pct'].
    """
    df = pd.concat([close.rename('c'), spy_close.rename('spy')], axis=1).dropna()
    if df.empty:
        return pd.DataFrame(columns=['rs', 'rs_pct'])
    df['rs'] = (df['c'] / df['spy'])
    df['rs'] = df['rs'] / df['rs'].iloc[0]
    def _pct_rank(x):
        return (x <= x.iloc[-1]).mean() * 100.0
    df['rs_pct'] = df['rs'].rolling(window=window, min_periods=max(10, window // 3)).apply(_pct_rank, raw=False)
    return df[['rs', 'rs_pct']]


# ---------------------------------------------------------------------------
# VCP Tightness (Volatility Contraction Pattern)
# ---------------------------------------------------------------------------

def calculate_vcp_tightness(df: pd.DataFrame, window: int = 10,
                             fwd_horizons: tuple = (5, 10, 20)) -> dict:
    """
    PM-01 support: rolling range-contraction detector.

    tightness = (rolling_max(h) - rolling_min(l)) / close * 100
    vol_ratio = rolling mean volume / longer-window mean volume (dry-up signal)

    Returns dict with:
      timeline     : DataFrame [tightness, vol_ratio] indexed by date
      current      : dict {tightness, vol_ratio, tight_pct_rank}
      conditional  : DataFrame — median fwd return by tight/loose tercile
                     (index: horizon, columns: Tight/Mid/Loose)
    """
    required = {'h', 'l', 'c', 'v'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for VCP tightness: {missing}")

    df = df.dropna(subset=list(required)).copy()
    roll_hi = df['h'].rolling(window).max()
    roll_lo = df['l'].rolling(window).min()
    tight = ((roll_hi - roll_lo) / df['c'] * 100).rename('tightness')

    vol_short = df['v'].rolling(window).mean()
    vol_long  = df['v'].rolling(window * 5).mean()
    vol_ratio = (vol_short / vol_long).rename('vol_ratio')

    timeline = pd.concat([tight, vol_ratio], axis=1)

    # Current state + historical percentile rank (lower tightness = tighter)
    cur_tight = float(tight.iloc[-1]) if not tight.empty else float('nan')
    cur_vol   = float(vol_ratio.iloc[-1]) if not vol_ratio.empty else float('nan')
    tight_clean = tight.dropna()
    tight_pct_rank = float((tight_clean <= cur_tight).mean() * 100) if not tight_clean.empty else float('nan')

    current = {
        'tightness': cur_tight,
        'vol_ratio': cur_vol,
        'tight_pct_rank': tight_pct_rank,  # 0 = tightest ever, 100 = loosest
    }

    # Conditional forward returns by tightness tercile
    cond_rows = {}
    if len(tight_clean) > 30:
        q33, q66 = tight_clean.quantile([1/3, 2/3]).values
        tight_label = pd.Series(index=df.index, dtype=object)
        tight_label[tight <= q33] = 'Tight'
        tight_label[(tight > q33) & (tight <= q66)] = 'Mid'
        tight_label[tight > q66] = 'Loose'

        for h in fwd_horizons:
            fwd = (df['c'].shift(-h) / df['c'] - 1) * 100
            row = {}
            for bucket in ['Tight', 'Mid', 'Loose']:
                mask = tight_label == bucket
                vals = fwd[mask].dropna()
                row[bucket] = float(vals.median()) if not vals.empty else float('nan')
            cond_rows[f'fwd_{h}'] = row

    conditional = pd.DataFrame(cond_rows).T if cond_rows else pd.DataFrame()

    return {
        'timeline': timeline,
        'current': current,
        'conditional': conditional,
    }


# ---------------------------------------------------------------------------
# Overnight Reversal (PM-08)
# ---------------------------------------------------------------------------

def calculate_overnight_reversal(df: pd.DataFrame, atr_col: str = 'atrp20',
                                  drop_buckets: tuple = (1.0, 2.0, 3.0),
                                  rolling_window: int = 60) -> dict:
    """
    Quantify the overnight reversal edge (PM-08).

    For each session, compute:
      intraday_pct = close-to-close return
      drop_atr     = |negative intraday_pct| / atrp20  (0 on up days)
      overnight    = next_open / close - 1
    Then bucket by drop_atr magnitude and compare overnight stats to the baseline.

    Parameters
    ----------
    df             : df_day with 'o', 'c', 'pct' (or computable), atr_col
    drop_buckets   : ATR-multiple thresholds defining Drop>1, Drop>2, Drop>3, …
    rolling_window : days for the rolling median overnight-after-drop edge

    Returns
    -------
    dict with:
      stats       : DataFrame indexed by bucket (Baseline, Drop>1, …) with columns
                    [n, median, mean, hit_rate]
      rolling     : Series — rolling median overnight return on drop>2 days (rolling_window)
                    minus rolling median overnight return on baseline (non-drop) days
      flag        : 'REVERSAL' | 'NEUTRAL' | 'CONTINUATION' — simple classification of
                    the strongest-drop bucket vs baseline
    """
    required = {'o', 'c', atr_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for overnight reversal: {missing}")

    df = df.dropna(subset=['o', 'c', atr_col]).copy()
    # intraday close-to-close % — use pct if present, otherwise compute
    intraday = df['pct'] if 'pct' in df.columns else (df['c'].pct_change() * 100)
    intraday = intraday.reindex(df.index)

    # overnight return from today's close to tomorrow's open
    overnight = (df['o'].shift(-1) / df['c'] - 1) * 100

    drop_atr = np.where(intraday < 0, (-intraday) / df[atr_col], 0.0)
    work = pd.DataFrame({
        'intraday': intraday,
        'overnight': overnight,
        'drop_atr': drop_atr,
    }).dropna()

    def _stats(vals: pd.Series) -> dict:
        v = vals.dropna()
        return {
            'n':        int(len(v)),
            'median':   float(v.median())   if len(v) else float('nan'),
            'mean':     float(v.mean())     if len(v) else float('nan'),
            'hit_rate': float((v > 0).mean() * 100) if len(v) else float('nan'),
        }

    baseline_mask = work['drop_atr'] < drop_buckets[0]
    rows = {'Baseline': _stats(work.loc[baseline_mask, 'overnight'])}
    for thresh in drop_buckets:
        label = f'Drop>{thresh:g}'
        mask = work['drop_atr'] >= thresh
        rows[label] = _stats(work.loc[mask, 'overnight'])
    stats = pd.DataFrame(rows).T

    # Rolling edge: median overnight after drop>2 − baseline median overnight
    strong_thresh = drop_buckets[min(1, len(drop_buckets) - 1)]  # default 2×
    mask_strong   = work['drop_atr'] >= strong_thresh
    strong_series = work['overnight'].where(mask_strong)
    base_series   = work['overnight'].where(baseline_mask)
    rolling = (
        strong_series.rolling(rolling_window, min_periods=max(5, rolling_window // 4)).median()
        - base_series.rolling(rolling_window, min_periods=max(5, rolling_window // 4)).median()
    )

    # Flag: strongest bucket median vs baseline median
    strongest_label = f'Drop>{drop_buckets[-1]:g}'
    base_med = stats.loc['Baseline', 'median']
    strong_med = stats.loc[strongest_label, 'median']
    if pd.isna(strong_med) or stats.loc[strongest_label, 'n'] < 20:
        flag = 'NEUTRAL'
    elif strong_med - base_med > 0.10:
        flag = 'REVERSAL'
    elif strong_med - base_med < -0.10:
        flag = 'CONTINUATION'
    else:
        flag = 'NEUTRAL'

    return {'stats': stats, 'rolling': rolling, 'flag': flag}
