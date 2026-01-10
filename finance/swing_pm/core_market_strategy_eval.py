# %%
import itertools

import numpy as np
import os

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import finance.utils as utils
import seaborn as sns
from glob import glob

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler  # Better for pnl_pct outliers
from sklearn.inspection import permutation_importance

from sklearn.feature_selection import mutual_info_regression

pd.set_option("display.max_columns", None)  # None means "no limit"
pd.set_option("display.max_rows", None)  # None means "no limit"
pd.set_option("display.width", 140)  # Set the width to 80 characters

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2

# %%

## %% Statistical ML Influence Analysis
def analyze_indicator_influence_ml(df_eval, name):
  """
  Uses ML to rank indicators by their influence on PnL.
  Provides both Predictive Power (RF) and Directional Influence (Ridge).
  """
  # 1. Prepare Data
  # Define the indicator feature set

  features = ['gappct', 'hurst50', 'hurst100',
              'ema20_slope', 'ema50_slope', 'ema100_slope', 'ema200_slope',
              'ema20_dist', 'ema50_dist', 'ema100_dist', 'ema200_dist', 'atrp9', 'atrp14', 'atrp20', 'rvol20',
              'rvol50'] + df_eval.filter(regex='^ac').columns.tolist()


  # Filter for rows where we have both PnL and all indicators
  df_ml = df_eval[[*features, 'pnl_pct']].dropna()
  if len(df_ml) < 50:
    print(f"Not enough data for ML analysis in {name}")
    return

  X = df_ml[features]
  y = df_ml['pnl_pct']

  # Scale features for the Linear model comparison
  # Use RobustScaler instead of StandardScaler to handle pnl_pct extremes
  scaler = RobustScaler()
  X_scaled = scaler.fit_transform(X)

  # 2. Train Models
  # Random Forest for non-linear predictive power
  rf = RandomForestRegressor(n_estimators=100, random_state=42)
  rf.fit(X, y)

  # Ridge (Linear) for directional influence
  ridge = Ridge(alpha=1.0)
  ridge.fit(X_scaled, y)

  # 3. Calculate Permutation Importance (More robust than default RF importance)
  perm_importance = permutation_importance(rf, X, y, n_repeats=10, random_state=42)

  # 4. Visualization
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

  # Plot A: Predictive Power (Permutation Importance)
  importance_df = pd.Series(perm_importance.importances_mean, index=features).sort_values()
  importance_df.plot(kind='barh', ax=ax1, color='tab:blue', alpha=0.7)
  ax1.set_title(f"{name}: Predictive Influence (Non-Linear Power)")
  ax1.set_xlabel("Importance Score (Mean Decrease in Accuracy)")

  # Plot B: Directional Influence (Ridge Coefficients)
  # Since data was scaled, coefficient magnitude represents strength
  coeff_df = pd.Series(ridge.coef_, index=features).sort_values()
  colors = ['green' if x > 0 else 'red' for x in coeff_df]
  coeff_df.plot(kind='barh', ax=ax2, color=colors, alpha=0.7)
  ax2.set_title(f"{name}: Directional Influence (Standardized Coeffs)")
  ax2.set_xlabel("Impact on PnL (Positive = Help, Negative = Hurt)")
  ax2.axvline(0, color='black', lw=1)

  plt.tight_layout()
  plt.show()
  fig.savefig(f"finance/analysis/core/{df_eval.symbol.iat[0]}/features_{name}.png", bbox_inches='tight')

  # Include Mutual Information in the candidate selection
  # to catch non-linear regime interactions
  mi_scores = mutual_info_regression(X, y, random_state=42)
  top_mi = pd.Series(mi_scores, index=features).nlargest(11).index.tolist()

  top_predictive = pd.Series(perm_importance.importances_mean, index=features).nlargest(11).index.tolist()
  top_directional = pd.Series(ridge.coef_, index=features).abs().nlargest(11).index.tolist()

  # Union now includes MI to ensure "Hidden Regime" candidates are captured
  candidate_union = list(set(top_predictive) | set(top_directional) | set(top_mi))

  print(f"\nRecommended Candidates for Permutation Analysis ({name}):")
  print(candidate_union)

  return candidate_union


def evaluate_frequency_outcomes_yearly(df_filtered, freq_weeks=[1, 2, 3, 4]):
  """
  Calculates best/worst annual PnL % for various trading frequencies.
  Returns a dataframe grouped by year and frequency for bar plotting.
  """
  df_daily = df_filtered[['pnl_pct']].resample('D').sum().fillna(0)
  yearly_results = []

  for wk in freq_weeks:
    days_step = wk * 5
    for offset in range(min(len(df_daily), days_step)):
      sampled = df_daily.iloc[offset::days_step]
      # Sum PnL per year for this specific offset/frequency
      annual_pnl = sampled.groupby(sampled.index.year)['pnl_pct'].sum()
      for year, pnl in annual_pnl.items():
        yearly_results.append({
          'frequency': f'{wk}wk',
          'year': year,
          'pnl': pnl
        })

  df_all = pd.DataFrame(yearly_results)
  if df_all.empty: return pd.DataFrame()

  # Aggregate to find the luck-based bounds per year
  return df_all.groupby(['year', 'frequency'])['pnl'].agg(['max', 'min']).reset_index()


def shorten_regime_label(label):
  """Shortens long indicator names for plot readability."""
  replacements = {
    'ac_': 'a', 'ema': 'e', '_dist': 'd', '_slope': 's',
    'atrp': 'ap', 'hurst': 'h', 'iv_pct': 'ivp'
  }
  for old, new in replacements.items():
    label = label.replace(old, new)
  return label


def _calculate_margin_for_intervals(df_filtered, freq_weeks=[1, 2, 3, 4]):
  """
  Calculates cumulative margin requirements by selecting the longest trade
  available within each frequency interval.
  """
  interval_margins = {}

  for wk in freq_weeks:
    days_step = wk * 5
    # Resample to pick the most capital-intensive trade path for this frequency
    df_daily = df_filtered.resample('D').first().dropna(subset=['margin'])
    if df_daily.empty:
      interval_margins[f'{wk}wk'] = pd.Series()
      continue

    sampled_trades = df_daily.iloc[::days_step]

    margin_events = []
    for _, row in sampled_trades.iterrows():
      margin_events.append({'time': row.name, 'change': row.margin})
      margin_events.append({'time': row.date_closed, 'change': -row.margin})

    df_m = pd.DataFrame(margin_events).sort_values('time')
    df_m['total_margin'] = df_m['change'].cumsum()
    interval_margins[f'{wk}wk'] = df_m.set_index('time')['total_margin'].resample('D').max().ffill()

  return interval_margins


def analyze_feature_permutation_influence(df_eval, name, features, num_features=3):
  """
  Targeted complexity analysis with logic reduction report and full visual overview.
  """
  df = df_eval.copy().sort_index()
  baseline_avg_pct = df['pnl_pct'].mean()
  total_trades_baseline = len(df)
  results = []

  MIN_REGIME_SAMPLES = 10
  MIN_TOTAL_TRADES = 50

  print(f"Deep Targeted Evaluation: {name} - Complexity {num_features}...")

  for combo in itertools.combinations(features, num_features):
    temp_df = df.copy()
    regime_cols = []
    slots = []
    for feat in combo:
      reg_name = f"reg_{feat}"
      if 'hurst' in feat:
        states = [f"{feat}_MR", f"{feat}_TR"]
        temp_df[reg_name] = np.where(temp_df[feat] < 0.45, f"{feat}_MR", f"{feat}_TR")
      elif 'iv' in feat or 'atrp' in feat or 'rvol' in feat:
        med = temp_df[feat].median();
        states = [f"{feat}H", f"{feat}L"]
        temp_df[reg_name] = np.where(temp_df[feat] > med, f"{feat}H", f"{feat}L")
      else:
        states = [f"{feat}+", f"{feat}-"]
        temp_df[reg_name] = np.where(temp_df[feat] > 0, f"{feat}+", f"{feat}-")
      regime_cols.append(reg_name)
      slots.append(states)

    temp_df['reg_key'] = temp_df[regime_cols].agg(' | '.join, axis=1)
    reg_stats = temp_df.groupby('reg_key')['pnl_pct'].agg(['mean', 'count', 'std']).fillna(0)
    reg_stats['conservative_mean'] = reg_stats['mean'] - (reg_stats['std'] / np.sqrt(reg_stats['count']))

    keep_regimes = reg_stats[(reg_stats['conservative_mean'] > 0) & (reg_stats['count'] >= MIN_REGIME_SAMPLES)].index
    df_filtered = temp_df[temp_df['reg_key'].isin(keep_regimes)].copy()

    if len(df_filtered) >= MIN_TOTAL_TRADES:
      # Yearly Best/Worst bounds
      yearly_bounds = evaluate_frequency_outcomes_yearly(df_filtered)

      # Margin analysis
      margin_events = []
      for _, row in df_filtered.iterrows():
        margin_events.append({'time': row.name, 'change': row.margin})
        margin_events.append({'time': row.date_closed, 'change': -row.margin})
      df_margin = pd.DataFrame(margin_events).sort_values('time')
      df_margin['total_margin'] = df_margin['change'].cumsum()
      peak_margin = df_margin.set_index('time')['total_margin'].resample('D').max().fillna(0)

      res_item = {
        'features': combo, 'regimes': keep_regimes.tolist(), 'slots': slots,
        'avg_pnl_pct': df_filtered['pnl_pct'].mean(), 'trade_count': len(df_filtered),
        'score': df_filtered['pnl_pct'].mean() * np.sqrt(len(df_filtered)),
        'filter_rate': 1 - (len(df_filtered) / total_trades_baseline),
        'df_filtered': df_filtered, 'yearly_bounds': evaluate_frequency_outcomes_yearly(df_filtered)
      }
      results.append(res_item)

  if not results: return pd.DataFrame()

  df_res = pd.DataFrame(results).sort_values('score', ascending=False)
  best_res = df_res.iloc[0]

  logic_str = _reduce_logic(best_res['regimes'], best_res['slots'])

  # Pass symbol (global) to the plotter and saver
  fig = _plot_permutation_results(df_res, best_res, df, name, num_features, logic_str)
  report_text = _print_strategy_summary(name, logic_str, best_res, baseline_avg_pct)

  # Apply Saving Mechanism
  _save_permutation_results(symbol, name, num_features, fig, df_res, report_text)

  return df_res


def _discretize_features(df, combo):
  """Encapsulates the discretization logic for regimes and reduction slots."""
  temp_df = df.copy()
  regime_cols = [];
  slots = []
  for feat in combo:
    reg_name = f"reg_{feat}"
    if 'hurst' in feat:
      states = [f"{feat}_H", f"{feat}_L"]
      temp_df[reg_name] = np.where(temp_df[feat] < 0.45, f"{feat}_L", f"{feat}_H")
    elif any(x in feat for x in ['rvol']):
      states = [f"{feat}H", f"{feat}L"]
      temp_df[reg_name] = np.where(temp_df[feat] > 1.1, f"{feat}H", f"{feat}L")
    elif any(x in feat for x in ['ac_', 'slope', 'dist']):
      states = [f"{feat}+", f"{feat}-"]
      temp_df[reg_name] = np.where(temp_df[feat] > 0, f"{feat}+", f"{feat}-")
    elif 'iv' in feat or 'atrp' in feat:
      med = temp_df[feat].median();
      states = [f"{feat}H", f"{feat}L"]
      temp_df[reg_name] = np.where(temp_df[feat] > med, f"{feat}H", f"{feat}L")
    else:
      states = [f"{feat}P", f"{feat}N"]
      temp_df[reg_name] = np.where(temp_df[feat] > 0, f"{feat}P", f"{feat}N")
    regime_cols.append(reg_name)
    slots.append(states)
  return temp_df, regime_cols, slots


def _reduce_logic(regimes, slots):
  """Applies recursive reduction to simplify vertical divider logic."""
  current_rules = [{'pattern': k.split(' | ')} for k in regimes]
  num_feats = len(slots)

  def simplify_step(rules, current_slots):
    new_rules = [];
    used = set()
    for i in range(len(rules)):
      for j in range(i + 1, len(rules)):
        r1, r2 = rules[i], rules[j]
        diff = [idx for idx in range(num_feats) if r1['pattern'][idx] != r2['pattern'][idx]]
        if len(diff) == 1:
          idx = diff[0]
          if r1['pattern'][idx] != '*' and r2['pattern'][idx] != '*' and \
              r1['pattern'][idx] in current_slots[idx] and r2['pattern'][idx] in current_slots[idx]:
            merged = list(r1['pattern']);
            merged[idx] = '*'
            if merged not in [nr['pattern'] for nr in new_rules]: new_rules.append({'pattern': merged})
            used.add(i);
            used.add(j)
    return new_rules, [rules[k] for k in range(len(rules)) if k not in used]

  reduced = current_rules
  while True:
    combined, unique = simplify_step(reduced, slots)
    if not combined: break
    reduced = combined + unique
  return " || ".join([" | ".join(r['pattern']) for r in reduced])


def _plot_permutation_results(df_res, best_res, df_baseline, name, num_features, logic_str):
  """Orchestrates the visualization dashboard for the targeted complexity level."""
  fig = plt.figure(figsize=(28, 22))
  gs = fig.add_gridspec(4, 3, height_ratios=[1, 1.2, 1.2, 1.2])

  ax_box = fig.add_subplot(gs[0, 0]);
  ax_scatter = fig.add_subplot(gs[0, 1])
  ax_margin = fig.add_subplot(gs[0, 2]);
  ax_freq = fig.add_subplot(gs[1, :])
  ax_equity = fig.add_subplot(gs[2, :]);
  ax_violin = fig.add_subplot(gs[3, :])

  # 1. Efficiency vs Baseline Dist
  baseline_avg = df_baseline['pnl_pct'].mean()
  plot_data = pd.DataFrame({'Strategy': df_res['avg_pnl_pct'], 'Baseline Trades': df_baseline['pnl_pct']})
  sns.boxenplot(data=plot_data, ax=ax_box, palette='Set2')
  ax_box.axhline(baseline_avg, color='red', ls='--');
  ax_box.set_title("Strategy vs Baseline Distribution")

  # 2. Efficiency vs Frequency (Scatter)
  sns.scatterplot(x='trade_count', y='avg_pnl_pct', data=df_res, ax=ax_scatter, color='tab:blue', alpha=0.6)
  ax_scatter.scatter(best_res['trade_count'], best_res['avg_pnl_pct'], color='red', s=100, label='Best Score')
  ax_scatter.set_title("Efficiency vs Frequency Trade-off");
  ax_scatter.grid(True, alpha=0.2);
  ax_scatter.legend()

  # 3. Multi-Interval Margin Utilization
  margins = _calculate_margin_for_intervals(best_res['df_filtered'])
  for label, m_series in margins.items():
    if not m_series.empty:
      ax_margin.plot(m_series.index, m_series, label=label, alpha=0.8, drawstyle='steps-post')
  ax_margin.set_title("Capital Required by Interval");
  ax_margin.set_ylabel("Margin ($)")
  ax_margin.legend(loc='upper left', fontsize=8);
  ax_margin.grid(True, alpha=0.2)

  # 4. Yearly Stability Bar Chart
  df_f = best_res['yearly_bounds']
  if not df_f.empty:
    years = sorted(df_f['year'].unique());
    freqs = df_f['frequency'].unique()
    x = np.arange(len(years));
    width = 0.8 / len(freqs)
    for i, fq in enumerate(freqs):
      d = df_f[df_f['frequency'] == fq].set_index('year').reindex(years)
      off = (i - len(freqs) / 2 + 0.5) * width
      ax_freq.bar(x + off, d['max'], width, label=f'{fq} Best', color=plt.cm.tab10(i), alpha=0.8)
      ax_freq.bar(x + off, d['min'], width, label=f'{fq} Worst', color=plt.cm.tab10(i), alpha=0.3, edgecolor='black',
                  hatch='//')
    ax_freq.set_xticks(x);
    ax_freq.set_xticklabels(years);
    ax_freq.legend(loc='upper left', ncol=len(freqs))
    ax_freq.set_title("Annual Range by Trading Interval (Luck-sensitivity)")

  # 5. Continuous Equity path
  ax_equity.plot(df_baseline.index, df_baseline['pnl_pct'].cumsum(), label='Baseline (All)', color='black', alpha=0.3)
  ax_equity.plot(df_baseline.index, best_res['df_filtered']['pnl_pct'].reindex(df_baseline.index).fillna(0).cumsum(),
                 label='Filtered Strategy', color='tab:blue', lw=2.5)
  ax_equity.set_title(f"Cumulative path: {' + '.join(best_res['features'])}");
  ax_equity.legend()

  # 6. Logic components using custom violin function with shortened labels
  df_comp = pd.DataFrame({shorten_regime_label(r): best_res['df_filtered'][best_res['df_filtered']['reg_key'] == r][
    'pnl_pct'].reset_index(drop=True)
                          for r in best_res['regimes']})
  utils.plots.violinplot_columns_with_labels(df_comp, rotate=0, ax=ax_violin, title="Logic Component Distribution")

  plt.suptitle(f"STRATEGY: {name} | COMPLEXITY: {num_features} FEATURES", fontsize=8, fontweight='bold')
  plt.tight_layout()

  # Return the fig object so it can be saved by the orchestrator
  current_fig = plt.gcf()
  plt.show()
  return current_fig


def _print_strategy_summary(name, logic_str, best_res, baseline_avg):
  """Generates and prints the structured analysis report."""
  report = [
    "=" * 60,
    f"ANALYSIS REPORT: {name}",
    "=" * 60,
    f"BEST FEATURES : {' + '.join(best_res['features'])}",
    f"LOGIC SUMMARY : IF {logic_str}",
    "--- PERFORMANCE ---",
    f"EFFICIENCY    : {best_res['avg_pnl_pct']:.2f}% per trade",
    f"BASELINE MEAN : {baseline_avg:.2f}% per trade",
    "--- RELIABILITY ---",
    f"TRADE VOLUME  : {best_res['trade_count']} trades found",
    f"FILTER RATE   : {best_res['filter_rate']:.1%} of baseline trades removed",
    "=" * 60, "\n"
  ]
  summary_text = "\n".join(report)
  print(summary_text)
  return summary_text


def _save_permutation_results(symbol, name, num_features, fig, df_res, summary_text):
  """Saves plot, data, and textual report to the finance/analysis/core/ directory."""
  base_path = f"finance/analysis/core/{symbol}/{name.replace(' ', '_')}"
  os.makedirs(base_path, exist_ok=True)

  prefix = f"{num_features}feat"

  # 1. Save Figure
  fig.savefig(f"{base_path}/{prefix}_analysis.png", bbox_inches='tight')

  # 2. Save Data
  # Drop complex objects like DataFrames before saving to CSV
  df_save = df_res.drop(columns=['df_filtered', 'yearly_bounds', 'slots'], errors='ignore')
  df_save.to_csv(f"{base_path}/{prefix}_metrics.csv", index=False)

  # 3. Save Printout
  with open(f"{base_path}/{prefix}_report.txt", "w") as f:
    f.write(summary_text)

  print(f"Results saved to: {base_path}")


#%%
# symbol = 'SPY'
# symbol = 'QQQ'
# symbol = 'IWM'

for symbol in ['SPY', 'QQQ', 'IWM']:
  df_barchart = utils.swing_trading_data.SwingTradingData(symbol, datasource='barchart')

  ## %%
  def prepare_optionstrat_data(name):
    df = pd.read_csv(name)
    df = df.rename(columns={'P/L': 'pnl', 'P/L %': 'pnl_pct', 'Reason For Close': 'reason_close', 'Date Opened': 'date',
                            'Margin Req.': 'margin', 'Date Closed': 'date_closed'})[
      ['pnl', 'pnl_pct', 'reason_close', 'date', 'margin', 'date_closed']]
    df['date'] = pd.to_datetime(df.date)
    df['date_closed'] = pd.to_datetime(df.date_closed)
    df.set_index('date', inplace=True)
    df = df[df.reason_close != 'Backtest Completed']
    df_eval = pd.merge(df, df_barchart.df_day.copy(), left_index=True, right_index=True, how='inner')
    return df_eval


  ## %%
  strategy_list = []
  for tradeType in ['NP', 'C', 'NC', 'P']:
    name = glob(f'finance/_data/optionsomega/{symbol}-{tradeType}*.csv')
    df_eval = prepare_optionstrat_data(name[0])
    strategy_list.append((df_eval, f'{tradeType} Strategy'))

  ## %% Execute ML Influence Analysis and collect candidates
  all_strategy_candidates = []

  for df_eval, name in strategy_list:
    strategy_candidates = analyze_indicator_influence_ml(df_eval, name)
    if strategy_candidates:
      all_strategy_candidates.extend(strategy_candidates)

  # Consolidate Final Candidate List
  final_candidates = sorted(list(set(all_strategy_candidates)))

  print("\n" + "=" * 50)
  print("FINAL CONSOLIDATED CANDIDATES FOR SIMPLIFIED LOGIC")
  print("=" * 50)
  for i, feat in enumerate(final_candidates, 1):
    print(f"{i}. {feat}")
  print("=" * 50)

  # # %% Execute permutation search using consolidated candidates
  # # analyze_feature_permutation_influence(*strategy_list[0], final_candidates, num_features=2)
  # # analyze_feature_permutation_influence(*strategy_list[1], final_candidates, num_features=3)
  # analyze_feature_permutation_influence(*strategy_list[2], final_candidates, num_features=2)

  ## %%
  for num_features in range(1, 5):
    for df_eval, name in strategy_list:
      analyze_feature_permutation_influence(df_eval, name, final_candidates, num_features=num_features)
  plt.close('all')
