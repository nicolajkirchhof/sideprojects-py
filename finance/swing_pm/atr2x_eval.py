#%%
import re

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import finance.utils as utils
import scipy.stats as stats

pd.options.plotting.backend = "matplotlib"
pd.set_option("display.max_columns", None)  # None means "no limit"
pd.set_option("display.max_rows", None)  # None means "no limit"
pd.set_option("display.width", 140)  # Set the width to 80 characters

# ... existing code ...
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import permutation_importance
from sklearn.inspection import PartialDependenceDisplay

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2


#%%
df_atr2x = pd.read_pickle(f'finance/_data/all_atr2x.pkl')

c_cols = df_atr2x.filter(regex=r"^c-?\d+$").columns

# 1. Basic Cleaning: Replace infinity and handle NaNs
df_atr2x.replace([np.inf, -np.inf], np.nan, inplace=True)

# 2. Filter for stocks and ensure c-columns are present
c_cols = df_atr2x.filter(regex=r"^c-?\d+$").columns
df_atr2x = df_atr2x.dropna(subset=list(c_cols))

# 3. Robust Data Cleaning for ML
# Cap extreme outliers that cause float32 overflow (common in rvol or pct columns)
numeric_cols = df_atr2x.select_dtypes(include=[np.number]).columns
for col_name in numeric_cols:
  upper_limit = df_atr2x[col_name].quantile(0.999)
  lower_limit = df_atr2x[col_name].quantile(0.001)
  df_atr2x[col_name] = df_atr2x[col_name].clip(lower_limit, upper_limit)


#%%
new_cols = {}
for i in range(-1, 21):
  new_cols[f'rperf{i}'] = df_atr2x[f'cpct{i}'] - df_atr2x[f'spy{i}']
# Join all at once
df_atr2x = pd.concat([df_atr2x, pd.DataFrame(new_cols, index=df_atr2x.index)], axis=1)

# Replace inf and -inf with NaN
df_atr2x.replace([np.inf, -np.inf], np.nan, inplace=True)
cols = ['c', 'cpct', 'v', 'atrp9', 'atrp14', 'atrp20', 'ac_lag_1', 'ac_lag_5', 'ac_lag_21', 'ac_comp', 'ac_mom', 'ac_mr', 'ac_inst',
        'pct', 'rvol20', 'rvol50', 'iv', 'hv9', 'hv14', 'hv20', 'hv30']

# Pre-calculate categories before calling meta analysis
df_atr2x= df_atr2x[df_atr2x.is_etf == 0].copy()
df_atr2x['mcap_bin'] = df_atr2x['market_cap'].map(utils.fundamentals.market_cap_classifier)
df_atr2x['mcap_bin'] = pd.Categorical(df_atr2x['mcap_bin'], categories=utils.fundamentals.MCAP_ORDER, ordered=True)

if 'c-1' in df_atr2x.columns:
  df_atr2x['price_bin'] = pd.qcut(df_atr2x['c-1'], q=3, labels=['Low', 'Medium', 'High'])

# Prepare RVOL categories for meta analysis
for rvol in  ['rvol200', 'rvol500']:
  if rvol in df_atr2x.columns:
      rvol_bins = [0, 1, 3, 10, df_atr2x[rvol].max()]
      rvol_labels = ['Low (<1x)', 'Normal (1-3x)', 'High (3-10x)', 'Extreme (>10x)']
      df_atr2x['vol_category'] = pd.cut(df_atr2x[rvol], bins=rvol_bins, labels=rvol_labels)

# 1. Defragment the DataFrame
df_atr2x = df_atr2x.copy()

df_atr2x.replace([np.inf, -np.inf], np.nan, inplace=True)
df_atr2x = df_atr2x.dropna(subset=[f'{c}0' for c in cols if f'{c}0' in df_atr2x.columns])

df_stocks = df_atr2x.copy()[df_atr2x.is_etf == 0]

# 1. Calculate 'ATR-Multiple' features
# This represents the move (pct0) as a multiple of the daily ATR percentages
for window in [9, 14, 20]:
    atr_col = f'atrp{window}0'
    if atr_col in df_stocks.columns:
        # actual_atrx = percentage_move / daily_atr_percentage
        df_stocks[f'actual_atr{window}x'] = df_stocks['pct0'] / df_stocks[atr_col]

# 2. Add these new features to the feature list
atr_x_features = [f'actual_atr{window}x' for window in [9, 14, 20] if f'actual_atr{window}x' in df_stocks.columns]
feature_cols = atr_x_features

# 3. Handle potential division by zero or extreme outliers from very low ATRs
df_stocks[atr_x_features] = df_stocks[atr_x_features].replace([np.inf, -np.inf], np.nan)

# 1. Target Definition
df_stocks['target'] = (df_stocks['c20'] > df_stocks['c0']).astype(int)

# 2. Identify and Encode Categorical Features
cat_cols = ['mcap_bin', 'price_bin', 'vol_category']

# Convert categories to their integer codes
for col in cat_cols:
    if col in df_stocks.columns:
        # Categorical codes preserve the order defined in utils.fundamentals
        df_stocks[f'{col}_code'] = df_stocks[col].cat.codes

# 3. Prepare Feature List
feature_cols += [f'{col}0' for col in cols if f'{col}0' in df_stocks.columns]
feature_cols += [f'{col}_code' for col in cat_cols if f'{col}_code' in df_stocks.columns]
feature_cols += ['earnings']

# --- DEFINE SPLITS ---
print(f"Splitting data ({len(df_stocks)} rows)...")
train_idx, test_idx = train_test_split(df_stocks.index, test_size=0.2, random_state=42)

X = df_stocks[feature_cols].fillna(df_stocks.loc[train_idx, feature_cols].median())
y = df_stocks['target']

X_train, X_test = X.loc[train_idx], X.loc[test_idx]
y_train, y_test = y.loc[train_idx], y.loc[test_idx]

# Train Initial Classifier
print("Training initial Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Importance Evaluation
importances = pd.DataFrame({
  'feature': feature_cols,
  'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("--- Feature Importance for Move Continuation ---")
print(importances.head(15))

# 1. Drop features with almost no variance
selector = VarianceThreshold(threshold=0.01)

# 2. Correlation Pruning Logic
def prune_features(df, features, threshold=0.85):
    print(f"Pruning correlated features from set of {len(features)}...")
    corr_matrix = df[features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return [f for f in features if f not in to_drop]

# Apply pruning to technical indicators (keep earnings/categories as they are structural)
technical_feats = [f for f in feature_cols if 'bin' not in f and 'category' not in f and 'earnings' not in f]
stable_tech_feats = prune_features(df_stocks, technical_feats)
final_feature_cols = stable_tech_feats + [f for f in feature_cols if f not in technical_feats]
print(f"Features remaining after pruning: {len(final_feature_cols)}")

# 3. Add Noise Baseline to check for "Real" signal
X = df_stocks[final_feature_cols].copy()
X['RANDOM_NOISE'] = np.random.normal(0, 1, X.shape[0])

# Re-run Imputation & Split
X_train = X.loc[train_idx].fillna(X.loc[train_idx].median())
X_test = X.loc[test_idx].fillna(X.loc[train_idx].median())

# 4. Use a more constrained "Stable" Forest
print("Training stable ensemble (500 trees)...")
rf = RandomForestClassifier(
    n_estimators=500,        # More trees = more stable averages
    max_features='sqrt',     # Force diversity in trees
    min_samples_leaf=20,     # Prevent overfitting to single trades
    max_samples=0.8,         # Bootstrap sampling for stability
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y.loc[train_idx])

# 5. Evaluate against Noise
print("Running Permutation Importance (this may take a minute)...")
perm_res = permutation_importance(rf, X_test, y.loc[test_idx], n_repeats=10, random_state=42, n_jobs=-1)
perm_df = pd.DataFrame({'feature': X.columns, 'imp': perm_res.importances_mean}).sort_values('imp', ascending=False)

noise_level = perm_df[perm_df['feature'] == 'RANDOM_NOISE']['imp'].values[0]
truly_stable_features = perm_df[perm_df['imp'] > noise_level]

print(f"--- Features beating Random Noise (Level: {noise_level:.4f}) ---")
print(truly_stable_features)

#%% Further use important features
# feature       imp
# 3         atrp90  0.024004
# 4      ac_lag_10  0.006591
# 0   actual_atr9x  0.004457
# 6     ac_lag_210  0.003844
# 1             c0  0.002985
# 7       ac_comp0  0.001080
# 8        ac_mom0  0.000694
# 5      ac_lag_50  0.000615
# 2             v0  0.000303
# 9       ac_inst0  0.000297
# 10       rvol200  0.000232
# 1. Visualize the "S-Curve" of the Top 3 Features
# This shows exactly where the 'inflection points' are for your criteria
top_3_features = truly_stable_features['feature'].head(3).tolist()
print(f"Generating Decision Criteria Plots for: {top_3_features}")

fig, ax = plt.subplots(figsize=(15, 5))
PartialDependenceDisplay.from_estimator(rf, X_train, top_3_features, ax=ax)
plt.suptitle('Criteria Influence: Impact of Feature Value on Continuation Probability')
plt.tight_layout()
plt.show()

# 2. Extract Concrete Statistical Thresholds
# Let's bucket the top feature and see the actual win rates
primary_feat = top_3_features[0]

# Create 10 deciles for the primary feature
df_stocks['feat_bucket'] = pd.qcut(df_stocks[primary_feat], q=10, duplicates='drop')
threshold_analysis = df_stocks.groupby('feat_bucket', observed=True)['target'].agg(['mean', 'count']).rename(columns={'mean': 'win_rate'})

print(f"\n--- Decision Thresholds for {primary_feat} ---")
print(threshold_analysis)

# 3. Plot the Criteria heatmap (Interaction between Top 2 features)
if len(top_3_features) >= 2:
  feat1, feat2 = top_3_features[0], top_3_features[1]

  # Create a 5x5 grid of probabilities
  df_stocks['f1_bin'] = pd.qcut(df_stocks[feat1], q=5, labels=False)
  df_stocks['f2_bin'] = pd.qcut(df_stocks[feat2], q=5, labels=False)

  pivot_table = df_stocks.pivot_table(index='f1_bin', columns='f2_bin', values='target', aggfunc='mean')

  plt.figure(figsize=(10, 8))
  sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap='RdYlGn')
  plt.title(f'Win Rate Heatmap: {feat1} vs {feat2}')
  plt.xlabel(f'{feat2} (Binned Low to High)')
  plt.ylabel(f'{feat1} (Binned Low to High)')
  plt.show()
