import numpy as np
import pandas as pd
import matplotlib as mpl

# --- CRITICAL FIX: Set backend BEFORE importing pyplot ---
try:
    mpl.use('QtAgg')  # Preferred for interactive windows
except ImportError:
    mpl.use('TkAgg')  # Fallback

import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, RangeSlider, Slider, Button
import matplotlib.gridspec as gridspec
import seaborn as sns

%load_ext autoreload
%autoreload 2

# Configure plot style
plt.style.use('dark_background')
sns.set_palette("viridis")

#%%
# The dataset has the following columns 'symbol', 'earnings', 'date', 'gappct', 'c', 'is_etf', 'atrp20',
#   '1M', '1M_chg', '3M', '3M_chg', '6M', '6M_chg', 'market_cap', 'mcap_class', 'atrp_change'
#   mcap_class contains one the following values ['Large-Cap', 'Mid-Cap', 'Small-Cap', 'Micro-Cap']
# In addition, the following columns track changes before and after the event they are tracked
# as {name}XX and w_{name}XX for daily and weekly values before and after the event.
#   Daily columns are from -25 to 24 whereas 0 is the breakout day
#   Weekly columns are from -8 to 8 whereas 0 is the breakout day
# Tracked names
#   'c' => close, 'spy' => spy changes, v => volume, atrp9/14/20 => ATR percentage,
#   ac100_lag_1/5/20 => autocorrelation 100day lag 1/5/20, ac_comp => Composite Swing Signal (20-day, Avg Lags 1-3)
#   ac_mom => Standard Momentum (20-day, Lag-1), ac_mr => Short-Term Mean Reversion (10-day, Lag-1)
#   ac_inst => Institutional "Hidden" Momentum (60-day, Lag-5), pct => Percent Change, std_mv => 20 day standard deviation
#   rvol20/50 => Relative Volatility 20/50-day, iv => implied volatility,
#   hv9/14/20/50 => Historical Volatility 9/14/20/50-day, ema10/20_dist => EMA 10/20 distance
#   ema10/20_slope => EMA 10/20 slope, cpct => Percentage change in reference to breakout point
#%%
def load_and_prep_data():
    """Loads and standardizes the dataset for the dashboard."""
    print("Loading data...")
    df = pd.read_pickle(f'finance/_data/all_clean_atr_x.pkl')
    df = df.reset_index(drop=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 1. Clean Path Columns
    c_cols = df.filter(regex=r"^c-?\d+$").columns
    df = df.dropna(subset=list(c_cols))

    # 2. Calculate ATRP Change
    if 'atrp_change' not in df.columns:
        df['atrp_change'] = df['cpct0'] / df['atrp200']

    # 3. Create Absolute Strength Column for Filtering
    df['atrp_change_abs'] = df['atrp_change'].abs()

    # Ensure date is properly typed for filtering
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    # Ensure mcap_class is string to handle Categorical issues in widgets
    df['mcap_class'] = df['mcap_class'].astype(str)

    # 5. Earnings Boolean
    if 'earnings' in df.columns:
        df['has_earnings'] = df['earnings'].fillna(False).astype(bool)
    else:
        df['has_earnings'] = False

    # 6. Directional Alignment
    df['direction'] = np.sign(df['atrp_change'])

    # 7. SPY Context Classification (Supporting/Neutral/Non-Supporting)
    # We use the 1-week post-breakout move (Day 5 - Day 0) to classify the environment
    if 'spy0' in df.columns and 'spy5' in df.columns:
        spy_change_1w = df['spy5'] - df['spy0']
        # Align: Positive means "Moving in favor of the trade"
        aligned_spy = spy_change_1w * df['direction']
        
        conditions = [
            aligned_spy > 0.25,   # Supporting
            aligned_spy < -0.25   # Non-Supporting
        ]
        choices = ['Supporting', 'Non-Supporting']
        df['spy_class'] = np.select(conditions, choices, default='Neutral')
    else:
        df['spy_class'] = 'Unknown'

    # --- ALIGN DAILY DATA (1 to 24) ---
    new_cols = {}
    future_days = range(1, 25)
    for d in future_days:
        col = f'cpct{d}'
        if col in df.columns:
            new_cols[f'aligned_{col}'] = df[col] * df['direction']

        for ema in ['ema10', 'ema20']:
            col = f'{ema}_dist{d}'
            if col in df.columns:
                new_cols[f'aligned_{col}'] = df[col] * df['direction']

    # --- ALIGN WEEKLY DATA (1 to 8) ---
    future_weeks = range(1, 9)
    for w in future_weeks:
        col = f'w_cpct{w}'
        if col in df.columns:
            new_cols[f'aligned_{col}'] = df[col] * df['direction']

        for ema in ['ema10', 'ema20']:
            col = f'w_{ema}_dist{w}'
            if col in df.columns:
                new_cols[f'aligned_{col}'] = df[col] * df['direction']

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)

    # 8. Clip outliers for the slider range calculation
    df = df[(df['atrp_change_abs'] < df['atrp_change_abs'].quantile(0.99)) & (df['atrp_change_abs'] > 0.01)].copy()

    print(f"Data Loaded: {len(df)} records.")
    return df

class AtrDashboard:
    def __init__(self, df):
        self.df = df
        self.slider_interacting = False # Flag to track drag state
    
        # Definitions
        self.daily_range = list(range(1, 25))
        self.weekly_range = list(range(1, 9))
        self.current_range = self.daily_range # Default

        # --- Layout Setup ---
        self.fig = plt.figure(figsize=(24, 13))

        # Grid: Left col (controls) | Right col (charts)
        # Right column will have 3 rows: Path, Violin, Probs
        self.gs = gridspec.GridSpec(3, 2, width_ratios=[1, 5], height_ratios=[2, 2, 1])
        self.fig.canvas.manager.set_window_title('Breakout Analysis Dashboard')
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.3)

        # --- Sidebar Controls Panel (Left Column) ---
        # Spanning all 3 rows of the left column
        # Increased to 11 rows to accommodate the Update button
        gs_controls = gridspec.GridSpecFromSubplotSpec(11, 1, subplot_spec=self.gs[:, 0],
                                                       height_ratios=[1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 3], hspace=0.4)

        # 0. Update Button (Top)
        self.ax_btn = self.fig.add_subplot(gs_controls[0])

        # 1. Slider Area (Top) - Abs Strength
        self.ax_slider = self.fig.add_subplot(gs_controls[1])
        self.ax_slider.set_title("Abs Strength", fontsize=10, pad=15)

        # 2. Direction Selection
        self.ax_radio_dir = self.fig.add_subplot(gs_controls[2])
        self.ax_radio_dir.set_title("Direction", fontsize=10)
        self.ax_radio_dir.set_frame_on(False)

        # 3. Timeframe Selection
        self.ax_radio_tf = self.fig.add_subplot(gs_controls[3])
        self.ax_radio_tf.set_title("Timeframe", fontsize=10)
        self.ax_radio_tf.set_frame_on(False)

        # 4. Duration Slider
        self.ax_slider_dur = self.fig.add_subplot(gs_controls[4])
        self.ax_slider_dur.set_title("Duration", fontsize=10, pad=15)

        # NEW: 5. Violin Y-Axis Limit Slider
        self.ax_slider_ylim = self.fig.add_subplot(gs_controls[5])
        self.ax_slider_ylim.set_title("Violin Y-Limit %", fontsize=10, pad=15)

        # NEW: 6. RVol50 Range Slider
        self.ax_slider_rvol = self.fig.add_subplot(gs_controls[6])
        self.ax_slider_rvol.set_title("RVol 50d Range", fontsize=10, pad=15)

        # NEW: 7. Date Range (Year) Slider
        self.ax_slider_year = self.fig.add_subplot(gs_controls[7])
        self.ax_slider_year.set_title("Year Range", fontsize=10, pad=15)

        # 8. Earnings Radio
        self.ax_radio_earn = self.fig.add_subplot(gs_controls[8])
        self.ax_radio_earn.set_title("Earnings", fontsize=10)
        self.ax_radio_earn.set_frame_on(False)

        # 9. SPY Context Radio
        self.ax_radio_spy = self.fig.add_subplot(gs_controls[9])
        self.ax_radio_spy.set_title("SPY Context", fontsize=10)
        self.ax_radio_spy.set_frame_on(False)

        # 10. Market Cap Radio
        self.ax_radio_mcap = self.fig.add_subplot(gs_controls[10])
        self.ax_radio_mcap.set_title("Market Cap", fontsize=10)
        self.ax_radio_mcap.set_frame_on(False)

        # --- Charts (Right Column) ---
        self.ax_path = self.fig.add_subplot(self.gs[0, 1])
        self.ax_violin = self.fig.add_subplot(self.gs[1, 1])
        self.ax_probs = self.fig.add_subplot(self.gs[2, 1]) # Re-added Probs chart

        # --- Widgets ---
        # 0. Update Button
        self.btn_update = Button(self.ax_btn, 'Update Charts', color='#00a8ff', hovercolor='#0097e6')

        # 1. Range Slider (Strength)
        min_s, max_s = df['atrp_change_abs'].min(), df['atrp_change_abs'].max()
        self.slider = RangeSlider(self.ax_slider, '', min_s, max_s, valinit=(min_s, max_s))

        # 2. Direction Radio
        self.radio_dir = RadioButtons(self.ax_radio_dir, ('Both', 'Positive', 'Negative'), active=0, activecolor='cyan')

        # 3. Timeframe Radio
        self.radio_tf = RadioButtons(self.ax_radio_tf, ('Daily', 'Weekly'), active=0, activecolor='cyan')

        # 4. Duration Slider (Integer) - Init with Daily Max
        self.slider_dur = Slider(self.ax_slider_dur, '', 1, 24, valinit=24, valstep=1, color='cyan')

        # NEW: 5. Violin Y-Limit Slider (Range: 10% to 200%)
        self.slider_ylim = Slider(self.ax_slider_ylim, '', 10, 200, valinit=50, valstep=5, color='cyan')

        # NEW: 6. RVol50 Range Slider
        if 'rvol500' in df.columns:
            rv_min, rv_max = df['rvol500'].min(), df['rvol500'].max()
            if pd.isna(rv_min): rv_min = 0
            if pd.isna(rv_max): rv_max = 5
            self.slider_rvol = RangeSlider(self.ax_slider_rvol, '', rv_min, rv_max, valinit=(rv_min, rv_max))
        else:
            self.slider_rvol = RangeSlider(self.ax_slider_rvol, '', 0, 1, valinit=(0, 1))

        # NEW: 7. Date Range (Year) Slider
        if 'date' in df.columns:
            years = df['date'].dt.year.dropna()
            if not years.empty:
                y_min, y_max = int(years.min()), int(years.max())
                if y_min == y_max: y_max += 1 # Ensure range exists
                self.slider_year = RangeSlider(self.ax_slider_year, '', y_min, y_max, valinit=(y_min, y_max), valstep=1, color='cyan')
            else:
                self.slider_year = RangeSlider(self.ax_slider_year, '', 2020, 2025, valinit=(2020, 2025), valstep=1, color='cyan')
        else:
            self.slider_year = RangeSlider(self.ax_slider_year, '', 2020, 2025, valinit=(2020, 2025), valstep=1, color='cyan')

        # 8. Earnings Radio
        self.radio_earn = RadioButtons(self.ax_radio_earn, ('All', 'Earnings', 'No Earnings'), active=0, activecolor='cyan')

        # 9. SPY Context Radio
        self.radio_spy = RadioButtons(self.ax_radio_spy, ('All', 'Supporting', 'Neutral', 'Non-Supporting'), active=0, activecolor='cyan')

        # 10. Market Cap Radio - STRICT ORDER
        available_mcaps = df['mcap_class'].unique()
        # 4. Market Cap Radio (With 'All' option)
        mcaps = ['All', 'Large-Cap', 'Mid-Cap', 'Small-Cap', 'Micro-Cap']
        self.radio_mcap = RadioButtons(self.ax_radio_mcap, mcaps, active=0, activecolor='cyan')

        # --- Event Connections ---
        # Decoupled updates: Only trigger on button click
        self.btn_update.on_clicked(self.update)

        # Adjust UI logic only (no data update)
        self.radio_tf.on_clicked(self.change_timeframe)

        # Initial Draw
        self.update(None)

    def change_timeframe(self, val):
        if val == 'Weekly':
            self.slider_dur.valmax = 8
            self.slider_dur.set_val(min(self.slider_dur.val, 8))
            self.slider_dur.ax.set_xlim(1, 8)
        else:
            self.slider_dur.valmax = 24
            self.slider_dur.ax.set_xlim(1, 24)
        
    def get_filtered_data(self):
        s_min, s_max = self.slider.val
        mask = (self.df['atrp_change_abs'] >= s_min) & (self.df['atrp_change_abs'] <= s_max)

        if hasattr(self, 'slider_rvol') and 'rvol500' in self.df.columns:
            rv_min, rv_max = self.slider_rvol.val
            mask &= (self.df['rvol500'] >= rv_min) & (self.df['rvol500'] <= rv_max)

        if hasattr(self, 'slider_year') and 'date' in self.df.columns:
            y_min, y_max = self.slider_year.val
            mask &= (self.df['date'].dt.year >= y_min) & (self.df['date'].dt.year <= y_max)

        dir_val = self.radio_dir.value_selected
        if dir_val == 'Positive': mask &= (self.df['atrp_change'] > 0)
        elif dir_val == 'Negative': mask &= (self.df['atrp_change'] < 0)

        earn_val = self.radio_earn.value_selected
        if earn_val == 'Earnings': mask &= (self.df['has_earnings'] == True)
        elif earn_val == 'No Earnings': mask &= (self.df['has_earnings'] == False)

        spy_val = self.radio_spy.value_selected
        if spy_val != 'All': mask &= (self.df['spy_class'] == spy_val)

        mcap_val = self.radio_mcap.value_selected
        if mcap_val != 'All': mask &= (self.df['mcap_class'] == mcap_val)

        return self.df[mask]

    def update(self, val):
        sub_df = self.get_filtered_data()

        self.ax_path.clear()
        self.ax_violin.clear()
        self.ax_probs.clear()

        if sub_df.empty:
            self.ax_path.text(0.5, 0.5, "No Data Found", ha='center', transform=self.ax_path.transAxes)
            self.fig.canvas.draw_idle()
            return

        tf = self.radio_tf.value_selected
        max_dur = int(self.slider_dur.val)
    
        if tf == 'Daily':
            prefix = 'aligned_cpct'
            xlabel = "Days After Breakout"
            ema_cols_gen = lambda dist, i: f'aligned_{dist}_dist{i}'
        else: # Weekly
            prefix = 'aligned_w_cpct'
            xlabel = "Weeks After Breakout"
            ema_cols_gen = lambda dist, i: f'aligned_w_{dist}_dist{i}'

        # Generate range based on available columns and selected duration
        periods = []
        for i in range(1, max_dur + 1):
            col_name = f'{prefix}{i}'
            if col_name in sub_df.columns:
                periods.append(i)
        
        path_cols = [f'{prefix}{i}' for i in periods]
        
        # --- 1. Path Analysis (Top) ---
        if path_cols:
            path_data = sub_df[path_cols]
            mean_path = path_data.mean()
            std_path = path_data.std()

            self.ax_path.plot(periods, mean_path, color='cyan', linewidth=2, label='Mean Trajectory')
            self.ax_path.fill_between(periods, mean_path - std_path, mean_path + std_path, color='cyan', alpha=0.15, label='1 Std Dev')
            self.ax_path.axhline(0, color='white', linestyle='--', alpha=0.5, label='Breakout Level')

            self.ax_path.set_title(f"Aligned Price Trajectory ({tf}, N={len(sub_df)}) | Positive = Continuation")
            self.ax_path.set_ylabel("% Change from Breakout")
            self.ax_path.legend(loc='upper left')
            self.ax_path.grid(True, alpha=0.2)
            self.ax_path.set_xticks(periods)

        # --- 2. Violin Distribution (Middle) ---
        if path_cols:
            violin_data = [sub_df[c].dropna().values for c in path_cols]
            
            parts = self.ax_violin.violinplot(violin_data, positions=periods, showmeans=False, showmedians=True, showextrema=False)
            
            for pc in parts['bodies']:
                pc.set_facecolor('#48dbfb')
                pc.set_edgecolor('white')
                pc.set_alpha(0.5)
            
            parts['cmedians'].set_color('white')
            
            y_lim = self.slider_ylim.val
            self.ax_violin.set_ylim(-y_lim, y_lim)
            self.ax_violin.axhline(0, color='white', linestyle='--', alpha=0.5)
            self.ax_violin.set_title(f"Distribution of Price Changes ({tf})")
            self.ax_violin.set_ylabel("% Change")
            self.ax_violin.set_xticks(periods)
            self.ax_violin.grid(True, alpha=0.2)

        # --- 3. Probability Analysis (Bottom) ---
        def get_prob(cols):
            valid = [c for c in cols if c in sub_df.columns]
            if not valid: return 0
            return (sub_df[valid] > 0).all(axis=1).mean() * 100

        prob_bk = get_prob([f'{prefix}{i}' for i in periods])
        prob_ema10 = get_prob([ema_cols_gen('ema10', i) for i in periods])
        prob_ema20 = get_prob([ema_cols_gen('ema20', i) for i in periods])

        # REORDERED: > EMA10, > EMA20, > Breakout
        bars = ['> EMA10', '> EMA20', '> Breakout']
        values = [prob_ema10, prob_ema20, prob_bk]
        colors = ['#feca57', '#48dbfb', '#ff6b6b']

        self.ax_probs.bar(bars, values, color=colors, alpha=0.8)
        self.ax_probs.set_ylim(0, 100)
        self.ax_probs.set_ylabel("Prob (%)")
        self.ax_probs.set_title(f"Prob. of Holding Support (Next {max_dur} {tf.replace('ly','s')})")
        self.ax_probs.set_xlabel(xlabel)

        for i, v in enumerate(values):
            self.ax_probs.text(i, v + 2, f"{v:.1f}%", ha='center', color='white', fontweight='bold')

        self.fig.canvas.draw_idle()

##%%
#data = load_and_prep_data()

#%%
dash = AtrDashboard(data)
plt.show()
