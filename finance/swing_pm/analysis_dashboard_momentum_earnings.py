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

# Configure plot style
plt.style.use('dark_background')
sns.set_palette("viridis")

#%%
def load_and_prep_data():
    """Loads and standardizes the momentum/earnings dataset for the dashboard."""
    print("Loading momentum earnings data...")
    try:
        df = pd.read_pickle('finance/_data/all_momentum_earnings.pkl')
    except FileNotFoundError:
        print("Data file 'finance/_data/all_momentum_earnings.pkl' not found.")
        print("Please run 'finance/swing_pm/create_data_momentum_earnings_analysis.py' first.")
        return pd.DataFrame()

    df = df.reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], np.nan).infer_objects(copy=False)

    # 1. Clean Path Columns (Must have data for the reaction period)
    # We look for 'c0' (event day) and at least 'c24' or similar to ensure we have data
    if 'c0' in df.columns:
        df = df.dropna(subset=['c0'])

    # 2. Define "Strength" of the move
    # We use the Close % Change on the event day (cpct0) relative to day -1
    # User requested: change the x atr, which is the cpct0 / atrp20
    if 'cpct0' in df.columns and 'atrp200' in df.columns:
        df['event_move'] = df['cpct0'] / df['atrp200']
    else:
        df['event_move'] = 0.0

    df['event_move_abs'] = df['event_move'].abs()

    # 3. Determine Direction
    df['direction'] = np.sign(df['event_move'])
    # Replace 0 direction with 1 to avoid zeroing out data
    df['direction'] = df['direction'].replace(0, 1)

    # 4. Handle Categorical Data
    if 'mcap_class' in df.columns:
        df['mcap_class'] = df['mcap_class'].astype(str)
    
    if 'is_earnings' in df.columns:
        df['is_earnings'] = df['is_earnings'].fillna(False).infer_objects(copy=False).astype(bool)

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    # 5. SPY Context (Simple alignment check over 5 days)
    if 'spy0' in df.columns and 'spy5' in df.columns:
        spy_change = df['spy5'] - df['spy0']
        # If SPY moves in same direction as the stock event
        aligned_spy = spy_change * df['direction']
        
        conditions = [
            aligned_spy > 0.5,   # Supporting
            aligned_spy < -0.5   # Non-Supporting
        ]
        choices = ['Supporting', 'Non-Supporting']
        df['spy_class'] = np.select(conditions, choices, default='Neutral')
    else:
        df['spy_class'] = 'Unknown'

    # 6. Create Aligned Path Columns (Multiplying by direction)
    # This allows us to view "Continuation" regardless of Up/Down start
    new_cols = {}
    
    # Daily Alignment
    for d in range(1, 26): # 1 to 25
        col = f'cpct{d}'
        if col in df.columns:
            new_cols[f'aligned_cpct{d}'] = df[col] * df['direction']
        
        # Align EMA distances too? usually less relevant to flip sign, 
        # but if we want "Distance in direction of trade", maybe. 
        # For now, we keep EMA distances raw in the filters.

    # Weekly Alignment
    for w in range(1, 10): # 1 to 9
        col = f'w_cpct{w}'
        if col in df.columns:
            new_cols[f'aligned_w_cpct{w}'] = df[col] * df['direction']

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)

    print(f"Data Loaded: {len(df)} records.")
    return df

class MomentumEarningsDashboard:
    def __init__(self, df):
        self.df = df
        
        # --- Config & State ---
        self.daily_range = list(range(1, 25))
        self.weekly_range = list(range(1, 9))
        self.active_mom_col = '1M_chg' # Default Momentum Metric
        self.active_ema_col = 'ema20_dist0' # Default Underlying EMA Metric
        self.active_spy_ema_col = 'spy_ema20_dist0' # Default SPY EMA Metric

        # --- Layout Setup ---
        self.fig = plt.figure(figsize=(28, 22))
        self.fig.canvas.manager.set_window_title('Momentum & Earnings Analysis Dashboard')
    
        # Grid: Left (Controls) | Right (Charts)
        self.gs = gridspec.GridSpec(1, 2, width_ratios=[1, 4])
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.15)

        # --- Controls Panel (Left Column) ---
        # We need many rows for the requested filters
        # 1. Update
        # 2. Abs Move
        # 3. Direction
        # 4. Duration
        # 5-7. Momentum Filter (1M, 3M, 6M)
        # 8-12. Underlying EMA Filter (EMA10-200)
        # 13-17. SPY EMA Filter (EMA10-200)
        # 18. Earnings
        # 19. Mcap
        # 20. Y-Lim
    
        gs_controls = gridspec.GridSpecFromSubplotSpec(20, 1, subplot_spec=self.gs[0], 
                                                       height_ratios=[1, 1.2, 1.2, 1.2, 
                                                                      0.6, 0.6, 0.6, # Momentum (Compact)
                                                                      0.6, 0.6, 0.6, 0.6, 0.6, # EMAs (Compact)
                                                                      0.6, 0.6, 0.6, 0.6, 0.6, # SPY EMAs (Compact)
                                                                      1.2, 1.2, 1],
                                                       hspace=0.4) # Reduced spacing

        # 1. Update Button
        self.ax_btn = self.fig.add_subplot(gs_controls[0])
        self.btn_update = Button(self.ax_btn, 'Update Charts', color='#00a8ff', hovercolor='#0097e6')
        self.btn_update.on_clicked(self.update)

        # 2. Abs Move Slider
        self.ax_slider_move = self.fig.add_subplot(gs_controls[1])
        min_m, max_m = 0, 30 # Default cap
        if not self.df.empty:
            max_m = self.df['event_move_abs'].max()

        self.ax_slider_move.set_title(f"Event Move x times ATR20: [{0:.1f}, {max_m:.1f}]", fontsize=9)
        self.slider_move = RangeSlider(self.ax_slider_move, '', 0, max_m, valinit=(0, max_m*0.5))
        self.slider_move.on_changed(lambda v: self.ax_slider_move.set_title(f"Event Move % (Abs): [{v[0]:.1f}, {v[1]:.1f}]", fontsize=9))

        # 3. Direction & Timeframe (Shared Row Split)
        gs_row3 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_controls[2])
        self.ax_rad_dir = self.fig.add_subplot(gs_row3[0])
        self.ax_rad_dir.set_title("Direction", fontsize=9)
        self.radio_dir = RadioButtons(self.ax_rad_dir, ('Both', 'Positive', 'Negative'), active=0, activecolor='cyan')

        self.ax_rad_tf = self.fig.add_subplot(gs_row3[1])
        self.ax_rad_tf.set_title("Chart TF", fontsize=9)
        self.radio_tf = RadioButtons(self.ax_rad_tf, ('Daily', 'Weekly'), active=0, activecolor='cyan')
        self.radio_tf.on_clicked(self.change_timeframe)

        # 4. Duration Slider
        self.ax_slider_dur = self.fig.add_subplot(gs_controls[3])
        self.ax_slider_dur.set_title("Chart Duration", fontsize=9)
        self.slider_dur = Slider(self.ax_slider_dur, '', 1, 24, valinit=24, valstep=1, color='cyan')

        # 5-7. Momentum Filters (All ranges)
        self.mom_sliders = {}
        for i, mom_col in enumerate(['1M_chg', '3M_chg', '6M_chg']):
            ax = self.fig.add_subplot(gs_controls[4+i])
            self.mom_sliders[mom_col] = self._create_range_slider(ax, mom_col, mom_col, -50, 50, cap_at_quantile=True)

        # 8-12. Underlying EMA Filters (All ranges)
        self.ema_sliders = {}
        ema_dists = ['ema10', 'ema20', 'ema50', 'ema100', 'ema200']
        for i, ema_name in enumerate(ema_dists):
            col = f"{ema_name}_dist0"
            ax = self.fig.add_subplot(gs_controls[7+i])
            self.ema_sliders[col] = self._create_range_slider(ax, col, f"{ema_name} Dist", -20, 20, cap_at_quantile=True)

        # 13-17. SPY EMA Filters (All ranges)
        self.spy_ema_sliders = {}
        for i, ema_name in enumerate(ema_dists):
            col = f"spy_{ema_name}_dist0"
            ax = self.fig.add_subplot(gs_controls[12+i])
            self.spy_ema_sliders[col] = self._create_range_slider(ax, col, f"SPY {ema_name} Dist", -10, 10, cap_at_quantile=True)

        # 18. Misc Filters (Earnings / SPY Context)
        gs_misc = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_controls[17])
        self.ax_rad_earn = self.fig.add_subplot(gs_misc[0])
        self.ax_rad_earn.set_title("Event Type", fontsize=9)
        self.radio_earn = RadioButtons(self.ax_rad_earn, ('All', 'Earnings Only', 'Non-Earnings'), active=0, activecolor='cyan')
    
        self.ax_rad_spy = self.fig.add_subplot(gs_misc[1])
        self.ax_rad_spy.set_title("SPY Context", fontsize=9)
        self.radio_spy = RadioButtons(self.ax_rad_spy, ('All', 'Supporting', 'Neutral', 'Non-Supporting'), active=0, activecolor='cyan')

        # 19. Market Cap (Horizontal Selection)
        # We create a nested grid: Title row + Button row
        gs_mcap_container = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_controls[18], height_ratios=[0.3, 1], hspace=0.1)
        
        # Title
        ax_mcap_title = self.fig.add_subplot(gs_mcap_container[0])
        ax_mcap_title.text(0.5, 0, "Market Cap", ha='center', va='bottom', fontsize=9, color='white')
        ax_mcap_title.axis('off')

        # Buttons Grid
        gs_mcap_btns = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs_mcap_container[1], wspace=0.05)
        
        self.mcap_selected = 'All'
        self.mcap_buttons = {}
        mcaps = ['All', 'Large-Cap', 'Mid-Cap', 'Small-Cap', 'Micro-Cap']
        labels = ['All', 'Large', 'Mid', 'Small', 'Micro']
        
        for i, (mcap, lbl) in enumerate(zip(mcaps, labels)):
            ax = self.fig.add_subplot(gs_mcap_btns[i])
            # Initial color
            c = '#00a8ff' if mcap == 'All' else 'black'
            btn = Button(ax, lbl, color=c, hovercolor='gray')
            btn.label.set_fontsize(8)
            # Use default argument m=mcap to capture the value in the closure
            btn.on_clicked(lambda event, m=mcap: self.set_mcap(m))
            self.mcap_buttons[mcap] = btn

        # 20. Y-Limit Slider for Charts
        self.ax_slider_ylim = self.fig.add_subplot(gs_controls[19])
        self.ax_slider_ylim.set_title("Y-Axis Limit %", fontsize=9)
        self.slider_ylim = Slider(self.ax_slider_ylim, '', 10, 200, valinit=50, valstep=5, color='cyan')

        # --- Charts (Right Column) ---
        gs_charts = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=self.gs[1], height_ratios=[2, 2, 1], hspace=0.3)
        self.ax_path = self.fig.add_subplot(gs_charts[0])
        self.ax_violin = self.fig.add_subplot(gs_charts[1])
        self.ax_probs = self.fig.add_subplot(gs_charts[2])

        # Initial Draw
        self.update(None)

    def set_mcap(self, val):
        """Callback for Market Cap horizontal buttons."""
        self.mcap_selected = val
        for m, btn in self.mcap_buttons.items():
            c = '#00a8ff' if m == val else 'black'
            btn.color = c
            btn.ax.set_facecolor(c)
            # Force redraw of the button area
            btn.ax.figure.canvas.draw_idle()

    def _create_range_slider(self, ax, col_name, title, default_min, default_max, cap_at_quantile=False):
        """Helper to create consistent range sliders with labels."""
        vmin, vmax = default_min, default_max
        if cap_at_quantile and not self.df.empty and col_name in self.df.columns:
            vals = self.df[col_name].dropna()
            if not vals.empty:
                q_min = vals.quantile(0.01)
                q_max = vals.quantile(0.99)
                vmin = q_min
                vmax = q_max

        if vmin >= vmax: vmax = vmin + 1.0

        # Pad the slider limits slightly so the handles aren't stuck at the edges
        rng = vmax - vmin
        vmin_s, vmax_s = vmin - rng*0.1, vmax + rng*0.1

        ax.set_title(f"{title}: [{vmin:.1f}, {vmax:.1f}]", fontsize=9)
        slider = RangeSlider(ax, '', vmin_s, vmax_s, valinit=(vmin, vmax))

        # Callback to update label
        def update_label(val):
            ax.set_title(f"{title}: [{val[0]:.1f}, {val[1]:.1f}]", fontsize=9)

        slider.on_changed(update_label)
        return slider

    def change_timeframe(self, val):
        if val == 'Weekly':
            self.slider_dur.valmax = 8
            self.slider_dur.set_val(min(self.slider_dur.val, 8))
            self.slider_dur.ax.set_xlim(1, 8)
        else:
            self.slider_dur.valmax = 24
            self.slider_dur.ax.set_xlim(1, 24)

    def get_filtered_data(self):
        mask = pd.Series(True, index=self.df.index)

        # 1. Abs Move
        min_m, max_m = self.slider_move.val
        mask &= (self.df['event_move_abs'] >= min_m) & (self.df['event_move_abs'] <= max_m)

        # 2. Momentum (Multi-Select)
        for mom_col, slider in self.mom_sliders.items():
            if mom_col in self.df.columns:
                min_v, max_v = slider.val
                mask &= (self.df[mom_col] >= min_v) & (self.df[mom_col] <= max_v)

        # 3. Underlying EMA (Multi-Select)
        for ema_col, slider in self.ema_sliders.items():
            if ema_col in self.df.columns:
                min_e, max_e = slider.val
                mask &= (self.df[ema_col] >= min_e) & (self.df[ema_col] <= max_e)

        # 4. SPY EMA (Multi-Select)
        for spy_col, slider in self.spy_ema_sliders.items():
            if spy_col in self.df.columns:
                min_s, max_s = slider.val
                mask &= (self.df[spy_col] >= min_s) & (self.df[spy_col] <= max_s)

        # 5. Direction
        dir_val = self.radio_dir.value_selected
        if dir_val == 'Positive': mask &= (self.df['event_move'] > 0)
        elif dir_val == 'Negative': mask &= (self.df['event_move'] < 0)

        # 6. Earnings
        earn_val = self.radio_earn.value_selected
        if earn_val == 'Earnings Only': mask &= (self.df['is_earnings'] == True)
        elif earn_val == 'Non-Earnings': mask &= (self.df['is_earnings'] == False)

        # 7. SPY Context
        spy_val = self.radio_spy.value_selected
        if spy_val != 'All': mask &= (self.df['spy_class'] == spy_val)

        # 8. Market Cap
        mcap_val = self.mcap_selected
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
        
        # Determine columns to plot
        if tf == 'Daily':
            prefix = 'aligned_cpct'
            xlabel = "Days After Event"
            # EMA support cols (using close > ema10/20 at future dates)
            # NOTE: We assume 'emaXX_distY' means distance at time Y. 
            # If dist > 0, price > EMA.
            ema10_gen = lambda i: f'ema10_dist{i}'
            ema20_gen = lambda i: f'ema20_dist{i}'
            ema50_gen = lambda i: f'ema50_dist{i}'
        else: # Weekly
            prefix = 'aligned_w_cpct'
            xlabel = "Weeks After Event"
            ema10_gen = lambda i: f'w_ema10_dist{i}'
            ema20_gen = lambda i: f'w_ema20_dist{i}'
            ema50_gen = lambda i: f'w_ema50_dist{i}'

        periods = []
        for i in range(1, max_dur + 1):
            col = f'{prefix}{i}'
            if col in sub_df.columns:
                periods.append(i)
        
        path_cols = [f'{prefix}{i}' for i in periods]
        
        # --- 1. Path Chart ---
        if path_cols:
            path_data = sub_df[path_cols]
            mean_path = path_data.mean()
            std_path = path_data.std()

            self.ax_path.plot(periods, mean_path, color='cyan', linewidth=2, label='Mean Trajectory')
            self.ax_path.fill_between(periods, mean_path - std_path, mean_path + std_path, color='cyan', alpha=0.15, label='1 Std Dev')
            self.ax_path.axhline(0, color='white', linestyle='--', alpha=0.5)

            title_str = f"Aligned Trajectory (N={len(sub_df)}) | Positive = Continuation"
            self.ax_path.set_title(title_str)
            self.ax_path.set_ylabel("% Change from Event")
            self.ax_path.legend(loc='upper left')
            self.ax_path.grid(True, alpha=0.2)
            self.ax_path.set_xticks(periods)
            
            # Use Y-Limit Slider
            yl = self.slider_ylim.val
            self.ax_path.set_ylim(-yl, yl)

        # --- 2. Violin Chart ---
        if path_cols:
            violin_data = [sub_df[c].dropna().values for c in path_cols]
            # Handle cases where some days might be empty after dropna
            if violin_data and all(len(d) > 0 for d in violin_data):
                parts = self.ax_violin.violinplot(violin_data, positions=periods, showmeans=False, showmedians=True, showextrema=False)
                
                for pc in parts['bodies']:
                    pc.set_facecolor('#48dbfb')
                    pc.set_edgecolor('white')
                    pc.set_alpha(0.5)
                parts['cmedians'].set_color('white')
            
                self.ax_violin.set_ylim(-self.slider_ylim.val, self.slider_ylim.val)
                self.ax_violin.axhline(0, color='white', linestyle='--', alpha=0.5)
                self.ax_violin.set_title(f"Distribution of Changes ({tf})")
                self.ax_violin.set_xticks(periods)
                self.ax_violin.grid(True, alpha=0.2)
            else:
                 self.ax_violin.text(0.5, 0.5, "Insufficient Data for Violins", ha='center', transform=self.ax_violin.transAxes)

        # --- 3. Probability Chart (Holding Levels) ---
        def get_survival_prob(col_gen_func, comparison_val=0):
            curve = []
            # Start assuming all valid
            current_idx = sub_df.index
            
            for i in periods:
                col = col_gen_func(i)
                # Check column existence first
                if col not in sub_df.columns:
                    curve.append(np.nan)
                    continue
                
                # We calculate probability of being > comparison at this specific step
                # Note: This is "Point-in-time" probability, not "Survival" (cumulative)
                # If we want cumulative (has NEVER broken), we need to maintain a mask.
                # Let's do Point-in-time % > 0 (or EMA)
                
                valid = sub_df.loc[current_idx, col].dropna()
                if valid.empty:
                    curve.append(np.nan)
                else:
                    prob = (valid > comparison_val).mean() * 100
                    curve.append(prob)
            return curve

        # For "Breakout Hold", we check if aligned change > 0
        y_bk = get_survival_prob(lambda i: f'{prefix}{i}', 0)
        
        # For EMAs, we check if dist > 0
        y_ema10 = get_survival_prob(ema10_gen, 0)
        y_ema20 = get_survival_prob(ema20_gen, 0)
        y_ema50 = get_survival_prob(ema50_gen, 0)

        self.ax_probs.plot(periods, y_bk, color='#ff6b6b', label='> Entry', linewidth=2, marker='o', markersize=4)
        self.ax_probs.plot(periods, y_ema10, color='#feca57', label='> EMA10', linewidth=2, marker='o', markersize=4)
        self.ax_probs.plot(periods, y_ema20, color='#48dbfb', label='> EMA20', linewidth=2, marker='o', markersize=4)
        self.ax_probs.plot(periods, y_ema50, color='#1dd1a1', label='> EMA50', linewidth=2, marker='o', markersize=4)

        self.ax_probs.set_ylim(0, 105)
        self.ax_probs.set_ylabel("Prob. (%)")
        self.ax_probs.set_title(f"Probability of Holding Levels ({tf})")
        self.ax_probs.set_xlabel(xlabel)
        self.ax_probs.legend(loc='lower left', fontsize=8, ncol=3)
        self.ax_probs.grid(True, alpha=0.2)
        self.ax_probs.set_xticks(periods)

        self.fig.canvas.draw_idle()

#%%
# if __name__ == '__main__':
data = load_and_prep_data()
#%%
if not data.empty:
    dash = MomentumEarningsDashboard(data)
    plt.show()
