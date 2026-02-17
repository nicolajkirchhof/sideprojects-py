import numpy as np
import pandas as pd
import matplotlib as mpl

# --- CRITICAL FIX: Set backend BEFORE importing pyplot ---
try:
    mpl.use('QtAgg')  # Preferred for interactive windows
except ImportError:
    mpl.use('TkAgg')  # Fallback

import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, RangeSlider, Slider, Button, TextBox
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
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

    # 3. Determine Direction
    df['direction'] = np.sign(df['event_move'])
    # Replace 0 direction with 1 to avoid zeroing out data
    df['direction'] = df['direction'].replace(0, 1)

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
    # new_cols = {}
    
    # Daily Alignment
    # for d in range(1, 26): # 1 to 25
    #     col = f'cpct{d}'
    #     if col in df.columns:
    #         new_cols[f'aligned_cpct{d}'] = df[col] * df['direction']
        
        # Align EMA distances too? usually less relevant to flip sign, 
        # but if we want "Distance in direction of trade", maybe. 
        # For now, we keep EMA distances raw in the filters.

    # Weekly Alignment
    # for w in range(1, 10): # 1 to 9
    #     col = f'w_cpct{w}'
    #     if col in df.columns:
    #         new_cols[f'aligned_w_cpct{w}'] = df[col] * df['direction']

    # if new_cols:
    #     df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)

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

        # State for tab durations (view limits)
        self.dur_view = {'Daily': 24, 'Weekly': 8}
        self.view_tab = 'Daily'
        self.direction_val = 'Positive'
        self._cond_timer = None

        # --- Layout Setup ---
        self.fig = plt.figure(figsize=(28, 22))
        self.fig.canvas.manager.set_window_title('Momentum & Earnings Analysis Dashboard')
    
        # Grid: Left (Controls) | Right (Charts)
        self.gs = gridspec.GridSpec(1, 2, width_ratios=[1, 4])
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.15)

        # --- Controls Panel (Left Column) ---
        # 23 Rows Total (Removed Duration Slider)
        gs_controls = gridspec.GridSpecFromSubplotSpec(23, 1, subplot_spec=self.gs[0], 
                                                       height_ratios=[0.8, 1, 1.0, 1.2, 1.2, 0.8, 
                                                                      0.6, 0.6, 0.6, 
                                                                      0.6, 0.6, 0.6, 0.6, 0.6, 
                                                                      0.6, 0.6, 0.6, 0.6, 0.6, 
                                                                      1.2, 1.5, 1.2, 1.0],
                                                       hspace=0.6)

        # 0. View Tab Switch (Horizontal Buttons)
        gs_tabs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_controls[0], wspace=0.05)
        self.btn_tab_d = Button(self.fig.add_subplot(gs_tabs[0]), 'Daily', color='#00a8ff', hovercolor='gray')
        self.btn_tab_w = Button(self.fig.add_subplot(gs_tabs[1]), 'Weekly', color='black', hovercolor='gray')
        self.btn_tab_d.label.set_fontsize(9)
        self.btn_tab_w.label.set_fontsize(9)
        self.btn_tab_d.on_clicked(lambda e: self.set_tab('Daily'))
        self.btn_tab_w.on_clicked(lambda e: self.set_tab('Weekly'))

        # 1. Update Button
        self.ax_btn = self.fig.add_subplot(gs_controls[1])
        self.btn_update = Button(self.ax_btn, 'Update Charts', color='#00a8ff', hovercolor='#0097e6')
        self.btn_update.on_clicked(self.update)

        # 2. Year Range Slider
        self.ax_slider_year = self.fig.add_subplot(gs_controls[2])
        
        min_y, max_y = 2000, pd.Timestamp.now().year
        if not self.df.empty and 'date' in self.df.columns:
            min_y = int(self.df['date'].dt.year.min())
            max_y = int(self.df['date'].dt.year.max())

        # Default: 2010 to Now (or max available)
        def_start = 2010
        def_end = max_y 
        if min_y > def_start: min_y = def_start 
        
        self.ax_slider_year.set_title(f"Year Range: [{def_start}, {def_end}]", fontsize=9)
        self.slider_year = RangeSlider(self.ax_slider_year, '', min_y, max_y, valinit=(def_start, def_end), valstep=1, color='cyan')
        self.slider_year.on_changed(lambda v: self.ax_slider_year.set_title(f"Year Range: [{int(v[0])}, {int(v[1])}]", fontsize=9))

        # 3. Event Move Slider (Signed)
        self.ax_slider_move = self.fig.add_subplot(gs_controls[3])
        min_m, max_m = -30, 30
        if not self.df.empty and 'event_move' in self.df.columns:
             vals = self.df['event_move'].dropna()
             if not vals.empty:
                 min_m = vals.quantile(0.01)
                 max_m = vals.quantile(0.99)

        # Ensure handles don't stick
        rng = max_m - min_m
        if rng == 0: rng = 1.0
        vmin_s, vmax_s = min_m - rng * 0.1, max_m + rng * 0.1

        self.ax_slider_move.set_title(f"Event Move (Signed): [{min_m:.1f}, {max_m:.1f}]", fontsize=9)
        self.slider_move = RangeSlider(self.ax_slider_move, '', vmin_s, vmax_s, valinit=(min_m, max_m))
        self.slider_move.on_changed(lambda v: self.ax_slider_move.set_title(f"Event Move (Signed): [{v[0]:.1f}, {v[1]:.1f}]", fontsize=9))

        # 4. Price Slider
        self.ax_slider_price = self.fig.add_subplot(gs_controls[4])
        min_p, max_p = 0, 500
        if not self.df.empty and 'c0' in self.df.columns:
             min_p = self.df['c0'].min()
             max_p = self.df['c0'].quantile(0.99)
        self.slider_price = self._create_range_slider(self.ax_slider_price, 'c0', "Breakout Price", min_p, max_p)

        # 5. Direction (Horizontal Buttons)
        gs_dir = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_controls[5], wspace=0.05)
        self.btns_dir = {}
        for i, (lbl, val) in enumerate([('Pos', 'Positive'), ('Neg', 'Negative')]):
            ax = self.fig.add_subplot(gs_dir[i])
            c = '#00a8ff' if val == self.direction_val else 'black'
            btn = Button(ax, lbl, color=c, hovercolor='gray')
            btn.label.set_fontsize(9)
            btn.on_clicked(lambda e, v=val: self.set_direction(v))
            self.btns_dir[val] = {'btn': btn, 'ax': ax}

        # 6-8. Momentum Filters (Shifted up, index 6)
        self.mom_sliders = {}
        for i, mom_col in enumerate(['1M_chg', '3M_chg', '6M_chg']):
            ax = self.fig.add_subplot(gs_controls[6+i])
            self.mom_sliders[mom_col] = self._create_range_slider(ax, mom_col, mom_col, -50, 50)

        # 9-13. Underlying EMA Filters (Shifted up, index 9)
        self.ema_sliders = {}
        ema_dists = ['ema10', 'ema20', 'ema50', 'ema100', 'ema200']
        for i, ema_name in enumerate(ema_dists):
            col = f"{ema_name}_dist0"
            ax = self.fig.add_subplot(gs_controls[9+i])
            self.ema_sliders[col] = self._create_range_slider(ax, col, f"{ema_name} Dist", -20, 20)

        # 14-18. SPY EMA Filters (Shifted up, index 14)
        self.spy_ema_sliders = {}
        for i, ema_name in enumerate(ema_dists):
            col = f"spy_{ema_name}_dist0"
            ax = self.fig.add_subplot(gs_controls[14+i])
            self.spy_ema_sliders[col] = self._create_range_slider(ax, col, f"SPY {ema_name} Dist", -10, 10)

        # 19. Conditional Survival Filter (Shifted up, index 19)
        gs_cond = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_controls[19], width_ratios=[1, 2], wspace=0.1)
        
        self.ax_slider_cond_t = self.fig.add_subplot(gs_cond[0])
        self.ax_slider_cond_t.set_title("Cond. Day/Wk", fontsize=9)
        # Use Slider instead of TextBox
        self.slider_cond_t = Slider(self.ax_slider_cond_t, '', 0, 24, valinit=0, valstep=1, color='cyan')
        self.slider_cond_t.on_changed(self._on_cond_t_change)
        
        self.ax_slider_cond_v = self.fig.add_subplot(gs_cond[1])
        self.ax_slider_cond_v.set_title("Cond. Range", fontsize=9)
        self.slider_cond_v = RangeSlider(self.ax_slider_cond_v, '', -20, 50, valinit=(-20, 50))
        # No auto-update

        # 20. Misc Filters (Shifted up, index 20)
        gs_misc = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_controls[20], hspace=0.1)
        self.ax_rad_earn = self.fig.add_subplot(gs_misc[0])
        self.ax_rad_earn.set_title("Event Type", fontsize=9)
        self.radio_earn = RadioButtons(self.ax_rad_earn, ('All', 'Earnings Only', 'Non-Earnings'), active=0, activecolor='cyan')
    
        self.ax_rad_spy = self.fig.add_subplot(gs_misc[1])
        self.ax_rad_spy.set_title("SPY Context", fontsize=9)
        self.radio_spy = RadioButtons(self.ax_rad_spy, ('All', 'Supporting', 'Neutral', 'Non-Supporting'), active=0, activecolor='cyan')

        # 21. Market Cap (Shifted up, index 21)
        gs_mcap_container = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_controls[21], height_ratios=[0.3, 1], hspace=0.1)
        ax_mcap_title = self.fig.add_subplot(gs_mcap_container[0])
        ax_mcap_title.text(0.5, 0, "Market Cap", ha='center', va='bottom', fontsize=9, color='white')
        ax_mcap_title.axis('off')

        gs_mcap_btns = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs_mcap_container[1], wspace=0.05)
        self.mcap_selected = 'All'
        self.mcap_buttons = {}
        mcaps = ['All', 'Large', 'Mid', 'Small', 'Micro']

        for i, mcap in enumerate(mcaps):
            ax = self.fig.add_subplot(gs_mcap_btns[i])
            c = '#00a8ff' if mcap == 'All' else 'black'
            btn = Button(ax, mcap, color=c, hovercolor='gray')
            btn.label.set_fontsize(8)
            btn.on_clicked(lambda event, m=mcap: self.set_mcap(m))
            self.mcap_buttons[mcap] = btn

        # 22. Y-Limit (Shifted up, index 22)
        self.ax_slider_ylim = self.fig.add_subplot(gs_controls[22])
        self.ax_slider_ylim.set_title("Y-Axis Limit %", fontsize=9)
        self.slider_ylim = Slider(self.ax_slider_ylim, '', 10, 200, valinit=50, valstep=5, color='cyan')

        # --- Charts (Right Column) ---
        gs_charts = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=self.gs[1], height_ratios=[2, 2, 1], hspace=0.3)
        
        # Daily Axes
        self.ax_d_path = self.fig.add_subplot(gs_charts[0], label='daily_path')
        self.ax_d_violin = self.fig.add_subplot(gs_charts[1], label='daily_violin')
        self.ax_d_probs = self.fig.add_subplot(gs_charts[2], label='daily_probs')

        # Weekly Axes (Stacked)
        self.ax_w_path = self.fig.add_subplot(gs_charts[0], label='weekly_path', frameon=False)
        self.ax_w_violin = self.fig.add_subplot(gs_charts[1], label='weekly_violin', frameon=False)
        self.ax_w_probs = self.fig.add_subplot(gs_charts[2], label='weekly_probs', frameon=False)

        self.update(None)

    def set_tab(self, val):
        """Manual Tab Switch logic with button persistence."""
        self.view_tab = val
        
        c_d = '#00a8ff' if val == 'Daily' else 'black'
        c_w = '#00a8ff' if val == 'Weekly' else 'black'
        
        # Fix: Update both color attr and facecolor
        self.btn_tab_d.color = c_d
        self.btn_tab_w.color = c_w
        self.btn_tab_d.ax.set_facecolor(c_d)
        self.btn_tab_w.ax.set_facecolor(c_w)
        
        is_daily = (val == 'Daily')
        for ax in [self.ax_d_path, self.ax_d_violin, self.ax_d_probs]: ax.set_visible(is_daily)
        for ax in [self.ax_w_path, self.ax_w_violin, self.ax_w_probs]: ax.set_visible(not is_daily)

        if is_daily:
            # Update Cond T Slider
            self.slider_cond_t.valmax = 24
            self.slider_cond_t.ax.set_xlim(0, 24)
            if self.slider_cond_t.val > 24: self.slider_cond_t.set_val(24)
        else:
            # Update Cond T Slider
            self.slider_cond_t.valmax = 8
            self.slider_cond_t.ax.set_xlim(0, 8)
            if self.slider_cond_t.val > 8: self.slider_cond_t.set_val(8)

        # Schedule a debounced refresh of conditional range after tab switch
        self._schedule_cond_v_update()

        self.fig.canvas.draw_idle()

    def set_direction(self, val):
        """Manual Direction Switch logic with persistence."""
        self.direction_val = val
        for k, btn_dict in self.btns_dir.items():
            c = '#00a8ff' if k == val else 'black'
            # Fix: Update both color attr and facecolor
            btn_dict['btn'].color = c
            btn_dict['ax'].set_facecolor(c)
        
        # No auto-update, wait for button click
        self.fig.canvas.draw_idle()

    def set_mcap(self, val):
        """Callback for Market Cap horizontal buttons."""
        self.mcap_selected = val
        for m, btn in self.mcap_buttons.items():
            c = '#00a8ff' if m == val else 'black'
            btn.color = c
            btn.ax.set_facecolor(c)
            btn.ax.figure.canvas.draw_idle()

    def _on_cond_t_change(self, val):
        """Debounced callback: after 1s of no changes, recompute slider_cond_v bounds."""
        self._schedule_cond_v_update()

    def _schedule_cond_v_update(self):
        """(Re)start a 1s debounce timer to update slider_cond_v limits."""
        if self._cond_timer is not None:
            try:
                self._cond_timer.stop()
            except Exception:
                pass

        self._cond_timer = self.fig.canvas.new_timer(interval=1000)  # 1s settlement time
        self._cond_timer.single_shot = True
        self._cond_timer.add_callback(self._update_cond_v_bounds_from_data)
        self._cond_timer.start()

    def _update_cond_v_bounds_from_data(self):
        """Update slider_cond_v allowed min/max based on selected cond_t column distribution."""
        try:
            cond_t = int(self.slider_cond_t.val)
        except Exception:
            cond_t = 0

        # If disabled, keep a broad default range
        if cond_t <= 0:
            vmin, vmax = -20.0, 50.0
        else:
            tf = self.view_tab
            col = f'cpct{cond_t}' if tf == 'Daily' else f'w_cpct{cond_t}'

            if col in self.df.columns:
                vals = self.df[col].dropna()
                if not vals.empty:
                    # Robust bounds (avoid outliers dominating)
                    vmin = float(vals.quantile(0.01))
                    vmax = float(vals.quantile(0.99))
                else:
                    vmin, vmax = -20.0, 50.0
            else:
                vmin, vmax = -20.0, 50.0

        if vmin >= vmax:
            vmax = vmin + 1.0

        # Expand slightly so handles aren't pinned to the edges
        rng = vmax - vmin
        vmin_s, vmax_s = vmin - rng * 0.1, vmax + rng * 0.1

        # Update RangeSlider bounds + axis limits
        self.slider_cond_v.valmin = vmin_s
        self.slider_cond_v.valmax = vmax_s
        self.slider_cond_v.ax.set_xlim(vmin_s, vmax_s)

        # Clamp current selection into the new bounds
        cur_min, cur_max = self.slider_cond_v.val
        new_min = min(max(cur_min, vmin_s), vmax_s)
        new_max = min(max(cur_max, vmin_s), vmax_s)
        if new_min > new_max:
            new_min, new_max = vmin, vmax

        self.slider_cond_v.set_val((new_min, new_max))
        self.ax_slider_cond_v.set_title(f"Cond. Range: [{new_min:.1f}, {new_max:.1f}]", fontsize=9)

        self.fig.canvas.draw_idle()

    def _create_range_slider(self, ax, col_name, title, default_min, default_max, cap_at_quantile=False):
        """Helper to create consistent range sliders with labels."""
        vmin, vmax = default_min, default_max
        if not self.df.empty and col_name in self.df.columns:
            vals = self.df[col_name].dropna()
            if not vals.empty:
                vmin = vals.min()
                vmax = vals.max()

        if vmin >= vmax: vmax = vmin + 1.0

        ax.set_title(f"{title}: [{vmin:.1f}, {vmax:.1f}]", fontsize=9)
        slider = RangeSlider(ax, '', vmin, vmax, valinit=(vmin, vmax))

        # Callback to update label
        def update_label(val):
            ax.set_title(f"{title}: [{val[0]:.1f}, {val[1]:.1f}]", fontsize=9)

        slider.on_changed(update_label)
        return slider

    def change_timeframe(self, val):
        pass # Obsolete with tabs

    def get_filtered_data(self):
        mask = pd.Series(True, index=self.df.index)

        # 0. Year Range
        min_y, max_y = self.slider_year.val
        if 'date' in self.df.columns:
             mask &= (self.df['date'].dt.year >= min_y) & (self.df['date'].dt.year <= max_y)

        # 1. Signed Move
        min_m, max_m = self.slider_move.val
        mask &= (self.df['event_move'] >= min_m) & (self.df['event_move'] <= max_m)

        # 1b. Breakout Price
        min_p, max_p = self.slider_price.val
        if 'c0' in self.df.columns:
             mask &= (self.df['c0'] >= min_p) & (self.df['c0'] <= max_p)

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
        dir_val = self.direction_val
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
        if mcap_val != 'All': mask &= (self.df['market_cap_class'] == mcap_val)

        # 9. Conditional Survival
        # Filter based on the value of cpctX column
        try:
            cond_t = int(self.slider_cond_t.val)
        except Exception:
            cond_t = 0

        if cond_t > 0:
            tf = self.view_tab
            min_c, max_c = self.slider_cond_v.val
            
            # Construct column name based on timeframe
            # We use aligned columns to match positive/negative logic
            if tf == 'Daily':
                col = f'cpct{cond_t}'
            else:
                col = f'w_cpct{cond_t}'
            
            if col in self.df.columns:
                 mask &= (self.df[col] >= min_c) & (self.df[col] <= max_c)

        return self.df[mask]

    def update(self, val):
        sub_df = self.get_filtered_data()

        for ax in [self.ax_d_path, self.ax_d_violin, self.ax_d_probs, 
                   self.ax_w_path, self.ax_w_violin, self.ax_w_probs]:
            ax.clear()

        if sub_df.empty:
            self.ax_d_path.text(0.5, 0.5, "No Data Found", ha='center', transform=self.ax_d_path.transAxes)
            self.fig.canvas.draw_idle()
            return

        self._plot_metrics(sub_df, 'Daily', [self.ax_d_path, self.ax_d_violin, self.ax_d_probs], 24)
        self._plot_metrics(sub_df, 'Weekly', [self.ax_w_path, self.ax_w_violin, self.ax_w_probs], 8)

        self.set_tab(self.view_tab)

    def _plot_metrics(self, sub_df, tf, axes, max_dur):
        ax_path, ax_violin, ax_probs = axes

        if tf == 'Daily':
            prefix = 'cpct'
            xlabel = "Days After Event"
            ema10_gen = lambda i: f'ema10_dist{i}'
            ema20_gen = lambda i: f'ema20_dist{i}'
            ema50_gen = lambda i: f'ema50_dist{i}'
        else: # Weekly
            prefix = 'w_cpct'
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

            ax_path.plot(periods, mean_path, color='cyan', linewidth=2, label='Mean Trajectory')
            ax_path.fill_between(periods, mean_path - std_path, mean_path + std_path, color='cyan', alpha=0.15, label='1 Std Dev')
            ax_path.axhline(0, color='white', linestyle='--', alpha=0.5)

            title_str = f"Trajectory (N={len(sub_df)}) | Positive = Continuation"
            ax_path.set_title(title_str)
            ax_path.set_ylabel("% Change from Event")
            ax_path.legend(loc='upper left')
            ax_path.grid(True, alpha=0.2)
            ax_path.set_xticks(periods)
            ax_path.xaxis.set_major_locator(MultipleLocator(1))

            yl = self.slider_ylim.val
            ax_path.set_ylim(-yl, yl)

            # --- 2. Violin Chart ---
            if path_cols:
                violin_data = [sub_df[c].dropna().values for c in path_cols]
                if violin_data and all(len(d) > 0 for d in violin_data):
                    parts = ax_violin.violinplot(violin_data, positions=periods, showmeans=False, showmedians=True, showextrema=False)

                    for pc in parts['bodies']:
                        pc.set_facecolor('#48dbfb')
                        pc.set_edgecolor('white')
                        pc.set_alpha(0.5)
                    parts['cmedians'].set_color('white')

                    ax_violin.set_ylim(-self.slider_ylim.val, self.slider_ylim.val)
                    ax_violin.axhline(0, color='white', linestyle='--', alpha=0.5)
                    ax_violin.set_title(f"Distribution of Changes ({tf})")
                    ax_violin.set_xticks(periods)
                    ax_violin.xaxis.set_major_locator(MultipleLocator(1))
                    ax_violin.grid(True, alpha=0.2)
                else:
                    ax_violin.text(0.5, 0.5, "Insufficient Data for Violins", ha='center', transform=ax_violin.transAxes)

                # --- 3. Probability Chart (Holding Levels) ---
            def get_survival_prob(col_gen_func, comparison_val=0):
                curve = []
                current_idx = sub_df.index

                for i in periods:
                    col = col_gen_func(i)
                    if col not in sub_df.columns:
                        curve.append(np.nan)
                        continue

                    valid = sub_df.loc[current_idx, col].dropna()
                    if valid.empty:
                        curve.append(np.nan)
                    else:
                        prob = (valid > comparison_val).mean() * 100
                        curve.append(prob)
                return curve

            y_bk = get_survival_prob(lambda i: f'{prefix}{i}', 0)
            y_ema10 = get_survival_prob(ema10_gen, 0)
            y_ema20 = get_survival_prob(ema20_gen, 0)
            y_ema50 = get_survival_prob(ema50_gen, 0)

            ax_probs.plot(periods, y_bk, color='#ff6b6b', label='> Entry', linewidth=2, marker='o', markersize=4)
            ax_probs.plot(periods, y_ema10, color='#feca57', label='> EMA10', linewidth=2, marker='o', markersize=4)
            ax_probs.plot(periods, y_ema20, color='#48dbfb', label='> EMA20', linewidth=2, marker='o', markersize=4)
            ax_probs.plot(periods, y_ema50, color='#1dd1a1', label='> EMA50', linewidth=2, marker='o', markersize=4)

            ax_probs.set_ylim(0, 105)
            ax_probs.set_ylabel("Prob. (%)")
            ax_probs.set_xlabel(xlabel)
            ax_probs.set_title(f"Probability of Holding Levels ({tf})")
            ax_probs.legend(loc='lower left', fontsize=8, ncol=4)
            ax_probs.grid(True, alpha=0.2)
            ax_probs.set_xticks(periods)
            ax_probs.xaxis.set_major_locator(MultipleLocator(1))

            self.fig.canvas.draw_idle()

#%%
# if __name__ == '__main__':
data = load_and_prep_data()
#%%
if not data.empty:
    dash = MomentumEarningsDashboard(data)
    plt.show()

