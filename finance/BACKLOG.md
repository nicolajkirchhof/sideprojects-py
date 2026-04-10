# Apps — Swing Plot

## B-01 — Force re-fetch of today's bar on manual refresh
- [ ] When `refresh_offset_days=0` and today's bar already exists in cache, override cursor to include today so intraday data is replaced with the latest available bar from IBKR.
- Affected: `finance/utils/ibkr.py::daily_w_volatility`

## B-02 — Move Refresh to far-right with icon
- [ ] Move the Refresh button to the right side of the toolbar (after stretch) and replace text label with a reload icon (`SP_BrowserReload`).
- Affected: `finance/apps/swing_plot/_app.py::_build_toolbar`

## B-03 — Use icon for Load button
- [ ] Replace the "Load" text button with an icon (`SP_MediaPlay`) and a tooltip.
- Affected: `finance/apps/swing_plot/_app.py::_build_toolbar`
