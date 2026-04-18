import { Routes } from '@angular/router';
import { Trades } from './trades/trades';
import { OptionPositions } from './option-positions/option-positions';
import { StockPositions } from './stock-positions/stock-positions';
import { CapitalComponent } from './capital/capital';
import { WeeklyPrepComponent } from './weekly-prep/weekly-prep';
import { AnalyticsComponent } from './analytics/analytics';
import { StrategyLibraryComponent } from './strategy-library/strategy-library';
import { GreeksHistoryComponent } from './greeks-history/greeks-history';
import { AccountsComponent } from './accounts/accounts';
import { About } from './about/about';
import { SettingsComponent } from './settings/settings';

export const routes: Routes = [
    { path: 'dashboard', component: AnalyticsComponent },
    { path: 'trades', component: Trades },
    { path: 'option-positions', component: OptionPositions },
    { path: 'stock-positions', component: StockPositions },
    { path: 'capital', component: CapitalComponent },
    { path: 'weekly-prep', component: WeeklyPrepComponent },
    { path: 'strategy-library', component: StrategyLibraryComponent },
    { path: 'greeks-history', component: GreeksHistoryComponent },
    { path: 'accounts', component: AccountsComponent },
    { path: 'settings', component: SettingsComponent },
    { path: 'about', component: About },
    { path: '', redirectTo: '/dashboard', pathMatch: 'full' }
];
