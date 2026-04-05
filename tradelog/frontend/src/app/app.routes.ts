import { Routes } from '@angular/router';
import { InstrumentSummaries } from './instrument-summaries/instrument-summaries';
import { Trades } from './trades/trades';
import { OptionPositions } from './option-positions/option-positions';
import { StockPositions } from './stock-positions/stock-positions';
import { CapitalComponent } from './capital/capital';
import { WeeklyPrepComponent } from './weekly-prep/weekly-prep';
import { AnalyticsComponent } from './analytics/analytics';
import { PortfolioComponent } from './portfolio/portfolio';
import { GreeksHistoryComponent } from './greeks-history/greeks-history';
import { AccountsComponent } from './accounts/accounts';
import { About } from './about/about';

export const routes: Routes = [
    { path: 'dashboard', component: InstrumentSummaries },
    { path: 'trades', component: Trades },
    { path: 'option-positions', component: OptionPositions },
    { path: 'stock-positions', component: StockPositions },
    { path: 'capital', component: CapitalComponent },
    { path: 'weekly-prep', component: WeeklyPrepComponent },
    { path: 'analytics', component: AnalyticsComponent },
    { path: 'portfolio', component: PortfolioComponent },
    { path: 'greeks-history', component: GreeksHistoryComponent },
    { path: 'accounts', component: AccountsComponent },
    { path: 'about', component: About },
    { path: '', redirectTo: '/dashboard', pathMatch: 'full' }
];
