import { Routes } from '@angular/router';
import { InstrumentSummaries } from './instrument-summaries/instrument-summaries';
import { TradeEntries } from './trade-entries/trade-entries';
import { OptionPositions } from './option-positions/option-positions';
import { Trades } from './trades/trades';
import { CapitalComponent } from './capital/capital';
import { WeeklyPrepComponent } from './weekly-prep/weekly-prep';
import { AnalyticsComponent } from './analytics/analytics';
import { PortfolioComponent } from './portfolio/portfolio';
import { GreeksHistoryComponent } from './greeks-history/greeks-history';
import { IbkrConfigComponent } from './ibkr-config/ibkr-config';
import { About } from './about/about';

export const routes: Routes = [
    { path: 'dashboard', component: InstrumentSummaries },
    { path: 'trade-entries', component: TradeEntries },
    { path: 'option-positions', component: OptionPositions },
    { path: 'trades', component: Trades },
    { path: 'capital', component: CapitalComponent },
    { path: 'weekly-prep', component: WeeklyPrepComponent },
    { path: 'analytics', component: AnalyticsComponent },
    { path: 'portfolio', component: PortfolioComponent },
    { path: 'greeks-history', component: GreeksHistoryComponent },
    { path: 'ibkr-config', component: IbkrConfigComponent },
    { path: 'about', component: About },
    { path: '', redirectTo: '/dashboard', pathMatch: 'full' }
];
