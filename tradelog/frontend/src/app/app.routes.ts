import { Routes } from '@angular/router';
import { TradeIdeas } from './trade-ideas/trade-ideas';
import { About } from './about/about';
import { Positions } from './positions/positions';
import { Instruments } from './instruments/instruments';

export const routes: Routes = [
    { path: 'trade-ideas', component: TradeIdeas },
    { path: 'positions', component: Positions },
    { path: 'instruments', component: Instruments },
    { path: 'about', component: About },
    { path: '', redirectTo: '/trade-ideas', pathMatch: 'full' }
];
