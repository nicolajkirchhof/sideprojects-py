import { Routes } from '@angular/router';
import { TradeIdeas } from './trade-ideas/trade-ideas';
import { About } from './about/about';

export const routes: Routes = [
    { path: 'trade-ideas', component: TradeIdeas },
    { path: 'about', component: About },
    { path: '', redirectTo: '/trade-ideas', pathMatch: 'full' }
];
