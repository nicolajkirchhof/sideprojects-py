import { Routes } from '@angular/router';
import { Log } from './logs/log.component';
import { About } from './about/about';
import { Positions } from './positions/positions';
import { Instruments } from './instruments/instruments';

export const routes: Routes = [
    { path: 'log', component: Log },
    { path: 'positions', component: Positions },
    { path: 'instruments', component: Instruments },
    { path: 'about', component: About },
    { path: '', redirectTo: '/log', pathMatch: 'full' }
];
