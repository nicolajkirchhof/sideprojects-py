import { ApplicationConfig, provideZonelessChangeDetection } from '@angular/core';
import { provideRouter } from '@angular/router';
import { provideHttpClient, withInterceptors } from '@angular/common/http';
import { provideCharts, withDefaultRegisterables } from 'ng2-charts';
import { DateAdapter } from '@angular/material/core';

import { routes } from './app.routes';
import { accountInterceptor } from './shared/account.interceptor';
import { AppDateAdapter } from './shared/app-date-adapter';

export const appConfig: ApplicationConfig = {
  providers: [
    provideZonelessChangeDetection(),
    provideRouter(routes),
    provideHttpClient(withInterceptors([accountInterceptor])),
    provideCharts(withDefaultRegisterables()),
    { provide: DateAdapter, useClass: AppDateAdapter },
  ]
};
