import { Component, computed, inject } from '@angular/core';
import { toSignal } from '@angular/core/rxjs-interop';
import { BreakpointObserver, Breakpoints } from '@angular/cdk/layout';
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatButtonModule } from '@angular/material/button';
import { MatSidenavModule } from '@angular/material/sidenav';
import { MatListModule } from '@angular/material/list';
import { MatIconModule } from '@angular/material/icon';
import { MatTooltipModule } from '@angular/material/tooltip';
import { map, filter, startWith } from 'rxjs/operators';
import { RouterOutlet, RouterLink, Router, NavigationEnd } from '@angular/router';
import { AccountSwitcherComponent } from './shared/account-switcher/account-switcher';
import { SyncControlComponent } from './shared/sync-control/sync-control';
import { ThemeService } from './shared/theme.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    RouterOutlet,
    RouterLink,
    MatToolbarModule,
    MatButtonModule,
    MatSidenavModule,
    MatListModule,
    MatIconModule,
    MatTooltipModule,
    AccountSwitcherComponent,
    SyncControlComponent
  ],
  templateUrl: './app.html'
})
export class App {
  private breakpointObserver = inject(BreakpointObserver);
  private router = inject(Router);
  protected theme = inject(ThemeService);

  isHandset = toSignal(
    this.breakpointObserver.observe(Breakpoints.Handset).pipe(
      map(result => result.matches)
    ),
    { initialValue: false }
  );

  sectionLabel = toSignal(
    this.router.events.pipe(
      filter((e): e is NavigationEnd => e instanceof NavigationEnd),
      map(() => this.router.url.split('?')[0]),
      startWith(this.router.url),
      map(url => {
        if (url.startsWith('/dashboard')) return 'Dashboard';
        if (url.startsWith('/stock-positions')) return 'Stock Positions';
        if (url.startsWith('/option-positions')) return 'Option Positions';
        if (url.startsWith('/trades')) return 'Trades';
        if (url.startsWith('/capital')) return 'Capital';
        if (url.startsWith('/weekly-prep')) return 'Weekly Prep';
        if (url.startsWith('/analytics')) return 'Analytics';
        if (url.startsWith('/portfolio')) return 'Portfolio';
        if (url.startsWith('/greeks-history')) return 'Greeks History';
        if (url.startsWith('/accounts')) return 'Accounts';
        if (url.startsWith('/settings')) return 'Settings';
        if (url.startsWith('/about')) return 'About';
        return 'Dashboard';
      })
    ),
    { initialValue: 'Dashboard' }
  );

  breadcrumb = computed(() => `${this.sectionLabel()} > List`);
}
