import { Component, computed, inject } from '@angular/core';
import { toSignal } from '@angular/core/rxjs-interop';
import { BreakpointObserver, Breakpoints } from '@angular/cdk/layout';
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatButtonModule } from '@angular/material/button';
import { MatSidenavModule } from '@angular/material/sidenav';
import { MatListModule } from '@angular/material/list';
import { MatIconModule } from '@angular/material/icon';
import { MatTooltipModule } from '@angular/material/tooltip';
import { of, timer } from 'rxjs';
import { map, filter, startWith, switchMap, catchError } from 'rxjs/operators';
import { RouterOutlet, RouterLink, Router, NavigationEnd } from '@angular/router';
import { OptionPositionsLogService } from './option-positions/option-positions.service';
import { AccountSwitcherComponent } from './shared/account-switcher/account-switcher';

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
    AccountSwitcherComponent
  ],
  templateUrl: './app.html'
})
export class App {
  private breakpointObserver = inject(BreakpointObserver);
  private router = inject(Router);
  private logService = inject(OptionPositionsLogService);

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
        if (url.startsWith('/about')) return 'About';
        return 'Dashboard';
      })
    ),
    { initialValue: 'Dashboard' }
  );

  breadcrumb = computed(() => `${this.sectionLabel()} > List`);

  lastSyncInfo = toSignal(
    timer(0, 60_000).pipe(
      switchMap(() => this.logService.getLastSync().pipe(
        catchError(() => of({ lastSync: null }))
      )),
      map(({ lastSync }) => {
        if (!lastSync) return { label: 'Never synced', stale: true };
        const diffMs = Date.now() - new Date(lastSync).getTime();
        const diffMin = Math.floor(diffMs / 60_000);
        if (diffMin < 1) return { label: 'Synced just now', stale: false };
        if (diffMin < 60) return { label: `Synced ${diffMin}m ago`, stale: false };
        const diffH = Math.floor(diffMin / 60);
        if (diffH < 24) return { label: `Synced ${diffH}h ago`, stale: false };
        const diffD = Math.floor(diffH / 24);
        return { label: `Synced ${diffD}d ago`, stale: true };
      })
    ),
    { initialValue: { label: 'Never synced', stale: true } }
  );
}
