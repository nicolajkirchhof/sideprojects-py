import { Component, inject } from '@angular/core';
import { BreakpointObserver, Breakpoints } from '@angular/cdk/layout';
import { AsyncPipe } from '@angular/common';
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatButtonModule } from '@angular/material/button';
import { MatSidenavModule } from '@angular/material/sidenav';
import { MatListModule } from '@angular/material/list';
import { MatIconModule } from '@angular/material/icon';
import { Observable } from 'rxjs';
import { map, shareReplay, filter, startWith } from 'rxjs/operators';
import { RouterOutlet, RouterLink, Router, NavigationEnd } from '@angular/router';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    RouterOutlet,
    RouterLink,
    AsyncPipe,
    MatToolbarModule,
    MatButtonModule,
    MatSidenavModule,
    MatListModule,
    MatIconModule
  ],
  templateUrl: './app.html'
})
export class App {
  private breakpointObserver = inject(BreakpointObserver);
  private router = inject(Router);

  isHandset$: Observable<boolean> = this.breakpointObserver.observe(Breakpoints.Handset)
    .pipe(
      map(result => result.matches),
      shareReplay()
    );

  // Derive a simple breadcrumb: Section > List
  sectionLabel$: Observable<string> = this.router.events.pipe(
    filter((e): e is NavigationEnd => e instanceof NavigationEnd),
    map(() => this.router.url.split('?')[0]),
    startWith(this.router.url)
  ).pipe(
    map(url => {
      if (url.startsWith('/positions')) return 'Positions';
      if (url.startsWith('/instruments')) return 'Instruments';
      if (url.startsWith('/about')) return 'About';
      return 'Log';
    }),
    shareReplay(1)
  );

  breadcrumb$: Observable<string> = this.sectionLabel$.pipe(
    map(section => `${section} > List`)
  );
}
