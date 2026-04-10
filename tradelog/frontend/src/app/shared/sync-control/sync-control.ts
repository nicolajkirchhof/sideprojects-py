import { Component, HostListener, OnInit, inject } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatMenuModule } from '@angular/material/menu';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatTooltipModule } from '@angular/material/tooltip';
import { SyncService } from '../sync.service';

@Component({
  selector: 'app-sync-control',
  standalone: true,
  imports: [MatButtonModule, MatIconModule, MatMenuModule, MatProgressBarModule, MatTooltipModule],
  template: `
    <div class="sync-control"
         [class.is-syncing]="sync.isSyncing()"
         [class.is-error]="sync.phase() === 'error'"
         [class.tone-fresh]="sync.freshness().tone === 'fresh' && !sync.isSyncing()"
         [class.tone-stale]="sync.freshness().tone === 'stale' && !sync.isSyncing()"
         [class.tone-error]="(sync.freshness().tone === 'error' || sync.phase() === 'error') && !sync.isSyncing()">
      <div class="sync-control__row">
        <span class="sync-control__dot" aria-hidden="true"></span>
        @if (sync.isSyncing()) {
          <mat-icon class="sync-control__icon spin">progress_activity</mat-icon>
          <span class="sync-control__label">{{ sync.stepLabel() }}</span>
        } @else if (sync.phase() === 'error') {
          <mat-icon class="sync-control__icon">error_outline</mat-icon>
          <span class="sync-control__label">Sync failed</span>
        } @else {
          <span class="sync-control__label">{{ sync.freshness().label }}</span>
        }

        <button mat-button
                type="button"
                class="sync-control__main"
                [disabled]="sync.isSyncing()"
                (click)="sync.syncAll()"
                matTooltip="Sync all (Ctrl+Shift+S)">
          <mat-icon class="!text-[1rem] !w-4 !h-4">sync</mat-icon>
          <span class="ml-1">Sync</span>
        </button>

        <button mat-icon-button
                type="button"
                class="sync-control__more"
                [matMenuTriggerFor]="syncMenu"
                [disabled]="sync.isSyncing()"
                aria-label="Sync options">
          <mat-icon class="!text-[1rem] !w-4 !h-4">arrow_drop_down</mat-icon>
        </button>

        <mat-menu #syncMenu="matMenu" xPosition="before">
          <button mat-menu-item type="button" (click)="sync.syncFlex()" [disabled]="!sync.status()?.flexConfigured">
            <mat-icon>cloud_download</mat-icon>
            <span>Flex Sync</span>
            <span class="sync-menu__meta">{{ flexMeta() }}</span>
          </button>
          <button mat-menu-item type="button" (click)="sync.syncLive()">
            <mat-icon>sync</mat-icon>
            <span>Live Sync (Greeks)</span>
            <span class="sync-menu__meta">{{ liveMeta() }}</span>
          </button>
        </mat-menu>
      </div>

      @if (sync.isSyncing()) {
        <mat-progress-bar mode="indeterminate" class="sync-control__progress"></mat-progress-bar>
      }
    </div>
  `,
  styles: [`
    .sync-control {
      display: inline-flex;
      flex-direction: column;
      min-width: 18rem;
      background-color: var(--mat-sys-surface-container-high);
      border: 1px solid var(--mat-sys-outline-variant);
      border-radius: 0.25rem;
      overflow: hidden;
      transition: border-color 120ms ease, background-color 120ms ease;
    }
    .sync-control__row {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      height: 2rem;
      padding: 0 0.5rem 0 0.625rem;
    }
    .sync-control__dot {
      width: 0.5rem;
      height: 0.5rem;
      border-radius: 999px;
      background-color: var(--mat-sys-outline);
      flex-shrink: 0;
    }
    .sync-control.tone-fresh .sync-control__dot { background-color: #4ade80; }
    .sync-control.tone-stale .sync-control__dot { background-color: #f59e0b; }
    .sync-control.tone-error .sync-control__dot { background-color: #f87171; }
    .sync-control.is-syncing .sync-control__dot { background-color: var(--mat-sys-primary); }

    .sync-control__icon {
      font-size: 0.875rem !important;
      width: 0.875rem !important;
      height: 0.875rem !important;
      line-height: 0.875rem !important;
      color: var(--mat-sys-on-surface-variant);
    }
    .sync-control.is-error .sync-control__icon { color: #f87171; }

    .sync-control__label {
      flex: 1 1 auto;
      font-size: 0.75rem;
      font-weight: 500;
      letter-spacing: 0.02em;
      color: var(--mat-sys-on-surface);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .sync-control__main.mat-mdc-button {
      --mat-button-text-state-layer-color: var(--mat-sys-on-surface);
      min-width: 0;
      height: 1.5rem;
      padding: 0 0.5rem;
      font-size: 0.75rem;
      line-height: 1.5rem;
      color: var(--mat-sys-on-surface);
    }
    .sync-control__more.mat-mdc-icon-button {
      --mdc-icon-button-state-layer-size: 1.5rem;
      width: 1.5rem !important;
      height: 1.5rem !important;
      padding: 0 !important;
      color: var(--mat-sys-on-surface-variant);
    }

    .sync-control__progress {
      height: 0.125rem !important;
      --mdc-linear-progress-track-height: 0.125rem;
      --mdc-linear-progress-active-indicator-height: 0.125rem;
    }

    .spin {
      animation: sync-spin 900ms linear infinite;
    }
    @keyframes sync-spin {
      from { transform: rotate(0deg); }
      to   { transform: rotate(360deg); }
    }

    /* Dropdown menu meta column */
    :host ::ng-deep .sync-menu__meta {
      margin-left: auto;
      font-size: 0.6875rem;
      opacity: 0.6;
      letter-spacing: 0.02em;
    }
  `],
})
export class SyncControlComponent implements OnInit {
  protected sync = inject(SyncService);

  ngOnInit(): void {
    this.sync.refreshStatus();
  }

  @HostListener('window:keydown', ['$event'])
  onKeydown(event: KeyboardEvent): void {
    if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key.toLowerCase() === 's') {
      event.preventDefault();
      this.sync.syncAll();
    }
  }

  flexMeta(): string {
    const at = this.sync.status()?.lastFlexSyncAt;
    if (!at) return 'Never';
    return this.relative(at);
  }
  liveMeta(): string {
    const at = this.sync.status()?.lastLiveSyncAt;
    if (!at) return 'Never';
    return this.relative(at);
  }
  private relative(iso: string): string {
    const diff = Date.now() - new Date(iso).getTime();
    const min = Math.floor(diff / 60_000);
    if (min < 1) return 'just now';
    if (min < 60) return `${min}m ago`;
    const h = Math.floor(min / 60);
    if (h < 24) return `${h}h ago`;
    return `${Math.floor(h / 24)}d ago`;
  }
}
