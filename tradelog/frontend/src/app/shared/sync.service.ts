import { Injectable, computed, inject, signal } from '@angular/core';
import { AccountsService, SyncStatus } from '../accounts/accounts.service';
import { NotificationService } from './notification.service';
import { firstValueFrom } from 'rxjs';

type SyncPhase = 'idle' | 'flex' | 'live' | 'done' | 'error';

/**
 * Global sync state + commands. Lives in the toolbar, keeps state
 * across route navigations.
 */
@Injectable({ providedIn: 'root' })
export class SyncService {
  private accounts = inject(AccountsService);
  private notify = inject(NotificationService);

  readonly status = signal<SyncStatus | null>(null);
  readonly phase = signal<SyncPhase>('idle');
  readonly stepLabel = signal<string>('');
  readonly lastError = signal<string | null>(null);

  readonly isSyncing = computed(() => {
    const p = this.phase();
    return p === 'flex' || p === 'live';
  });

  /** Derived freshness info for the chip display. */
  readonly freshness = computed(() => {
    const s = this.status();
    if (!s?.lastLiveSyncAt) return { label: 'Never synced', tone: 'error' as const };
    const diffMs = Date.now() - new Date(s.lastLiveSyncAt).getTime();
    const min = Math.floor(diffMs / 60_000);
    const h = Math.floor(min / 60);
    const d = Math.floor(h / 24);
    // Tone: fresh ≤ 24h, stale ≤ 3 days, error beyond
    const tone: 'fresh' | 'stale' | 'error' =
      h < 24 ? 'fresh' : d <= 3 ? 'stale' : 'error';
    if (min < 1) return { label: 'Synced just now', tone };
    if (min < 60) return { label: `Synced ${min}m ago`, tone };
    if (h < 24) return { label: `Synced ${h}h ago`, tone };
    return { label: `Synced ${d}d ago`, tone };
  });

  refreshStatus(): void {
    this.accounts.getSyncStatus().subscribe({
      next: s => this.status.set(s),
      error: () => this.status.set(null),
    });
  }

  async syncAll(): Promise<void> {
    if (this.isSyncing()) return;
    this.lastError.set(null);
    try {
      if (this.status()?.flexConfigured) {
        await this.runFlex();
      }
      await this.runLive();
      this.phase.set('done');
      this.notify.success('Sync complete');
    } catch (err: any) {
      this.phase.set('error');
      const msg = err?.error?.message ?? err?.message ?? 'Sync failed';
      this.lastError.set(msg);
      this.notify.error(msg);
    } finally {
      this.refreshStatus();
      setTimeout(() => {
        if (this.phase() === 'done' || this.phase() === 'error') this.phase.set('idle');
      }, 2000);
    }
  }

  async syncFlex(): Promise<void> {
    if (this.isSyncing()) return;
    this.lastError.set(null);
    try {
      await this.runFlex();
      this.phase.set('done');
      this.notify.success('Flex sync complete');
    } catch (err: any) {
      this.phase.set('error');
      const msg = err?.error?.message ?? err?.message ?? 'Flex sync failed';
      this.lastError.set(msg);
      this.notify.error(msg);
    } finally {
      this.refreshStatus();
      setTimeout(() => {
        if (this.phase() === 'done' || this.phase() === 'error') this.phase.set('idle');
      }, 2000);
    }
  }

  async syncLive(): Promise<void> {
    if (this.isSyncing()) return;
    this.lastError.set(null);
    try {
      await this.runLive();
      this.phase.set('done');
      this.notify.success('Live sync complete');
    } catch (err: any) {
      this.phase.set('error');
      const msg = err?.error?.message ?? err?.message ?? 'Live sync failed';
      this.lastError.set(msg);
      this.notify.error(msg);
    } finally {
      this.refreshStatus();
      setTimeout(() => {
        if (this.phase() === 'done' || this.phase() === 'error') this.phase.set('idle');
      }, 2000);
    }
  }

  private async runFlex(): Promise<void> {
    this.phase.set('flex');
    this.stepLabel.set('Parsing Flex report…');
    await firstValueFrom(this.accounts.triggerFlexSync());
  }

  private async runLive(): Promise<void> {
    this.phase.set('live');
    this.stepLabel.set('Fetching Greeks & prices…');
    await firstValueFrom(this.accounts.triggerLiveSync());
  }
}
