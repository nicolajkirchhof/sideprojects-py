import { Component, OnInit, OnDestroy, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatCardModule } from '@angular/material/card';
import { IbkrConfigService, IbkrConfig, SyncStatus } from './ibkr-config.service';

@Component({
  selector: 'app-ibkr-config',
  standalone: true,
  imports: [
    CommonModule,
    ReactiveFormsModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatIconModule,
    MatCardModule,
  ],
  templateUrl: './ibkr-config.html',
  host: { class: 'flex flex-col flex-1 overflow-auto' },
})
export class IbkrConfigComponent implements OnInit, OnDestroy {
  private service = inject(IbkrConfigService);
  private fb = inject(FormBuilder);

  form!: FormGroup;
  syncStatus: SyncStatus | null = null;
  syncing = false;
  syncMessage: string | null = null;
  saved = false;

  private pollTimer: any = null;

  ngOnInit(): void {
    this.form = this.fb.group({
      host: ['127.0.0.1', [Validators.required]],
      port: [7497, [Validators.required, Validators.min(1), Validators.max(65535)]],
      clientId: [1, [Validators.required, Validators.min(0)]],
    });

    this.service.getConfig().subscribe({
      next: (config) => {
        this.form.patchValue({
          host: config.host,
          port: config.port,
          clientId: config.clientId,
        });
      },
    });

    this.refreshSyncStatus();
    this.pollTimer = setInterval(() => this.refreshSyncStatus(), 30_000);
  }

  ngOnDestroy(): void {
    if (this.pollTimer) clearInterval(this.pollTimer);
  }

  onSave(): void {
    if (this.form.invalid) {
      this.form.markAllAsTouched();
      return;
    }
    const v = this.form.value;
    const config: IbkrConfig = {
      host: v.host,
      port: Number(v.port),
      clientId: Number(v.clientId),
    };
    this.saved = false;
    this.service.updateConfig(config).subscribe({
      next: () => {
        this.saved = true;
        setTimeout(() => this.saved = false, 3000);
      },
      error: (err) => console.error('Save failed', err),
    });
  }

  onSync(): void {
    this.syncing = true;
    this.syncMessage = null;
    this.service.triggerSync().subscribe({
      next: (res) => {
        this.syncMessage = res?.message ?? 'Sync completed.';
        this.syncing = false;
        this.refreshSyncStatus();
      },
      error: (err) => {
        this.syncMessage = err.error?.message ?? 'Sync failed.';
        this.syncing = false;
        this.refreshSyncStatus();
      },
    });
  }

  private refreshSyncStatus(): void {
    this.service.getSyncStatus().subscribe({
      next: (status) => (this.syncStatus = status),
    });
  }

  get cooldownLabel(): string {
    if (!this.syncStatus?.cooldownRemainingSeconds) return '';
    const min = Math.ceil(this.syncStatus.cooldownRemainingSeconds / 60);
    return `Available in ${min}m`;
  }

  get lastSyncLabel(): string {
    if (!this.syncStatus?.lastSyncAt) return 'Never synced';
    const diff = Date.now() - new Date(this.syncStatus.lastSyncAt).getTime();
    const min = Math.floor(diff / 60_000);
    if (min < 1) return 'Synced just now';
    if (min < 60) return `Synced ${min}m ago`;
    const h = Math.floor(min / 60);
    if (h < 24) return `Synced ${h}h ago`;
    return `Synced ${Math.floor(h / 24)}d ago`;
  }
}
