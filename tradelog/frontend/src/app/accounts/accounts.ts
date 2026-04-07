import { Component, inject, signal } from '@angular/core';

import { ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { MatTableModule } from '@angular/material/table';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { MatCardModule } from '@angular/material/card';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { ContentArea } from '../shared/content-area/content-area';
import { NotificationService } from '../shared/notification.service';
import { AccountsService, Account, SyncStatus } from './accounts.service';

@Component({
  selector: 'app-accounts',
  standalone: true,
  imports: [
    ReactiveFormsModule,
    MatTableModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatIconModule,
    MatCheckboxModule,
    MatCardModule,
    MatProgressBarModule,
    ContentArea
],
  templateUrl: './accounts.html',
  host: { class: 'flex flex-col flex-1' },
})
export class AccountsComponent {
  private service = inject(AccountsService);
  private fb = inject(FormBuilder);
  private notify = inject(NotificationService);

  loading = signal(false);

  accounts = signal<Account[]>([]);
  displayedColumns = ['name', 'ibkrAccountId', 'host', 'port', 'clientId', 'isDefault', 'lastSyncAt'];

  showSidebar = signal(false);
  form!: FormGroup;
  isCreating = signal(false);
  editMode = signal(false);
  selected = signal<Account | null>(null);

  syncStatus = signal<SyncStatus | null>(null);
  syncing = signal(false);
  Math = Math;
  syncMessage = signal<string | null>(null);

  constructor() {
    this.load();
    this.form = this.fb.group({
      id: [{ value: null, disabled: true }],
      name: ['', [Validators.required]],
      ibkrAccountId: ['', [Validators.required]],
      host: ['127.0.0.1', [Validators.required]],
      port: [7497, [Validators.required, Validators.min(1), Validators.max(65535)]],
      clientId: [1, [Validators.required, Validators.min(0)]],
      isDefault: [false],
      flexToken: [''],
      flexQueryId: [''],
    });
  }

  load(): void {
    this.loading.set(true);
    this.service.getAll().subscribe({
      next: (data) => {
        this.accounts.set(data ?? []);
        this.loading.set(false);
        if (!this.selected() && (data?.length ?? 0) > 0) {
          this.onRowSelect(data![0]);
        }
      },
      error: () => {
        this.notify.error('Failed to load accounts');
        this.loading.set(false);
      },
    });
    this.refreshSyncStatus();
  }

  onRowSelect(row: Account): void {
    this.isCreating.set(false);
    this.editMode.set(false);
    this.selected.set(row);
    this.form.reset({
      id: row.id,
      name: row.name,
      ibkrAccountId: row.ibkrAccountId,
      host: row.host,
      port: row.port,
      clientId: row.clientId,
      isDefault: row.isDefault,
      flexToken: row.flexToken ?? '',
      flexQueryId: row.flexQueryId ?? '',
    });
    this.form.disable({ emitEvent: false });
    this.showSidebar.set(true);
    this.syncMessage.set(null);
    this.refreshSyncStatus();
  }

  onEdit(): void {
    this.editMode.set(true);
    this.form.enable({ emitEvent: false });
    this.form.get('id')?.disable({ emitEvent: false });
  }

  onCancelEdit(): void {
    if (this.isCreating()) {
      this.onCancel();
      return;
    }
    const row = this.selected();
    if (row) this.onRowSelect(row);
  }

  onNew(): void {
    this.isCreating.set(true);
    this.editMode.set(true);
    this.selected.set(null);
    this.form.reset({
      id: null,
      name: '',
      ibkrAccountId: '',
      host: '127.0.0.1',
      port: 7497,
      clientId: 1,
      isDefault: false,
      flexToken: '',
      flexQueryId: '',
    });
    this.form.enable({ emitEvent: false });
    this.form.get('id')?.disable({ emitEvent: false });
    this.showSidebar.set(true);
    this.syncMessage.set(null);
  }

  onCancel(): void {
    this.showSidebar.set(false);
    this.selected.set(null);
    this.isCreating.set(false);
    this.editMode.set(false);
  }

  onSave(): void {
    if (this.form.invalid) {
      this.form.markAllAsTouched();
      return;
    }
    const v = this.form.getRawValue();
    const payload: Partial<Account> = {
      id: v.id ?? undefined,
      name: v.name,
      ibkrAccountId: v.ibkrAccountId,
      host: v.host,
      port: Number(v.port),
      clientId: Number(v.clientId),
      isDefault: v.isDefault,
      flexToken: v.flexToken || null,
      flexQueryId: v.flexQueryId || null,
    };

    if (this.isCreating() || !payload.id) {
      this.service.create(payload).subscribe({
        next: (created) => {
          this.notify.success('Account created');
          this.load();
          this.onCancel();
          // Auto-select if first account
          if (this.accounts().length === 0) {
            this.service.selectAccount(created.id);
          }
        },
        error: () => this.notify.error('Failed to create account'),
      });
    } else {
      this.service.update(payload.id as number, payload).subscribe({
        next: () => {
          this.notify.success('Account updated');
          this.load();
          this.onCancel();
        },
        error: () => this.notify.error('Failed to update account'),
      });
    }
  }

  onDelete(): void {
    if (!this.selected()) return;
    if (!confirm(`Delete account "${this.selected()!.name}"?`)) return;
    this.service.delete(this.selected()!.id).subscribe({
      next: () => {
        this.notify.success('Account deleted');
        this.load();
        this.onCancel();
      },
      error: () => this.notify.error('Failed to delete account'),
    });
  }

  onFlexSync(): void {
    if (!this.selected()) return;
    this.service.selectAccount(this.selected()!.id);
    this.syncing.set(true);
    this.syncMessage.set(null);
    this.service.triggerFlexSync().subscribe({
      next: (res) => {
        this.syncMessage.set(res?.message ?? 'Flex sync completed.');
        this.syncing.set(false);
        this.load();
      },
      error: (err) => {
        this.syncMessage.set(err.error?.message ?? 'Flex sync failed.');
        this.syncing.set(false);
        this.load();
      },
    });
  }

  onLiveSync(): void {
    if (!this.selected()) return;
    this.service.selectAccount(this.selected()!.id);
    this.syncing.set(true);
    this.syncMessage.set(null);
    this.service.triggerLiveSync().subscribe({
      next: (res) => {
        this.syncMessage.set(res?.message ?? 'Live sync completed.');
        this.syncing.set(false);
        this.load();
      },
      error: (err) => {
        this.syncMessage.set(err.error?.message ?? 'Live sync failed.');
        this.syncing.set(false);
        this.load();
      },
    });
  }

  private refreshSyncStatus(): void {
    this.service.getSyncStatus().subscribe({
      next: (status) => this.syncStatus.set(status),
    });
  }

  lastSyncLabel(account: Account): string {
    if (!account.lastSyncAt) return 'Never';
    const diff = Date.now() - new Date(account.lastSyncAt).getTime();
    const min = Math.floor(diff / 60_000);
    if (min < 60) return `${min}m ago`;
    const h = Math.floor(min / 60);
    if (h < 24) return `${h}h ago`;
    return `${Math.floor(h / 24)}d ago`;
  }
}
