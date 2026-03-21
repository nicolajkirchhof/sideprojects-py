import { Component, OnInit, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { MatTableModule } from '@angular/material/table';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { MatCardModule } from '@angular/material/card';
import { ContentArea } from '../shared/content-area/content-area';
import { AccountsService, Account, SyncStatus } from './accounts.service';

@Component({
  selector: 'app-accounts',
  standalone: true,
  imports: [
    CommonModule,
    ReactiveFormsModule,
    MatTableModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatIconModule,
    MatCheckboxModule,
    MatCardModule,
    ContentArea,
  ],
  templateUrl: './accounts.html',
  host: { class: 'flex flex-col flex-1' },
})
export class AccountsComponent implements OnInit {
  private service = inject(AccountsService);
  private fb = inject(FormBuilder);

  accounts: Account[] = [];
  displayedColumns = ['name', 'ibkrAccountId', 'host', 'port', 'clientId', 'isDefault', 'lastSyncAt'];

  showSidebar = false;
  form!: FormGroup;
  isCreating = false;
  selected: Account | null = null;

  syncStatus: SyncStatus | null = null;
  syncing = false;
  syncMessage: string | null = null;

  ngOnInit(): void {
    this.load();
    this.form = this.fb.group({
      id: [{ value: null, disabled: true }],
      name: ['', [Validators.required]],
      ibkrAccountId: ['', [Validators.required]],
      host: ['127.0.0.1', [Validators.required]],
      port: [7497, [Validators.required, Validators.min(1), Validators.max(65535)]],
      clientId: [1, [Validators.required, Validators.min(0)]],
      isDefault: [false],
    });
  }

  load(): void {
    this.service.getAll().subscribe({
      next: (data) => (this.accounts = data ?? []),
      error: (err) => console.error('Failed to load accounts', err),
    });
    this.refreshSyncStatus();
  }

  onRowSelect(row: Account): void {
    this.isCreating = false;
    this.selected = row;
    this.form.reset({
      id: row.id,
      name: row.name,
      ibkrAccountId: row.ibkrAccountId,
      host: row.host,
      port: row.port,
      clientId: row.clientId,
      isDefault: row.isDefault,
    });
    this.showSidebar = true;
    this.syncMessage = null;
    this.refreshSyncStatus();
  }

  onNew(): void {
    this.isCreating = true;
    this.selected = null;
    this.form.reset({
      id: null,
      name: '',
      ibkrAccountId: '',
      host: '127.0.0.1',
      port: 7497,
      clientId: 1,
      isDefault: false,
    });
    this.showSidebar = true;
    this.syncMessage = null;
  }

  onCancel(): void {
    this.showSidebar = false;
    this.selected = null;
    this.isCreating = false;
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
    };

    if (this.isCreating || !payload.id) {
      this.service.create(payload).subscribe({
        next: (created) => {
          this.load();
          this.onCancel();
          // Auto-select if first account
          if (this.accounts.length === 0) {
            this.service.selectAccount(created.id);
          }
        },
        error: (err) => console.error('Save failed', err),
      });
    } else {
      this.service.update(payload.id as number, payload).subscribe({
        next: () => { this.load(); this.onCancel(); },
        error: (err) => console.error('Save failed', err),
      });
    }
  }

  onDelete(): void {
    if (!this.selected) return;
    if (!confirm(`Delete account "${this.selected.name}"?`)) return;
    this.service.delete(this.selected.id).subscribe({
      next: () => { this.load(); this.onCancel(); },
      error: (err) => console.error('Delete failed', err),
    });
  }

  onSync(): void {
    if (!this.selected) return;
    // Ensure this account is the selected one for the sync request
    this.service.selectAccount(this.selected.id);
    this.syncing = true;
    this.syncMessage = null;
    this.service.triggerSync().subscribe({
      next: (res) => {
        this.syncMessage = res?.message ?? 'Sync completed.';
        this.syncing = false;
        this.load();
      },
      error: (err) => {
        this.syncMessage = err.error?.message ?? 'Sync failed.';
        this.syncing = false;
        this.load();
      },
    });
  }

  private refreshSyncStatus(): void {
    this.service.getSyncStatus().subscribe({
      next: (status) => (this.syncStatus = status),
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
