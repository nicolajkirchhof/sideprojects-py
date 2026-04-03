import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, Observable } from 'rxjs';

export interface Account {
  id: number;
  ibkrAccountId: string;
  name: string;
  host: string;
  port: number;
  clientId: number;
  isDefault: boolean;
  flexToken?: string | null;
  flexQueryId?: string | null;
  lastSyncAt?: string | null;
  lastSyncResult?: string | null;
  lastFlexSyncAt?: string | null;
  lastFlexSyncResult?: string | null;
}

export interface SyncStatus {
  flexConfigured: boolean;
  lastFlexSyncAt: string | null;
  lastFlexSyncResult: string | null;
  lastLiveSyncAt: string | null;
  lastLiveSyncResult: string | null;
  canLiveSync: boolean;
  liveSyncCooldownSeconds: number | null;
}

const STORAGE_KEY = 'tradelog-selected-account-id';

@Injectable({ providedIn: 'root' })
export class AccountsService {
  private http = inject(HttpClient);

  private selectedAccountId$ = new BehaviorSubject<number>(this.loadSelectedId());

  get selectedAccountId(): number {
    return this.selectedAccountId$.value;
  }

  get selectedAccountId$Obs(): Observable<number> {
    return this.selectedAccountId$.asObservable();
  }

  selectAccount(id: number): void {
    localStorage.setItem(STORAGE_KEY, id.toString());
    this.selectedAccountId$.next(id);
  }

  private loadSelectedId(): number {
    const stored = localStorage.getItem(STORAGE_KEY);
    return stored ? parseInt(stored, 10) : 0;
  }

  // CRUD
  getAll(): Observable<Account[]> {
    return this.http.get<Account[]>('/api/accounts');
  }

  getById(id: number): Observable<Account> {
    return this.http.get<Account>(`/api/accounts/${id}`);
  }

  create(account: Partial<Account>): Observable<Account> {
    return this.http.post<Account>('/api/accounts', account);
  }

  update(id: number, account: Partial<Account>): Observable<void> {
    return this.http.put<void>(`/api/accounts/${id}`, account);
  }

  delete(id: number): Observable<void> {
    return this.http.delete<void>(`/api/accounts/${id}`);
  }

  // Sync
  getSyncStatus(): Observable<SyncStatus> {
    return this.http.get<SyncStatus>('/api/ibkr/sync/status');
  }

  triggerFlexSync(): Observable<any> {
    return this.http.post('/api/ibkr/flex-sync', {});
  }

  triggerLiveSync(): Observable<any> {
    return this.http.post('/api/ibkr/live-sync', {});
  }
}
