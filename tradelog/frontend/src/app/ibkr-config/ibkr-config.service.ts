import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface IbkrConfig {
  id?: number;
  host: string;
  port: number;
  clientId: number;
  lastSyncAt?: string | null;
  lastSyncResult?: string | null;
}

export interface SyncStatus {
  lastSyncAt: string | null;
  lastSyncResult: string | null;
  canSync: boolean;
  cooldownRemainingSeconds: number | null;
}

@Injectable({ providedIn: 'root' })
export class IbkrConfigService {
  private http = inject(HttpClient);

  getConfig(): Observable<IbkrConfig> {
    return this.http.get<IbkrConfig>('/api/ibkr/config');
  }

  updateConfig(config: IbkrConfig): Observable<IbkrConfig> {
    return this.http.put<IbkrConfig>('/api/ibkr/config', config);
  }

  getSyncStatus(): Observable<SyncStatus> {
    return this.http.get<SyncStatus>('/api/ibkr/sync/status');
  }

  triggerSync(): Observable<any> {
    return this.http.post('/api/ibkr/sync', {});
  }
}
