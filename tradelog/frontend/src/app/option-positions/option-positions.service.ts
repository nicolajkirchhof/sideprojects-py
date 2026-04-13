import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';

export enum PositionRight {
  Call = 'Call',
  Put = 'Put'
}

export const POSITION_RIGHT_LABELS: Record<string, string> = {
  [PositionRight.Call]: 'C',
  [PositionRight.Put]: 'P',
};

export interface OptionPositionDto {
  id: number;
  symbol: string;
  contractId: string;
  opened: string;
  expiry: string;
  closed?: string | null;
  pos: number;
  right: PositionRight;
  strike: number;
  cost: number;
  closePrice?: number | null;
  commission: number;
  multiplier: number;
  closeReasons?: number | null;
  bestExitPrice?: number | null;
  bestExitDate?: string | null;

  // From latest log
  lastPrice?: number | null;
  lastValue?: number | null;
  timeValue?: number | null;
  delta?: number | null;
  theta?: number | null;
  gamma?: number | null;
  vega?: number | null;
  iv?: number | null;
  margin?: number | null;

  // Computed
  unrealizedPnl?: number | null;
  unrealizedPnlPct?: number | null;
  realizedPnl?: number | null;
  realizedPnlPct?: number | null;
  durationPct?: number | null;
  roic?: number | null;

  /** Cached count of greek log samples recorded since the position was opened. Null = never computed. */
  logCount: number | null;

  /** FK to the linked Trade (null if unlinked). */
  tradeId?: number | null;
}

export interface OptionPositionUpsert {
  id?: number;
  symbol: string;
  contractId: string;
  opened: string;
  expiry: string;
  closed?: string | null;
  pos: number;
  right: PositionRight;
  strike: number;
  cost: number;
  closePrice?: number | null;
  commission: number;
  multiplier: number;
  closeReasons?: number | null;
  bestExitPrice?: number | null;
  bestExitDate?: string | null;
}

export interface OptionPositionsLog {
  id?: number;
  dateTime: string;
  contractId: string;
  underlying: number;
  iv: number;
  price: number;
  timeValue: number;
  delta: number;
  theta: number;
  gamma: number;
  vega: number;
  margin: number;
}

@Injectable({ providedIn: 'root' })
export class OptionPositionsService {
  private http = inject(HttpClient);

  getAll(filters?: { symbol?: string; status?: string }): Observable<OptionPositionDto[]> {
    let params = new HttpParams();
    if (filters?.symbol) params = params.set('symbol', filters.symbol);
    if (filters?.status) params = params.set('status', filters.status);
    return this.http.get<OptionPositionDto[]>('/api/option-positions', { params });
  }

  getById(id: number): Observable<OptionPositionDto> {
    return this.http.get<OptionPositionDto>(`/api/option-positions/${id}`);
  }

  create(payload: OptionPositionUpsert): Observable<OptionPositionDto> {
    return this.http.post<OptionPositionDto>('/api/option-positions', payload);
  }

  update(id: number, payload: OptionPositionUpsert): Observable<OptionPositionDto> {
    return this.http.put<OptionPositionDto>(`/api/option-positions/${id}`, payload);
  }

  delete(id: number): Observable<void> {
    return this.http.delete<void>(`/api/option-positions/${id}`);
  }
}

@Injectable({ providedIn: 'root' })
export class OptionPositionsLogService {
  private http = inject(HttpClient);

  getByContract(contractId: string): Observable<OptionPositionsLog[]> {
    return this.http.get<OptionPositionsLog[]>('/api/option-positions-log', {
      params: new HttpParams().set('contractId', contractId)
    });
  }

  getLatest(): Observable<OptionPositionsLog[]> {
    return this.http.get<OptionPositionsLog[]>('/api/option-positions-log/latest');
  }

  bulkInsert(entries: OptionPositionsLog[]): Observable<{ inserted: number; skipped: number }> {
    return this.http.post<{ inserted: number; skipped: number }>('/api/option-positions-log/bulk', entries);
  }

  getLastSync(): Observable<{ lastSync: string | null }> {
    return this.http.get<{ lastSync: string | null }>('/api/option-positions-log/last-sync');
  }
}
