import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

// Flags enum mirror
export enum Sentiments {
  None = 0,
  Bullish = 1,
  Neutral = 2,
  Bearish = 4,
}

// [Flags] enum mirror for ProfitMechanisms
export enum ProfitMechanisms {
  Momentum = 1,
  Time = 2,
  Volatility = 4,
  Drift = 8,
  Other = 16,
}

// Mirror of backend Log model (camelCase JSON)
export interface LogEntry {
  id: number;
  instrumentId: number;
  date: string; // ISO date string
  notes?: string | null;
  profitMechanism?: number | null; // flags bitmask
  sentiment?: number | null; // flags bitmask
  ta?: string | null;
  expectedOutcome?: string | null;
  lernings?: string | null; // note: spelling per backend
  fa?: string | null;
}

// DTO for create/update without id when creating
export type LogUpsert = Omit<LogEntry, 'id'> & Partial<Pick<LogEntry, 'id'>>;

@Injectable({ providedIn: 'root' })
export class LogsService {
  private http = inject(HttpClient);

  getLogs(): Observable<LogEntry[]> {
    return this.http.get<LogEntry[]>('/api/logs');
  }

  createLog(payload: LogUpsert): Observable<LogEntry> {
    return this.http.post<LogEntry>('/api/logs', payload);

  }

  updateLog(id: number, payload: LogUpsert): Observable<LogEntry> {
    return this.http.put<LogEntry>(`/api/logs/${id}`, payload);
  }
}
