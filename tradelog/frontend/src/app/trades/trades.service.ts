import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface TradeDto {
  id: number;
  symbol: string;
  date: string;
  posChange: number;
  price: number;
  commission: number;
  multiplier: number;
  bestExitPrice?: number | null;
  bestExitDate?: string | null;

  // Computed
  lastPos: number;
  totalPos: number;
  avgPrice: number;
  pnl: number;
}

export interface TradeUpsert {
  id?: number;
  symbol: string;
  date: string;
  posChange: number;
  price: number;
  commission: number;
  multiplier: number;
  bestExitPrice?: number | null;
  bestExitDate?: string | null;
}

@Injectable({ providedIn: 'root' })
export class TradesService {
  private http = inject(HttpClient);

  getAll(filters?: { symbol?: string }): Observable<TradeDto[]> {
    let params = new HttpParams();
    if (filters?.symbol) params = params.set('symbol', filters.symbol);
    return this.http.get<TradeDto[]>('/api/trades', { params });
  }

  getById(id: number): Observable<TradeDto> {
    return this.http.get<TradeDto>(`/api/trades/${id}`);
  }

  create(payload: TradeUpsert): Observable<TradeDto> {
    return this.http.post<TradeDto>('/api/trades', payload);
  }

  update(id: number, payload: TradeUpsert): Observable<TradeDto> {
    return this.http.put<TradeDto>(`/api/trades/${id}`, payload);
  }

  delete(id: number): Observable<void> {
    return this.http.delete<void>(`/api/trades/${id}`);
  }
}
