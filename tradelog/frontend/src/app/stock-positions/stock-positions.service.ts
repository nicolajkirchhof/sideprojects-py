import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface StockPositionDto {
  id: number;
  symbol: string;
  date: string;
  posChange: number;
  price: number;
  commission: number;
  multiplier: number;
  bestExitPrice?: number | null;
  bestExitDate?: string | null;
  tradeId?: number | null;

  // Computed
  lastPos: number;
  totalPos: number;
  avgPrice: number;
  pnl: number;
}

export interface StockPositionUpsert {
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
export class StockPositionsService {
  private http = inject(HttpClient);

  getAll(filters?: { symbol?: string }): Observable<StockPositionDto[]> {
    let params = new HttpParams();
    if (filters?.symbol) params = params.set('symbol', filters.symbol);
    return this.http.get<StockPositionDto[]>('/api/stock-positions', { params });
  }

  getById(id: number): Observable<StockPositionDto> {
    return this.http.get<StockPositionDto>(`/api/stock-positions/${id}`);
  }

  create(payload: StockPositionUpsert): Observable<StockPositionDto> {
    return this.http.post<StockPositionDto>('/api/stock-positions', payload);
  }

  update(id: number, payload: StockPositionUpsert): Observable<StockPositionDto> {
    return this.http.put<StockPositionDto>(`/api/stock-positions/${id}`, payload);
  }

  delete(id: number): Observable<void> {
    return this.http.delete<void>(`/api/stock-positions/${id}`);
  }
}
