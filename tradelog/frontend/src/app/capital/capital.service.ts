import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface Capital {
  id: number;
  date: string;
  netLiquidity: number;
  maintenance: number;
  excessLiquidity: number;
  bpr: number;
  // Snapshotted aggregations
  maintenancePct: number;
  totalPnl: number;
  unrealizedPnl: number;
  realizedPnl: number;
  netDelta: number;
  netTheta: number;
  netVega: number;
  netGamma: number;
  avgIv: number;
  totalMargin: number;
  totalCommissions: number;
}

export interface CapitalUpsert {
  id?: number;
  date: string;
  netLiquidity: number;
  maintenance: number;
  excessLiquidity: number;
  bpr: number;
}

@Injectable({ providedIn: 'root' })
export class CapitalService {
  private http = inject(HttpClient);

  getAll(): Observable<Capital[]> {
    return this.http.get<Capital[]>('/api/capital');
  }

  getById(id: number): Observable<Capital> {
    return this.http.get<Capital>(`/api/capital/${id}`);
  }

  create(payload: CapitalUpsert): Observable<Capital> {
    return this.http.post<Capital>('/api/capital', payload);
  }

  update(id: number, payload: CapitalUpsert): Observable<Capital> {
    return this.http.put<Capital>(`/api/capital/${id}`, payload);
  }

  delete(id: number): Observable<void> {
    return this.http.delete<void>(`/api/capital/${id}`);
  }
}
