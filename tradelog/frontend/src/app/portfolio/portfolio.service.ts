import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface PortfolioDto {
  id: number;
  budget: string;
  strategy: string;
  minAllocation: number;
  maxAllocation: number;
  currentAllocation: number;
  pnl: number;
}

export interface PortfolioUpsert {
  id?: number;
  budget: string;
  strategy: string;
  minAllocation: number;
  maxAllocation: number;
}

@Injectable({ providedIn: 'root' })
export class PortfolioService {
  private http = inject(HttpClient);

  getAll(): Observable<PortfolioDto[]> {
    return this.http.get<PortfolioDto[]>('/api/portfolio');
  }

  create(payload: PortfolioUpsert): Observable<PortfolioDto> {
    return this.http.post<PortfolioDto>('/api/portfolio', payload);
  }

  update(id: number, payload: PortfolioUpsert): Observable<void> {
    return this.http.put<void>(`/api/portfolio/${id}`, payload);
  }

  delete(id: number): Observable<void> {
    return this.http.delete<void>(`/api/portfolio/${id}`);
  }
}
