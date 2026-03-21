import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface WeeklyPrep {
  id: number;
  date: string;
  indexBias?: string | null;
  breadth?: string | null;
  notableSectors?: string | null;
  volatilityNotes?: string | null;
  openPositionsRequiringManagement?: string | null;
  currentPortfolioRisk?: string | null;
  portfolioNotes?: string | null;
  scanningFor?: string | null;
  indexSectorPreference?: string | null;
  watchlist?: string | null;
  learnings?: string | null;
  focusForImprovement?: string | null;
  externalComments?: string | null;
}

export type WeeklyPrepUpsert = Omit<WeeklyPrep, 'id'> & Partial<Pick<WeeklyPrep, 'id'>>;

@Injectable({ providedIn: 'root' })
export class WeeklyPrepService {
  private http = inject(HttpClient);

  getAll(filters?: { year?: number }): Observable<WeeklyPrep[]> {
    let params = new HttpParams();
    if (filters?.year) params = params.set('year', filters.year.toString());
    return this.http.get<WeeklyPrep[]>('/api/weekly-prep', { params });
  }

  getById(id: number): Observable<WeeklyPrep> {
    return this.http.get<WeeklyPrep>(`/api/weekly-prep/${id}`);
  }

  create(payload: WeeklyPrepUpsert): Observable<WeeklyPrep> {
    return this.http.post<WeeklyPrep>('/api/weekly-prep', payload);
  }

  update(id: number, payload: WeeklyPrepUpsert): Observable<WeeklyPrep> {
    return this.http.put<WeeklyPrep>(`/api/weekly-prep/${id}`, payload);
  }

  delete(id: number): Observable<void> {
    return this.http.delete<void>(`/api/weekly-prep/${id}`);
  }
}
