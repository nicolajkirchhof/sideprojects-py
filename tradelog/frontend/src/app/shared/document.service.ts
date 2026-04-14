import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface DocumentDto {
  id: number;
  title: string;
  content: string | null;
  updatedAt: string;
  strategyIds: number[];
}

export interface DocumentUpsert {
  title: string;
  content: string | null;
  strategyIds?: number[];
}

@Injectable({ providedIn: 'root' })
export class DocumentService {
  private http = inject(HttpClient);

  getAll(): Observable<DocumentDto[]> { return this.http.get<DocumentDto[]>('/api/documents'); }
  getById(id: number): Observable<DocumentDto> { return this.http.get<DocumentDto>(`/api/documents/${id}`); }
  create(dto: DocumentUpsert): Observable<DocumentDto> { return this.http.post<DocumentDto>('/api/documents', dto); }
  update(id: number, dto: DocumentUpsert): Observable<void> { return this.http.put<void>(`/api/documents/${id}`, dto); }
  delete(id: number): Observable<void> { return this.http.delete<void>(`/api/documents/${id}`); }
}
