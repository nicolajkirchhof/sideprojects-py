import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

// Mirror of backend enum
export enum SecTypes {
  Stock = 'Stock',
  Future = 'Future',
  Forex = 'Forex',
}

// Mirror of backend Instrument class (camelCase property names as sent by JSON)
export interface Instrument {
  id: number;
  secType: SecTypes | number; // backend may serialize enum as string or number
  symbol: string;
  multiplier: number;
  sector: string;
  subsector?: string | null;
}

// DTO for create/update without id when creating
export type InstrumentUpsert = Omit<Instrument, 'id'> & Partial<Pick<Instrument, 'id'>>;

@Injectable({ providedIn: 'root' })
export class InstrumentsService {
  private http = inject(HttpClient);

  getInstruments(): Observable<Instrument[]> {
    return this.http.get<Instrument[]>('/api/instruments');
  }

  createInstrument(payload: InstrumentUpsert): Observable<Instrument> {
    return this.http.post<Instrument>('/api/instruments', payload);
  }

  updateInstrument(id: number, payload: InstrumentUpsert): Observable<Instrument> {
    return this.http.put<Instrument>(`/api/instruments/${id}`, payload);
  }
}
