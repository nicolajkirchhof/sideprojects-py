import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

// Mirror of backend enums
export enum PositionTypes {
  Call = 'Call',
  Put = 'Put',
  Underlying = 'Underlying',
}

// [Flags] enum mirror. Use bitmask number in payloads.
export enum CloseReasons {
  TakeLoss = 1,
  TakeProfit = 2,
  Roll = 4,
  AssumptionInvalidated = 8,
  TimeLimit = 16,
  Other = 32,
}

// Mirror of backend Position class (using camelCase JSON names)
export interface Position {
  id: number;
  instrumentId: number;
  contractId: string;
  type: PositionTypes | number; // backend may serialize enum as number or string
  opened: string; // ISO date string
  expiry: string; // ISO date string
  closed?: string | null; // ISO date string or null
  size: number;
  strike: number;
  cost: number;
  close?: number | null;
  comission?: number | null;
  multiplier: number;
  closeReasons?: number | null; // bitmask of CloseReasons
}

// DTO for create/update without id when creating
export type PositionUpsert = Omit<Position, 'id'> & Partial<Pick<Position, 'id'>>;

@Injectable({ providedIn: 'root' })
export class PositionsService {
  private http = inject(HttpClient);

  getPositions(): Observable<Position[]> {
    return this.http.get<Position[]>('/api/positions');
  }

  createPosition(payload: PositionUpsert): Observable<Position> {
    return this.http.post<Position>('/api/positions', payload);
  }

  updatePosition(id: number, payload: PositionUpsert): Observable<Position> {
    return this.http.put<Position>(`/api/positions/${id}`, payload);
  }
}
