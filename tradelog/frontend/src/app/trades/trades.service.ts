import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';

// Enums are now DB-backed lookups — see LookupService.
// Trade fields store lookup IDs (int). Display names come from LookupService.name().

export interface Trade {
  id: number;
  symbol: string;
  date: string;
  typeOfTrade: number;
  notes?: string | null;
  directional?: number | null;
  timeframe?: number | null;
  budget: number;
  strategy: number;
  newsCatalyst: boolean;
  recentEarnings: boolean;
  sectorSupport: boolean;
  ath: boolean;
  rvol?: number | null;
  institutionalSupport?: string | null;
  gapPct?: number | null;
  xAtrMove?: number | null;
  taFaNotes?: string | null;
  intendedManagement?: string | null;
  actualManagement?: string | null;
  managementRating?: number | null;
  learnings?: string | null;
  parentTradeId?: number | null;
  status?: string | null;
}

export interface TradeDetail extends Trade {
  // Resolved names from the backend
  typeOfTradeName: string;
  directionalName?: string | null;
  timeframeName?: string | null;
  budgetName: string;
  strategyName: string;
  managementRatingName?: string | null;
  childTradeIds: number[];
  events: TradeEventDto[];
  optionPositions: OptionLegDto[];
  stockPositions: StockLegDto[];
}

export enum TradeEventType {
  ScaleIn = 'ScaleIn',
  ProfitTake = 'ProfitTake',
  Roll = 'Roll',
  Stop = 'Stop',
}

export const TRADE_EVENT_TYPE_LABELS: Record<TradeEventType, string> = {
  [TradeEventType.ScaleIn]: 'Scale In',
  [TradeEventType.ProfitTake]: 'Profit Take',
  [TradeEventType.Roll]: 'Roll',
  [TradeEventType.Stop]: 'Stop',
};

export interface TradeEventDto {
  id: number;
  tradeId: number;
  type: string;
  date: string;
  notes?: string | null;
  pnlImpact?: number | null;
}

export interface OptionLegDto {
  id: number;
  symbol: string;
  contractId: string;
  right: string;
  strike: number;
  expiry: string;
  pos: number;
  cost: number;
  closed?: string | null;
  unrealizedPnl?: number | null;
  realizedPnl?: number | null;
  delta?: number | null;
  theta?: number | null;
}

export interface StockLegDto {
  id: number;
  symbol: string;
  date: string;
  posChange: number;
  price: number;
  totalPos: number;
  pnl: number;
}

export type TradeUpsert = Omit<Trade, 'id'> & Partial<Pick<Trade, 'id'>>;

@Injectable({ providedIn: 'root' })
export class TradesService {
  private http = inject(HttpClient);

  getAll(filters?: { symbol?: string; budget?: number; strategy?: number }): Observable<Trade[]> {
    let params = new HttpParams();
    if (filters?.symbol) params = params.set('symbol', filters.symbol);
    if (filters?.budget != null) params = params.set('budget', filters.budget.toString());
    if (filters?.strategy != null) params = params.set('strategy', filters.strategy.toString());
    return this.http.get<Trade[]>('/api/trades', { params });
  }

  getById(id: number): Observable<TradeDetail> {
    return this.http.get<TradeDetail>(`/api/trades/${id}`);
  }

  create(payload: TradeUpsert): Observable<Trade> {
    return this.http.post<Trade>('/api/trades', payload);
  }

  update(id: number, payload: TradeUpsert): Observable<Trade> {
    return this.http.put<Trade>(`/api/trades/${id}`, payload);
  }

  delete(id: number): Observable<void> {
    return this.http.delete<void>(`/api/trades/${id}`);
  }

  // Trade events
  getEvents(tradeId: number): Observable<TradeEventDto[]> {
    return this.http.get<TradeEventDto[]>(`/api/trades/${tradeId}/events`);
  }

  createEvent(tradeId: number, event: Partial<TradeEventDto>): Observable<TradeEventDto> {
    return this.http.post<TradeEventDto>(`/api/trades/${tradeId}/events`, event);
  }

  deleteEvent(eventId: number): Observable<void> {
    return this.http.delete<void>(`/api/trade-events/${eventId}`);
  }

  getChain(id: number): Observable<Trade[]> {
    return this.http.get<Trade[]>(`/api/trades/${id}/chain`);
  }

  // Position assignment
  getUnassignedOptionPositions(symbol?: string): Observable<OptionLegDto[]> {
    let params = new HttpParams().set('unassigned', 'true').set('status', 'open');
    if (symbol) params = params.set('symbol', symbol);
    return this.http.get<OptionLegDto[]>('/api/option-positions', { params });
  }

  getUnassignedStockPositions(symbol?: string): Observable<StockLegDto[]> {
    let params = new HttpParams().set('unassigned', 'true');
    if (symbol) params = params.set('symbol', symbol);
    return this.http.get<StockLegDto[]>('/api/stock-positions', { params });
  }

  assignOptionPosition(positionId: number, tradeId: number | null): Observable<void> {
    return this.http.patch<void>(`/api/option-positions/${positionId}/assign`, { tradeId });
  }

  assignStockPosition(positionId: number, tradeId: number | null): Observable<void> {
    return this.http.patch<void>(`/api/stock-positions/${positionId}/assign`, { tradeId });
  }
}
