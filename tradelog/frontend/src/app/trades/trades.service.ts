import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';

export enum Budget {
  Drift = 'Drift',
  Speculative = 'Speculative',
  LongTerm = 'LongTerm',
  Swing = 'Swing'
}

export enum Strategy {
  PositiveDrift = 'PositiveDrift',
  RangeBound = 'RangeBound',
  PEAD = 'PEAD',
  BreakoutMomentum = 'BreakoutMomentum',
  IVMeanReversion = 'IVMeanReversion',
  SectorStrength = 'SectorStrength',
  SectorWeakness = 'SectorWeakness',
  GreenLineBreakout = 'GreenLineBreakout',
  Slingshot = 'Slingshot',
  PreEarnings = 'PreEarnings'
}

export enum TypeOfTrade {
  ShortStrangle = 'ShortStrangle',
  ShortPutSpread = 'ShortPutSpread',
  ShortCallSpread = 'ShortCallSpread',
  LongCall = 'LongCall',
  LongPut = 'LongPut',
  LongCallVertical = 'LongCallVertical',
  LongPutVertical = 'LongPutVertical',
  SyntheticLong = 'SyntheticLong',
  CoveredStrangle = 'CoveredStrangle',
  Butterfly = 'Butterfly',
  RatioDiagonalSpread = 'RatioDiagonalSpread',
  LongStrangle = 'LongStrangle',
  ShortPut = 'ShortPut',
  ShortCall = 'ShortCall',
  LongStock = 'LongStock',
  ShortStock = 'ShortStock',
  IronCondor = 'IronCondor',
  XYZ = 'XYZ',
  PMCC = 'PMCC',
  CalendarSpread = 'CalendarSpread'
}

export enum DirectionalBias {
  Bullish = 'Bullish',
  Neutral = 'Neutral',
  Bearish = 'Bearish'
}

export enum Timeframe {
  OneDay = 'OneDay',
  OneWeek = 'OneWeek',
  DeltaBand = 'DeltaBand'
}

export enum ManagementRating {
  AsPlanned = 'AsPlanned',
  MinorAdjustments = 'MinorAdjustments',
  Rogue = 'Rogue'
}

export interface Trade {
  id: number;
  symbol: string;
  date: string;
  typeOfTrade: TypeOfTrade;
  notes?: string | null;
  directional?: DirectionalBias | null;
  timeframe?: Timeframe | null;
  budget: Budget;
  strategy: Strategy;
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
  managementRating?: ManagementRating | null;
  learnings?: string | null;
}

export interface TradeDetail extends Trade {
  optionPositions: OptionLegDto[];
  stockPositions: StockLegDto[];
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

// Display labels for enums
export const STRATEGY_LABELS: Record<Strategy, string> = {
  [Strategy.PositiveDrift]: 'Positive Drift',
  [Strategy.RangeBound]: 'Range Bound',
  [Strategy.PEAD]: 'PEAD',
  [Strategy.BreakoutMomentum]: 'Breakout Momentum',
  [Strategy.IVMeanReversion]: 'IV Mean Reversion',
  [Strategy.SectorStrength]: 'Sector Strength',
  [Strategy.SectorWeakness]: 'Sector Weakness',
  [Strategy.GreenLineBreakout]: 'Green Line Breakout',
  [Strategy.Slingshot]: 'Slingshot',
  [Strategy.PreEarnings]: 'Pre Earnings',
};

export const TYPE_OF_TRADE_LABELS: Record<TypeOfTrade, string> = {
  [TypeOfTrade.ShortStrangle]: 'Short Strangle',
  [TypeOfTrade.ShortPutSpread]: 'Short Put Spread',
  [TypeOfTrade.ShortCallSpread]: 'Short Call Spread',
  [TypeOfTrade.LongCall]: 'Long Call',
  [TypeOfTrade.LongPut]: 'Long Put',
  [TypeOfTrade.LongCallVertical]: 'Long Call Vertical',
  [TypeOfTrade.LongPutVertical]: 'Long Put Vertical',
  [TypeOfTrade.SyntheticLong]: 'Synthetic Long',
  [TypeOfTrade.CoveredStrangle]: 'Covered Strangle',
  [TypeOfTrade.Butterfly]: 'Butterfly',
  [TypeOfTrade.RatioDiagonalSpread]: 'Ratio Diagonal Spread',
  [TypeOfTrade.LongStrangle]: 'Long Strangle',
  [TypeOfTrade.ShortPut]: 'Short Put',
  [TypeOfTrade.ShortCall]: 'Short Call',
  [TypeOfTrade.LongStock]: 'Long Stock',
  [TypeOfTrade.ShortStock]: 'Short Stock',
  [TypeOfTrade.IronCondor]: 'Iron Condor',
  [TypeOfTrade.XYZ]: 'XYZ',
  [TypeOfTrade.PMCC]: 'PMCC',
  [TypeOfTrade.CalendarSpread]: 'Calendar Spread',
};

export const TIMEFRAME_LABELS: Record<Timeframe, string> = {
  [Timeframe.OneDay]: '1 Day',
  [Timeframe.OneWeek]: '1 Week',
  [Timeframe.DeltaBand]: 'Delta Band',
};

export const MANAGEMENT_RATING_LABELS: Record<ManagementRating, string> = {
  [ManagementRating.AsPlanned]: 'As Planned',
  [ManagementRating.MinorAdjustments]: 'Minor Adjustments',
  [ManagementRating.Rogue]: 'Rogue',
};

@Injectable({ providedIn: 'root' })
export class TradesService {
  private http = inject(HttpClient);

  getAll(filters?: { symbol?: string; budget?: Budget; strategy?: Strategy }): Observable<Trade[]> {
    let params = new HttpParams();
    if (filters?.symbol) params = params.set('symbol', filters.symbol);
    if (filters?.budget) params = params.set('budget', filters.budget);
    if (filters?.strategy) params = params.set('strategy', filters.strategy);
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
