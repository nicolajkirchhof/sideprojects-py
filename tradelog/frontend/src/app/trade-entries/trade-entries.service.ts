import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';

export enum Budget {
  Core = 'Core',
  Speculative = 'Speculative'
}

export enum Strategy {
  PositiveDrift = 'PositiveDrift',
  RangeBoundCommodities = 'RangeBoundCommodities',
  PEADS = 'PEADS',
  Momentum = 'Momentum',
  IVMeanReversion = 'IVMeanReversion',
  SectorStrength = 'SectorStrength',
  SectorWeakness = 'SectorWeakness',
  Breakout = 'Breakout',
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
  ShortStock = 'ShortStock'
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

export interface TradeEntry {
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

export type TradeEntryUpsert = Omit<TradeEntry, 'id'> & Partial<Pick<TradeEntry, 'id'>>;

// Display labels for enums
export const STRATEGY_LABELS: Record<Strategy, string> = {
  [Strategy.PositiveDrift]: 'Positive Drift',
  [Strategy.RangeBoundCommodities]: 'Range Bound Commodities',
  [Strategy.PEADS]: 'PEADS',
  [Strategy.Momentum]: 'Momentum',
  [Strategy.IVMeanReversion]: 'IV Mean Reversion',
  [Strategy.SectorStrength]: 'Sector Strength',
  [Strategy.SectorWeakness]: 'Sector Weakness',
  [Strategy.Breakout]: 'Breakout',
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
export class TradeEntriesService {
  private http = inject(HttpClient);

  getAll(filters?: { symbol?: string; budget?: Budget; strategy?: Strategy }): Observable<TradeEntry[]> {
    let params = new HttpParams();
    if (filters?.symbol) params = params.set('symbol', filters.symbol);
    if (filters?.budget) params = params.set('budget', filters.budget);
    if (filters?.strategy) params = params.set('strategy', filters.strategy);
    return this.http.get<TradeEntry[]>('/api/trade-entries', { params });
  }

  getById(id: number): Observable<TradeEntry> {
    return this.http.get<TradeEntry>(`/api/trade-entries/${id}`);
  }

  create(payload: TradeEntryUpsert): Observable<TradeEntry> {
    return this.http.post<TradeEntry>('/api/trade-entries', payload);
  }

  update(id: number, payload: TradeEntryUpsert): Observable<TradeEntry> {
    return this.http.put<TradeEntry>(`/api/trade-entries/${id}`, payload);
  }

  delete(id: number): Observable<void> {
    return this.http.delete<void>(`/api/trade-entries/${id}`);
  }
}
