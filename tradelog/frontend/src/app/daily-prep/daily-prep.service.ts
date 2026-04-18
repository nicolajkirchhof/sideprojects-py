import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface DailyPrepDto {
  id: number;
  date: string;
  marketSummary: string | null;
  watchlist: string | null;
  emailCount: number;
  candidateCount: number;
  createdAt: string;
  updatedAt: string;
}

export interface MarketSummary {
  regime: string;
  regime_reasoning: string;
  themes: string[];
  movers: string[];
  risks: string[];
  action_items: string[];
}

export interface WatchlistCandidate {
  symbol: string;
  score: number;
  price: number | null;
  change_5d_pct: number | null;
  change_1m_pct: number | null;
  setup_type: string | null;
  profit_mechanism: string | null;
  confidence: string | null;
  thesis: string | null;
  reasoning: string | null;
  recommended_structure: string | null;
  entry: number | null;
  stop: number | null;
  target: number | null;
  boxes: { box: number; name: string; status: string; reason: string }[];
}

export interface TradeAnalysisDto {
  id: number;
  tradeId: number;
  analysisDate: string;
  score: number;
  analysis: string | null;
  model: string | null;
  createdAt: string;
}

export interface TradeAnalysisUpdate {
  score: number;
  analysis: string | null;
}

@Injectable({ providedIn: 'root' })
export class DailyPrepService {
  private http = inject(HttpClient);

  getLatest(): Observable<DailyPrepDto> {
    return this.http.get<DailyPrepDto>('/api/daily-prep/latest');
  }

  getByDate(date: string): Observable<DailyPrepDto> {
    return this.http.get<DailyPrepDto>('/api/daily-prep', { params: { date } });
  }

  getTradeAnalyses(tradeId: number): Observable<TradeAnalysisDto[]> {
    return this.http.get<TradeAnalysisDto[]>(`/api/trades/${tradeId}/analysis`);
  }

  updateTradeAnalysis(tradeId: number, analysisId: number, dto: TradeAnalysisUpdate): Observable<TradeAnalysisDto> {
    return this.http.put<TradeAnalysisDto>(`/api/trades/${tradeId}/analysis/${analysisId}`, dto);
  }
}
