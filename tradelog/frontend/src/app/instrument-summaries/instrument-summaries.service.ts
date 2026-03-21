import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface OptionInstrumentSummary {
  symbol: string;
  opened?: string | null;
  closed?: string | null;
  dit: number;
  dte?: number | null;
  status: string;
  budget?: string | null;
  currentSetup?: string | null;
  strikes?: string | null;
  intendedManagement?: string | null;
  pnl: number;
  unrealizedPnl: number;
  unrealizedPnlPct?: number | null;
  realizedPnl: number;
  realizedPnlPct?: number | null;
  timeValue: number;
  delta: number;
  theta: number;
  gamma: number;
  vega: number;
  avgIv?: number | null;
  margin: number;
  durationPct?: number | null;
  pnlPerDurationPct?: number | null;
  roic?: number | null;
  commissions: number;
}

export interface TradeInstrumentSummary {
  symbol: string;
  status: string;
  budget?: string | null;
  positionType?: string | null;
  intendedManagement?: string | null;
  totalPos: number;
  avgPrice: number;
  multiplier: number;
  pnl: number;
  unrealizedPnl: number;
  unrealizedPnlPct?: number | null;
  realizedPnl: number;
  commissions: number;
}

@Injectable({ providedIn: 'root' })
export class InstrumentSummariesService {
  private http = inject(HttpClient);

  getOptionSummaries(status?: string): Observable<OptionInstrumentSummary[]> {
    let params = new HttpParams();
    if (status) params = params.set('status', status);
    return this.http.get<OptionInstrumentSummary[]>('/api/instrument-summaries/options', { params });
  }

  getTradeSummaries(): Observable<TradeInstrumentSummary[]> {
    return this.http.get<TradeInstrumentSummary[]>('/api/instrument-summaries/trades');
  }
}
