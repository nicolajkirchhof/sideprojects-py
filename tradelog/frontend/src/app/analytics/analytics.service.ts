import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface StrategyPerformance {
  strategy: string;
  tradeCount: number;
  totalPnl: number;
  avgWin: number;
  avgLoss: number;
  winRate: number;
  expectancy: number;
  maxDrawdown: number;
  totalCommissions: number;
}

export interface BudgetPerformance {
  budget: string;
  tradeCount: number;
  totalPnl: number;
  avgWin: number;
  avgLoss: number;
  winRate: number;
  expectancy: number;
  totalCommissions: number;
}

export interface OverallPerformance {
  totalPnl: number;
  totalCommissions: number;
  netPnl: number;
  dailyPnl: number;
  annualizedRoi: number;
  tradingDays: number;
  tradeCount: number;
  winRate: number;
  avgWin: number;
  avgLoss: number;
}

export interface EquityCurvePoint {
  date: string;
  cumulativePnl: number;
}

export interface ChainSummary {
  rootTradeId: number;
  symbol: string;
  strategy: string;
  budget: string;
  chainLength: number;
  status: string;
  startDate: string;
  totalPnl: number;
  premiumCollected: number;
  premiumLost: number;
  eventCount: number;
}

@Injectable({ providedIn: 'root' })
export class AnalyticsService {
  private http = inject(HttpClient);

  getStrategies(): Observable<StrategyPerformance[]> {
    return this.http.get<StrategyPerformance[]>('/api/analytics/strategies');
  }

  getStrategyEquityCurve(strategy: string): Observable<EquityCurvePoint[]> {
    return this.http.get<EquityCurvePoint[]>(`/api/analytics/strategies/${encodeURIComponent(strategy)}/equity-curve`);
  }

  getBudgets(): Observable<BudgetPerformance[]> {
    return this.http.get<BudgetPerformance[]>('/api/analytics/budgets');
  }

  getBudgetEquityCurve(budget: string): Observable<EquityCurvePoint[]> {
    return this.http.get<EquityCurvePoint[]>(`/api/analytics/budgets/${encodeURIComponent(budget)}/equity-curve`);
  }

  getOverall(): Observable<OverallPerformance> {
    return this.http.get<OverallPerformance>('/api/analytics/overall');
  }

  getOverallEquityCurve(): Observable<EquityCurvePoint[]> {
    return this.http.get<EquityCurvePoint[]>('/api/analytics/overall/equity-curve');
  }

  getChains(): Observable<ChainSummary[]> {
    return this.http.get<ChainSummary[]>('/api/analytics/chains');
  }
}
