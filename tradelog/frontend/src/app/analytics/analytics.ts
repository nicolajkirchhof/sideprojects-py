import { Component, OnInit, inject } from '@angular/core';
import { CommonModule, DecimalPipe } from '@angular/common';
import { forkJoin } from 'rxjs';
import { MatTableModule } from '@angular/material/table';
import { MatCardModule } from '@angular/material/card';
import { MatButtonToggleModule } from '@angular/material/button-toggle';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { BaseChartDirective } from 'ng2-charts';
import { ChartConfiguration } from 'chart.js/auto';
import {
  AnalyticsService,
  StrategyPerformance,
  BudgetPerformance,
  OverallPerformance,
  EquityCurvePoint,
} from './analytics.service';
import { pnlColor } from '../shared/utils';
import { NotificationService } from '../shared/notification.service';

@Component({
  selector: 'app-analytics',
  standalone: true,
  imports: [
    CommonModule,
    MatTableModule,
    MatCardModule,
    MatButtonToggleModule,
    MatProgressBarModule,
    BaseChartDirective,
    DecimalPipe,
  ],
  templateUrl: './analytics.html',
  host: { class: 'flex flex-col flex-1 overflow-auto' },
})
export class AnalyticsComponent implements OnInit {
  private service = inject(AnalyticsService);
  private notify = inject(NotificationService);

  loading = false;
  overall: OverallPerformance | null = null;
  strategies: StrategyPerformance[] = [];
  budgets: BudgetPerformance[] = [];

  strategyColumns = ['strategy', 'tradeCount', 'totalPnl', 'winRate', 'avgWin', 'avgLoss', 'expectancy', 'maxDrawdown'];
  budgetColumns = ['budget', 'tradeCount', 'totalPnl', 'winRate', 'avgWin', 'avgLoss', 'expectancy'];

  pnlColor = pnlColor;

  // Equity curve chart
  curveMode: 'overall' | 'strategy' | 'budget' = 'overall';
  chartData: ChartConfiguration<'line'>['data'] = { labels: [], datasets: [] };
  chartOptions: ChartConfiguration<'line'>['options'] = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { labels: { color: '#ccc' } },
    },
    scales: {
      x: { ticks: { color: '#999', maxTicksLimit: 20 }, grid: { color: '#333' } },
      y: { ticks: { color: '#ccc' }, grid: { color: '#333' } },
    },
  };

  private readonly COLORS = ['#48dbfb', '#ff6b6b', '#feca57', '#54a0ff', '#5f27cd', '#01a3a4', '#f368e0', '#ff9f43'];

  ngOnInit(): void {
    this.loading = true;
    let remaining = 3;
    const done = () => { if (--remaining === 0) this.loading = false; };

    this.service.getOverall().subscribe({
      next: (data) => { this.overall = data; done(); },
      error: () => { this.notify.error('Failed to load overall analytics'); done(); },
    });
    this.service.getStrategies().subscribe({
      next: (data) => { this.strategies = data; done(); },
      error: () => { this.notify.error('Failed to load strategy data'); done(); },
    });
    this.service.getBudgets().subscribe({
      next: (data) => { this.budgets = data; done(); },
      error: () => { this.notify.error('Failed to load budget data'); done(); },
    });
    this.loadEquityCurve();
  }

  onCurveModeChange(): void {
    this.loadEquityCurve();
  }

  private loadEquityCurve(): void {
    if (this.curveMode === 'overall') {
      this.service.getOverallEquityCurve().subscribe({
        next: (curve) => this.setChartSingle('Portfolio', curve),
        error: () => this.notify.error('Failed to load equity curve'),
      });
    } else if (this.curveMode === 'strategy') {
      this.service.getStrategies().subscribe({
        next: (strats) => {
          if (strats.length === 0) {
            this.chartData = { labels: [], datasets: [] };
            return;
          }
          const curveRequests = strats.map(s =>
            this.service.getStrategyEquityCurve(s.strategy));
          forkJoin(curveRequests).subscribe({
            next: (curves) => {
              const datasets = strats.map((s, i) => ({ label: s.strategy, data: curves[i] }));
              this.setChartMulti(datasets);
            },
            error: () => this.notify.error('Failed to load strategy equity curves'),
          });
        },
      });
    } else {
      this.service.getBudgets().subscribe({
        next: (budgets) => {
          if (budgets.length === 0) {
            this.chartData = { labels: [], datasets: [] };
            return;
          }
          const curveRequests = budgets.map(b =>
            this.service.getBudgetEquityCurve(b.budget));
          forkJoin(curveRequests).subscribe({
            next: (curves) => {
              const datasets = budgets.map((b, i) => ({ label: b.budget, data: curves[i] }));
              this.setChartMulti(datasets);
            },
            error: () => this.notify.error('Failed to load budget equity curves'),
          });
        },
      });
    }
  }

  private setChartSingle(label: string, curve: EquityCurvePoint[]): void {
    this.chartData = {
      labels: curve.map(p => new Date(p.date).toLocaleDateString()),
      datasets: [
        {
          label,
          data: curve.map(p => p.cumulativePnl),
          borderColor: '#48dbfb',
          backgroundColor: 'rgba(72, 219, 251, 0.1)',
          fill: true,
          tension: 0.3,
        },
      ],
    };
  }

  private setChartMulti(datasets: { label: string; data: EquityCurvePoint[] }[]): void {
    // Collect all unique dates across all curves
    const allDates = new Set<string>();
    datasets.forEach(ds => ds.data.forEach(p => allDates.add(p.date)));
    const sortedDates = [...allDates].sort();
    const labels = sortedDates.map(d => new Date(d).toLocaleDateString());

    this.chartData = {
      labels,
      datasets: datasets.map((ds, i) => {
        const dateMap = new Map(ds.data.map(p => [p.date, p.cumulativePnl]));
        return {
          label: ds.label,
          data: sortedDates.map(d => dateMap.get(d) ?? null) as (number | null)[],
          borderColor: this.COLORS[i % this.COLORS.length],
          fill: false,
          tension: 0.3,
          spanGaps: true,
        };
      }),
    };
  }
}
