import { Component, OnInit, inject } from '@angular/core';
import { CommonModule, DecimalPipe } from '@angular/common';
import { MatTableModule } from '@angular/material/table';
import { MatCardModule } from '@angular/material/card';
import { MatButtonToggleModule } from '@angular/material/button-toggle';
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

@Component({
  selector: 'app-analytics',
  standalone: true,
  imports: [
    CommonModule,
    MatTableModule,
    MatCardModule,
    MatButtonToggleModule,
    BaseChartDirective,
    DecimalPipe,
  ],
  templateUrl: './analytics.html',
  host: { class: 'flex flex-col flex-1 overflow-auto' },
})
export class AnalyticsComponent implements OnInit {
  private service = inject(AnalyticsService);

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
    this.service.getOverall().subscribe({
      next: (data) => (this.overall = data),
    });
    this.service.getStrategies().subscribe({
      next: (data) => (this.strategies = data),
    });
    this.service.getBudgets().subscribe({
      next: (data) => (this.budgets = data),
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
      });
    } else if (this.curveMode === 'strategy') {
      // Load one curve per strategy
      this.service.getStrategies().subscribe({
        next: (strats) => {
          const datasets: { label: string; data: EquityCurvePoint[] }[] = [];
          let remaining = strats.length;
          if (remaining === 0) {
            this.chartData = { labels: [], datasets: [] };
            return;
          }
          strats.forEach((s, i) => {
            this.service.getStrategyEquityCurve(s.strategy).subscribe({
              next: (curve) => {
                datasets.push({ label: s.strategy, data: curve });
                remaining--;
                if (remaining === 0) this.setChartMulti(datasets);
              },
            });
          });
        },
      });
    } else {
      this.service.getBudgets().subscribe({
        next: (budgets) => {
          const datasets: { label: string; data: EquityCurvePoint[] }[] = [];
          let remaining = budgets.length;
          if (remaining === 0) {
            this.chartData = { labels: [], datasets: [] };
            return;
          }
          budgets.forEach((b, i) => {
            this.service.getBudgetEquityCurve(b.budget).subscribe({
              next: (curve) => {
                datasets.push({ label: b.budget, data: curve });
                remaining--;
                if (remaining === 0) this.setChartMulti(datasets);
              },
            });
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
