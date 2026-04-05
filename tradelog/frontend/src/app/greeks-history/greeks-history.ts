import { Component, inject, signal } from '@angular/core';

import { ActivatedRoute, Router } from '@angular/router';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { BaseChartDirective } from 'ng2-charts';
import { ChartConfiguration } from 'chart.js/auto';
import { OptionPositionsLogService, OptionPositionsLog } from '../option-positions/option-positions.service';

@Component({
  selector: 'app-greeks-history',
  standalone: true,
  imports: [MatCardModule, MatButtonModule, BaseChartDirective],
  templateUrl: './greeks-history.html',
  host: { class: 'flex flex-col flex-1 overflow-auto' },
})
export class GreeksHistoryComponent {
  private route = inject(ActivatedRoute);
  private router = inject(Router);
  private logService = inject(OptionPositionsLogService);

  contractId = signal('');
  logs = signal<OptionPositionsLog[]>([]);

  deltaChart = signal<ChartConfiguration<'line'>['data']>({ labels: [], datasets: [] });
  thetaChart = signal<ChartConfiguration<'line'>['data']>({ labels: [], datasets: [] });
  ivChart = signal<ChartConfiguration<'line'>['data']>({ labels: [], datasets: [] });
  priceChart = signal<ChartConfiguration<'line'>['data']>({ labels: [], datasets: [] });

  chartOptions: ChartConfiguration<'line'>['options'] = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: { legend: { labels: { color: '#ccc' } } },
    scales: {
      x: { ticks: { color: '#999', maxTicksLimit: 15 }, grid: { color: '#333' } },
      y: { ticks: { color: '#ccc' }, grid: { color: '#333' } },
    },
  };

  dualAxisOptions: ChartConfiguration<'line'>['options'] = {
    ...this.chartOptions,
    scales: {
      x: { ticks: { color: '#999', maxTicksLimit: 15 }, grid: { color: '#333' } },
      y: { position: 'left', ticks: { color: '#48dbfb' }, grid: { color: '#333' } },
      y1: { position: 'right', ticks: { color: '#feca57' }, grid: { drawOnChartArea: false } },
    },
  };

  constructor() {
    const id = this.route.snapshot.queryParamMap.get('contractId') ?? '';
    this.contractId.set(id);
    if (id) {
      this.logService.getByContract(id).subscribe({
        next: (data) => {
          this.logs.set(data);
          this.buildCharts();
        },
      });
    }
  }

  private buildCharts(): void {
    const logs = this.logs();
    const labels = logs.map(l => new Date(l.dateTime).toLocaleDateString());

    this.deltaChart.set({
      labels,
      datasets: [
        { label: 'Delta', data: logs.map(l => l.delta), borderColor: '#48dbfb', tension: 0.3, fill: false },
        { label: 'Gamma', data: logs.map(l => l.gamma), borderColor: '#feca57', tension: 0.3, fill: false, yAxisID: 'y1' as any },
      ],
    });

    this.thetaChart.set({
      labels,
      datasets: [
        { label: 'Theta', data: logs.map(l => l.theta), borderColor: '#ff6b6b', tension: 0.3, fill: false },
        { label: 'Vega', data: logs.map(l => l.vega), borderColor: '#54a0ff', tension: 0.3, fill: false, yAxisID: 'y1' as any },
      ],
    });

    this.ivChart.set({
      labels,
      datasets: [
        { label: 'IV', data: logs.map(l => l.iv * 100), borderColor: '#f368e0', tension: 0.3, fill: false },
      ],
    });

    this.priceChart.set({
      labels,
      datasets: [
        { label: 'Price', data: logs.map(l => l.price), borderColor: '#48dbfb', tension: 0.3, fill: false },
        { label: 'Time Value', data: logs.map(l => l.timeValue), borderColor: '#feca57', tension: 0.3, fill: false, yAxisID: 'y1' as any },
      ],
    });
  }

  goBack(): void {
    this.router.navigate(['/option-positions']);
  }
}
