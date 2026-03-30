import { Component, OnInit, inject, ViewChild, ElementRef } from '@angular/core';
import { CommonModule, DatePipe, DecimalPipe } from '@angular/common';
import { MatTableModule } from '@angular/material/table';
import { ContentArea } from '../shared/content-area/content-area';
import { ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatDatepickerModule } from '@angular/material/datepicker';
import { MatNativeDateModule } from '@angular/material/core';
import { BaseChartDirective } from 'ng2-charts';
import { ChartConfiguration } from 'chart.js/auto';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { CapitalService, Capital, CapitalUpsert } from './capital.service';
import { toIsoOrNull, pnlColor } from '../shared/utils';
import { NotificationService } from '../shared/notification.service';

@Component({
  selector: 'app-capital',
  standalone: true,
  imports: [
    CommonModule,
    MatTableModule,
    ContentArea,
    ReactiveFormsModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatDatepickerModule,
    MatNativeDateModule,
    BaseChartDirective,
    MatProgressBarModule,
    DatePipe,
    DecimalPipe,
  ],
  templateUrl: './capital.html',
  host: { class: 'flex flex-col flex-1' },
})
export class CapitalComponent implements OnInit {
  private service = inject(CapitalService);
  private fb = inject(FormBuilder);
  private notify = inject(NotificationService);

  loading = false;

  capitals: Capital[] = [];
  displayedColumns = [
    'date', 'netLiquidity', 'maintenance', 'maintenancePct',
    'excessLiquidity', 'bpr', 'totalPnl', 'netDelta', 'netTheta', 'totalMargin',
  ];

  showSidebar = false;
  form!: FormGroup;
  isCreating = false;
  selected: Capital | null = null;

  pnlColor = pnlColor;

  // Chart
  chartData: ChartConfiguration<'line'>['data'] = { labels: [], datasets: [] };
  chartOptions: ChartConfiguration<'line'>['options'] = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { labels: { color: '#ccc' } },
    },
    scales: {
      x: { ticks: { color: '#999' }, grid: { color: '#333' } },
      y: { position: 'left', ticks: { color: '#ccc' }, grid: { color: '#333' } },
      y1: {
        position: 'right',
        ticks: { color: '#f5a623' },
        grid: { drawOnChartArea: false },
      },
    },
  };

  ngOnInit(): void {
    this.load();
    this.form = this.fb.group({
      id: [{ value: null, disabled: true }],
      date: [null, [Validators.required]],
      netLiquidity: [null, [Validators.required]],
      maintenance: [null, [Validators.required]],
      excessLiquidity: [null, [Validators.required]],
      bpr: [null, [Validators.required]],
    });
  }

  load(): void {
    this.loading = true;
    this.service.getAll().subscribe({
      next: (data) => {
        this.capitals = data ?? [];
        this.updateChart();
        this.loading = false;
      },
      error: () => {
        this.notify.error('Failed to load capital snapshots');
        this.loading = false;
      },
    });
  }

  onRowSelect(row: Capital): void {
    this.isCreating = false;
    this.selected = row;
    this.form.reset({
      id: row.id,
      date: row.date ? new Date(row.date) : null,
      netLiquidity: row.netLiquidity,
      maintenance: row.maintenance,
      excessLiquidity: row.excessLiquidity,
      bpr: row.bpr,
    });
    this.showSidebar = true;
  }

  onNew(): void {
    this.isCreating = true;
    this.selected = null;
    this.form.reset({
      id: null,
      date: new Date(),
      netLiquidity: null,
      maintenance: null,
      excessLiquidity: null,
      bpr: null,
    });
    this.showSidebar = true;
  }

  onCancel(): void {
    this.showSidebar = false;
    this.selected = null;
    this.isCreating = false;
  }

  onSave(): void {
    if (this.form.invalid) {
      this.form.markAllAsTouched();
      return;
    }
    const v = this.form.getRawValue();
    const payload: CapitalUpsert = {
      id: v.id ?? undefined,
      date: toIsoOrNull(v.date) as string,
      netLiquidity: Number(v.netLiquidity),
      maintenance: Number(v.maintenance),
      excessLiquidity: Number(v.excessLiquidity),
      bpr: Number(v.bpr),
    };

    const obs = this.isCreating || !payload.id
      ? this.service.create(payload)
      : this.service.update(payload.id as number, payload);

    obs.subscribe({
      next: () => {
        this.notify.success(this.isCreating ? 'Snapshot created' : 'Snapshot updated');
        this.load();
        this.onCancel();
      },
      error: () => this.notify.error('Failed to save snapshot'),
    });
  }

  onDelete(): void {
    if (!this.selected) return;
    if (!confirm('Delete this capital snapshot?')) return;
    this.service.delete(this.selected.id).subscribe({
      next: () => {
        this.notify.success('Snapshot deleted');
        this.load();
        this.onCancel();
      },
      error: () => this.notify.error('Failed to delete snapshot'),
    });
  }

  private updateChart(): void {
    // Reverse to chronological order (API returns desc)
    const sorted = [...this.capitals].reverse();
    const labels = sorted.map(c => new Date(c.date).toLocaleDateString());

    this.chartData = {
      labels,
      datasets: [
        {
          label: 'Net Liquidity',
          data: sorted.map(c => c.netLiquidity),
          borderColor: '#48dbfb',
          backgroundColor: 'rgba(72, 219, 251, 0.1)',
          fill: true,
          tension: 0.3,
          yAxisID: 'y',
        },
        {
          label: 'Maintenance %',
          data: sorted.map(c => c.maintenancePct),
          borderColor: '#f5a623',
          borderDash: [5, 5],
          fill: false,
          tension: 0.3,
          yAxisID: 'y1',
        },
      ],
    };
  }
}
