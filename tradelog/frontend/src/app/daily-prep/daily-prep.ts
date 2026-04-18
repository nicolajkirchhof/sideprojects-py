import { Component, inject, signal } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { MatChipsModule } from '@angular/material/chips';
import { MatDatepickerModule } from '@angular/material/datepicker';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatIconModule } from '@angular/material/icon';
import { MatInputModule } from '@angular/material/input';
import { MatNativeDateModule } from '@angular/material/core';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatTableModule } from '@angular/material/table';
import { MatTooltipModule } from '@angular/material/tooltip';
import {
  DailyPrepService,
  DailyPrepDto,
  MarketSummary,
  WatchlistCandidate,
} from './daily-prep.service';
import { NotificationService } from '../shared/notification.service';
import { pnlColor } from '../shared/utils';

@Component({
  selector: 'app-daily-prep',
  standalone: true,
  imports: [
    DecimalPipe,
    FormsModule,
    MatButtonModule,
    MatCardModule,
    MatChipsModule,
    MatDatepickerModule,
    MatFormFieldModule,
    MatIconModule,
    MatInputModule,
    MatNativeDateModule,
    MatProgressBarModule,
    MatTableModule,
    MatTooltipModule,
  ],
  templateUrl: './daily-prep.html',
  host: { class: 'flex flex-col flex-1 overflow-auto' },
})
export class DailyPrepComponent {
  private service = inject(DailyPrepService);
  private notify = inject(NotificationService);

  loading = signal(false);
  report = signal<DailyPrepDto | null>(null);
  marketSummary = signal<MarketSummary | null>(null);
  watchlist = signal<WatchlistCandidate[]>([]);
  selectedCandidate = signal<WatchlistCandidate | null>(null);
  selectedDate = signal<Date>(new Date());

  watchlistColumns = ['symbol', 'score', 'price', 'change5d', 'change1m', 'setupType', 'confidence'];

  constructor() {
    this.loadLatest();
  }

  loadLatest(): void {
    this.loading.set(true);
    this.service.getLatest().subscribe({
      next: (data) => this.applyReport(data),
      error: () => {
        this.loading.set(false);
        this.report.set(null);
      },
    });
  }

  loadByDate(): void {
    const d = this.selectedDate();
    if (!d) return;
    this.loading.set(true);
    const dateStr = d.toISOString().split('T')[0];
    this.service.getByDate(dateStr).subscribe({
      next: (data) => this.applyReport(data),
      error: () => {
        this.loading.set(false);
        this.report.set(null);
        this.marketSummary.set(null);
        this.watchlist.set([]);
        this.notify.error('No report for this date');
      },
    });
  }

  onRowClick(row: WatchlistCandidate): void {
    this.selectedCandidate.set(
      this.selectedCandidate()?.symbol === row.symbol ? null : row,
    );
  }

  regimeClass(regime: string): string {
    switch (regime) {
      case 'GO': return 'text-green-400';
      case 'CAUTION': return 'text-yellow-400';
      case 'NO-GO': return 'text-red-400';
      default: return '';
    }
  }

  confidenceClass(confidence: string | null): string {
    switch (confidence) {
      case 'high': return 'text-green-400';
      case 'medium': return 'text-yellow-400';
      case 'low': return 'text-red-400';
      default: return 'opacity-50';
    }
  }

  private applyReport(data: DailyPrepDto): void {
    this.report.set(data);
    this.selectedDate.set(new Date(data.date));

    try {
      this.marketSummary.set(data.marketSummary ? JSON.parse(data.marketSummary) : null);
    } catch {
      this.marketSummary.set(null);
    }

    try {
      this.watchlist.set(data.watchlist ? JSON.parse(data.watchlist) : []);
    } catch {
      this.watchlist.set([]);
    }

    this.selectedCandidate.set(null);
    this.loading.set(false);
  }
}
