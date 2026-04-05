import { Component, effect, inject, signal, viewChild } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { MatTableModule, MatTableDataSource } from '@angular/material/table';
import { MatSort, MatSortModule } from '@angular/material/sort';
import { FormsModule } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatButtonToggleModule } from '@angular/material/button-toggle';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { Router } from '@angular/router';
import {
  InstrumentSummariesService,
  OptionInstrumentSummary,
  TradeInstrumentSummary,
} from './instrument-summaries.service';
import { pnlColor } from '../shared/utils';
import { NotificationService } from '../shared/notification.service';
import {
  TYPE_OF_TRADE_LABELS,
} from '../trades/trades.service';

@Component({
  selector: 'app-instrument-summaries',
  standalone: true,
  imports: [
    FormsModule,
    MatTableModule,
    MatButtonModule,
    MatButtonToggleModule,
    MatFormFieldModule,
    MatInputModule,
    DecimalPipe,
    MatProgressBarModule,
    MatSortModule,
  ],
  templateUrl: './instrument-summaries.html',
  host: { class: 'flex flex-col flex-1 overflow-auto' },
})
export class InstrumentSummaries {
  private service = inject(InstrumentSummariesService);
  private router = inject(Router);
  private notify = inject(NotificationService);

  optionSort = viewChild<MatSort>('optionSort');
  tradeSort = viewChild<MatSort>('tradeSort');

  loading = signal(false);

  optionDataSource = new MatTableDataSource<OptionInstrumentSummary>([]);
  tradeDataSource = new MatTableDataSource<TradeInstrumentSummary>([]);

  optionColumns = [
    'symbol', 'status', 'currentSetup', 'strikes', 'dit', 'dte',
    'pnl', 'unrealizedPnlPct', 'delta', 'theta', 'vega', 'margin', 'roic',
  ];
  tradeColumns = [
    'symbol', 'status', 'positionType', 'totalPos', 'pnl', 'unrealizedPnlPct', 'realizedPnl', 'commissions',
  ];

  statusFilter = signal('open');
  filterSymbol = signal('');
  typeOfTradeLabel: Record<string, string> = TYPE_OF_TRADE_LABELS;
  pnlColor = pnlColor;

  // Portfolio totals
  totals = signal({
    pnl: 0, delta: 0, theta: 0, vega: 0, gamma: 0, margin: 0,
  });

  constructor() {
    this.load();

    effect(() => {
      const optSort = this.optionSort();
      const trSort = this.tradeSort();
      const symbolFilter = (row: { symbol: string }, filter: string) =>
        !filter || row.symbol.toLowerCase().includes(filter);
      if (optSort) {
        this.optionDataSource.sort = optSort;
        this.optionDataSource.filterPredicate = symbolFilter;
      }
      if (trSort) {
        this.tradeDataSource.sort = trSort;
        this.tradeDataSource.filterPredicate = symbolFilter;
      }
    });
  }

  applyFilter(): void {
    const f = this.filterSymbol().trim().toLowerCase();
    this.optionDataSource.filter = f;
    this.tradeDataSource.filter = f;
  }

  load(): void {
    this.loading.set(true);
    let remaining = 2;
    const done = () => { if (--remaining === 0) this.loading.set(false); };

    const status = this.statusFilter() === 'all' ? undefined : this.statusFilter();
    this.service.getOptionSummaries(status).subscribe({
      next: (data) => {
        this.optionDataSource.data = data ?? [];
        this.computeTotals();
        done();
      },
      error: (err) => {
        this.notify.error('Failed to load option summaries');
        done();
      },
    });
    this.service.getTradeSummaries().subscribe({
      next: (data) => {
        this.tradeDataSource.data = this.statusFilter() === 'all'
          ? (data ?? [])
          : (data ?? []).filter(t =>
              this.statusFilter() === 'open' ? t.totalPos !== 0 : t.totalPos === 0);
        done();
      },
      error: (err) => {
        this.notify.error('Failed to load trade summaries');
        done();
      },
    });
  }

  onStatusFilterChange(value: string): void {
    this.statusFilter.set(value);
    this.load();
  }

  onOptionRowClick(row: OptionInstrumentSummary): void {
    this.router.navigate(['/option-positions'], { queryParams: { symbol: row.symbol } });
  }

  onTradeRowClick(row: TradeInstrumentSummary): void {
    this.router.navigate(['/stock-positions'], { queryParams: { symbol: row.symbol } });
  }

  private computeTotals(): void {
    const options = this.optionDataSource.data;
    this.totals.set({
      pnl: options.reduce((s, r) => s + r.pnl, 0),
      delta: options.reduce((s, r) => s + r.delta, 0),
      theta: options.reduce((s, r) => s + r.theta, 0),
      vega: options.reduce((s, r) => s + r.vega, 0),
      gamma: options.reduce((s, r) => s + r.gamma, 0),
      margin: options.reduce((s, r) => s + r.margin, 0),
    });
  }
}
