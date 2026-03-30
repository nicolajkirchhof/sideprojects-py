import { Component, OnInit, inject } from '@angular/core';
import { CommonModule, DecimalPipe } from '@angular/common';
import { MatTableModule } from '@angular/material/table';
import { MatButtonModule } from '@angular/material/button';
import { MatButtonToggleModule } from '@angular/material/button-toggle';
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
} from '../trade-entries/trade-entries.service';

@Component({
  selector: 'app-instrument-summaries',
  standalone: true,
  imports: [
    CommonModule,
    MatTableModule,
    MatButtonModule,
    MatButtonToggleModule,
    DecimalPipe,
    MatProgressBarModule,
  ],
  templateUrl: './instrument-summaries.html',
  host: { class: 'flex flex-col flex-1 overflow-auto' },
})
export class InstrumentSummaries implements OnInit {
  private service = inject(InstrumentSummariesService);
  private router = inject(Router);
  private notify = inject(NotificationService);

  loading = false;

  optionSummaries: OptionInstrumentSummary[] = [];
  tradeSummaries: TradeInstrumentSummary[] = [];

  optionColumns = [
    'symbol', 'status', 'currentSetup', 'strikes', 'dit', 'dte',
    'pnl', 'unrealizedPnlPct', 'delta', 'theta', 'vega', 'margin', 'roic',
  ];
  tradeColumns = [
    'symbol', 'status', 'positionType', 'totalPos', 'pnl', 'unrealizedPnlPct', 'realizedPnl', 'commissions',
  ];

  statusFilter = 'open';
  typeOfTradeLabel: Record<string, string> = TYPE_OF_TRADE_LABELS;
  pnlColor = pnlColor;

  // Portfolio totals
  totals = {
    pnl: 0, delta: 0, theta: 0, vega: 0, gamma: 0, margin: 0,
  };

  ngOnInit(): void {
    this.load();
  }

  load(): void {
    this.loading = true;
    let remaining = 2;
    const done = () => { if (--remaining === 0) this.loading = false; };

    const status = this.statusFilter === 'all' ? undefined : this.statusFilter;
    this.service.getOptionSummaries(status).subscribe({
      next: (data) => {
        this.optionSummaries = data ?? [];
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
        this.tradeSummaries = this.statusFilter === 'all'
          ? (data ?? [])
          : (data ?? []).filter(t =>
              this.statusFilter === 'open' ? t.totalPos !== 0 : t.totalPos === 0);
        done();
      },
      error: (err) => {
        this.notify.error('Failed to load trade summaries');
        done();
      },
    });
  }

  onStatusFilterChange(value: string): void {
    this.statusFilter = value;
    this.load();
  }

  onOptionRowClick(row: OptionInstrumentSummary): void {
    this.router.navigate(['/option-positions'], { queryParams: { symbol: row.symbol } });
  }

  onTradeRowClick(row: TradeInstrumentSummary): void {
    this.router.navigate(['/trades'], { queryParams: { symbol: row.symbol } });
  }

  private computeTotals(): void {
    this.totals = {
      pnl: this.optionSummaries.reduce((s, r) => s + r.pnl, 0),
      delta: this.optionSummaries.reduce((s, r) => s + r.delta, 0),
      theta: this.optionSummaries.reduce((s, r) => s + r.theta, 0),
      vega: this.optionSummaries.reduce((s, r) => s + r.vega, 0),
      gamma: this.optionSummaries.reduce((s, r) => s + r.gamma, 0),
      margin: this.optionSummaries.reduce((s, r) => s + r.margin, 0),
    };
  }
}
