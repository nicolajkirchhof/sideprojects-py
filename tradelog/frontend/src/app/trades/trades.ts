import { Component, OnInit, inject } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { CommonModule, DatePipe, DecimalPipe } from '@angular/common';
import { MatTableModule } from '@angular/material/table';
import { ContentArea } from '../shared/content-area/content-area';
import { ReactiveFormsModule, FormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatDatepickerModule } from '@angular/material/datepicker';
import { MatNativeDateModule } from '@angular/material/core';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { TradesService, TradeDto, TradeUpsert } from './trades.service';
import { toIsoOrNull, pnlColor } from '../shared/utils';
import { NotificationService } from '../shared/notification.service';

@Component({
  selector: 'app-trades',
  standalone: true,
  imports: [
    CommonModule,
    MatTableModule,
    ContentArea,
    ReactiveFormsModule,
    FormsModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatIconModule,
    MatDatepickerModule,
    MatNativeDateModule,
    MatProgressBarModule,
    DatePipe,
    DecimalPipe,
  ],
  templateUrl: './trades.html',
  host: { class: 'flex flex-col flex-1' },
})
export class Trades implements OnInit {
  private service = inject(TradesService);
  private route = inject(ActivatedRoute);
  private fb = inject(FormBuilder);
  private notify = inject(NotificationService);

  loading = false;

  trades: TradeDto[] = [];
  displayedColumns = [
    'symbol', 'date', 'posChange', 'price', 'multiplier',
    'totalPos', 'avgPrice', 'pnl', 'commission',
  ];

  // Check if any trade has multiplier != 1 to show/hide the column
  hasMultiplier = false;

  // Sidebar state
  showSidebar = false;
  form!: FormGroup;
  isCreating = false;
  selected: TradeDto | null = null;

  // Filter
  filterSymbol = '';

  ngOnInit(): void {
    const qp = this.route.snapshot.queryParams;
    if (qp['symbol']) this.filterSymbol = qp['symbol'];
    this.load();
    this.form = this.fb.group({
      id: [{ value: null, disabled: true }],
      symbol: ['', [Validators.required]],
      date: [null, [Validators.required]],
      posChange: [null, [Validators.required]],
      price: [null, [Validators.required]],
      commission: [0],
      multiplier: [1],
      bestExitPrice: [null],
      bestExitDate: [null],
    });
  }

  load(): void {
    this.loading = true;
    const filters: any = {};
    if (this.filterSymbol.trim()) filters.symbol = this.filterSymbol.trim().toUpperCase();
    this.service.getAll(filters).subscribe({
      next: (data) => {
        this.trades = data ?? [];
        this.hasMultiplier = this.trades.some(t => t.multiplier !== 1);
        this.displayedColumns = this.hasMultiplier
          ? ['symbol', 'date', 'posChange', 'price', 'multiplier', 'totalPos', 'avgPrice', 'pnl', 'commission']
          : ['symbol', 'date', 'posChange', 'price', 'totalPos', 'avgPrice', 'pnl', 'commission'];
        this.loading = false;
      },
      error: () => {
        this.notify.error('Failed to load trades');
        this.loading = false;
      },
    });
  }

  onFilterChange(): void {
    this.load();
  }

  onRowSelect(row: TradeDto): void {
    this.isCreating = false;
    this.selected = row;
    this.form.reset({
      id: row.id,
      symbol: row.symbol,
      date: row.date ? new Date(row.date) : null,
      posChange: row.posChange,
      price: row.price,
      commission: row.commission,
      multiplier: row.multiplier,
      bestExitPrice: row.bestExitPrice ?? null,
      bestExitDate: row.bestExitDate ? new Date(row.bestExitDate) : null,
    });
    this.showSidebar = true;
  }

  onNew(): void {
    this.isCreating = true;
    this.selected = null;
    this.form.reset({
      id: null,
      symbol: '',
      date: new Date(),
      posChange: null,
      price: null,
      commission: 0,
      multiplier: 1,
      bestExitPrice: null,
      bestExitDate: null,
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
    const payload: TradeUpsert = {
      id: v.id ?? undefined,
      symbol: v.symbol.toUpperCase().trim(),
      date: toIsoOrNull(v.date) as string,
      posChange: Number(v.posChange),
      price: Number(v.price),
      commission: Number(v.commission ?? 0),
      multiplier: Number(v.multiplier ?? 1),
      bestExitPrice: v.bestExitPrice ? Number(v.bestExitPrice) : null,
      bestExitDate: toIsoOrNull(v.bestExitDate),
    };

    const obs = this.isCreating || !payload.id
      ? this.service.create(payload)
      : this.service.update(payload.id as number, payload);

    obs.subscribe({
      next: () => {
        this.notify.success(this.isCreating ? 'Trade created' : 'Trade updated');
        this.load();
        this.onCancel();
      },
      error: () => this.notify.error('Failed to save trade'),
    });
  }

  onDelete(): void {
    if (!this.selected) return;
    if (!confirm(`Delete ${this.selected.symbol} trade from ${new Date(this.selected.date).toLocaleDateString()}?`)) return;
    this.service.delete(this.selected.id).subscribe({
      next: () => {
        this.notify.success('Trade deleted');
        this.load();
        this.onCancel();
      },
      error: () => this.notify.error('Failed to delete trade'),
    });
  }

  pnlColor = pnlColor;
}
