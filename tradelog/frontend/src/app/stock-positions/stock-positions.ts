import { Component, computed, effect, inject, signal, viewChild } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { DecimalPipe } from '@angular/common';
import { AppDatePipe } from '../shared/app-date.pipe';
import { MatTableModule, MatTableDataSource } from '@angular/material/table';
import { MatSort, MatSortModule } from '@angular/material/sort';
import { MatPaginator, MatPaginatorModule } from '@angular/material/paginator';
import { ContentArea } from '../shared/content-area/content-area';
import { ReactiveFormsModule, FormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatDatepickerModule } from '@angular/material/datepicker';
import { MatNativeDateModule } from '@angular/material/core';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { StockPositionsService, StockPositionDto, StockPositionUpsert } from './stock-positions.service';
import { toIsoOrNull, pnlColor } from '../shared/utils';
import { NotificationService } from '../shared/notification.service';

@Component({
  selector: 'app-stock-positions',
  standalone: true,
  imports: [
    MatTableModule,
    ContentArea,
    ReactiveFormsModule,
    FormsModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatIconModule,
    MatCheckboxModule,
    MatTooltipModule,
    MatDatepickerModule,
    MatNativeDateModule,
    MatProgressBarModule,
    MatSortModule,
    MatPaginatorModule,
    AppDatePipe,
    DecimalPipe,
  ],
  templateUrl: './stock-positions.html',
  host: { class: 'flex flex-col flex-1' },
})
export class StockPositions {
  private service = inject(StockPositionsService);
  private route = inject(ActivatedRoute);
  private router = inject(Router);
  private fb = inject(FormBuilder);
  private notify = inject(NotificationService);

  sort = viewChild(MatSort);
  paginator = viewChild(MatPaginator);

  loading = signal(false);

  dataSource = new MatTableDataSource<StockPositionDto>([]);
  displayedColumns = signal([
    'symbol', 'date', 'posChange', 'price', 'multiplier',
    'totalPos', 'avgPrice', 'pnl', 'commission',
  ]);

  // Check if any trade has multiplier != 1 to show/hide the column
  hasMultiplier = signal(false);

  // Multi-select for "Create Trade"
  selectedIds = signal<Set<number>>(new Set());
  hasUnlinkedSelection = computed(() => {
    const ids = this.selectedIds();
    if (ids.size === 0) return false;
    return this.dataSource.data.some(p => ids.has(p.id) && !p.tradeId);
  });

  toggleSelect(id: number): void {
    this.selectedIds.update(s => {
      const next = new Set(s);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }

  onCreateTradeFromSelected(): void {
    const ids = [...this.selectedIds()].filter(id =>
      this.dataSource.data.some(p => p.id === id && !p.tradeId)
    );
    if (ids.length === 0) return;
    this.router.navigate(['/trades'], { queryParams: { createFrom: 'stk:' + ids.join(',') } });
  }

  // Sidebar state
  showSidebar = signal(false);
  form!: FormGroup;
  isCreating = signal(false);
  editMode = signal(false);
  selected = signal<StockPositionDto | null>(null);

  // Filter
  filterSymbol = signal('');

  constructor() {
    const qp = this.route.snapshot.queryParams;
    if (qp['symbol']) this.filterSymbol.set(qp['symbol']);
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
      notes: [''],
    });
    this.load();

    effect(() => {
      const sort = this.sort();
      const paginator = this.paginator();
      if (sort) {
        this.dataSource.sort = sort;
        this.dataSource.filterPredicate = (row: StockPositionDto, filter: string) => {
          if (!filter) return true;
          return row.symbol.toLowerCase().includes(filter);
        };
      }
      if (paginator) this.dataSource.paginator = paginator;
    });
  }

  applyFilter(): void {
    this.dataSource.filter = this.filterSymbol().trim().toLowerCase();
  }

  load(): void {
    this.loading.set(true);
    this.service.getAll().subscribe({
      next: (data) => {
        this.dataSource.data = data ?? [];
        this.applyFilter();
        this.hasMultiplier.set(this.dataSource.data.some(t => t.multiplier !== 1));
        this.displayedColumns.set(this.hasMultiplier()
          ? ['select', 'symbol', 'date', 'posChange', 'price', 'multiplier', 'totalPos', 'avgPrice', 'pnl', 'commission']
          : ['select', 'symbol', 'date', 'posChange', 'price', 'totalPos', 'avgPrice', 'pnl', 'commission']);
        this.loading.set(false);
        if (!this.selected() && (data?.length ?? 0) > 0) {
          this.onRowSelect(data![0]);
        }
      },
      error: () => {
        this.notify.error('Failed to load trades');
        this.loading.set(false);
      },
    });
  }

  onRowSelect(row: StockPositionDto): void {
    this.isCreating.set(false);
    this.editMode.set(false);
    this.selected.set(row);
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
      notes: row.notes ?? '',
    });
    this.form.disable({ emitEvent: false });
    this.showSidebar.set(true);
  }

  onEdit(): void {
    this.editMode.set(true);
    this.form.enable({ emitEvent: false });
    this.form.get('id')?.disable({ emitEvent: false });
  }

  onCancelEdit(): void {
    if (this.isCreating()) {
      this.onCancel();
      return;
    }
    const row = this.selected();
    if (row) this.onRowSelect(row);
  }

  copyTickers(): void {
    const symbols = [...new Set(this.dataSource.filteredData.map(r => r.symbol))];
    if (symbols.length === 0) {
      this.notify.error('No tickers to copy');
      return;
    }
    navigator.clipboard.writeText(symbols.join(',')).then(
      () => this.notify.success(`Copied ${symbols.length} tickers to clipboard`),
      () => this.notify.error('Failed to access clipboard'),
    );
  }

  onNew(): void {
    this.isCreating.set(true);
    this.editMode.set(true);
    this.selected.set(null);
    this.form.enable({ emitEvent: false });
    this.form.get('id')?.disable({ emitEvent: false });
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
      notes: '',
    });
    this.showSidebar.set(true);
  }

  onCancel(): void {
    this.showSidebar.set(false);
    this.selected.set(null);
    this.isCreating.set(false);
    this.editMode.set(false);
  }

  onSave(): void {
    if (this.form.invalid) {
      this.form.markAllAsTouched();
      return;
    }
    const v = this.form.getRawValue();
    const payload: StockPositionUpsert = {
      id: v.id ?? undefined,
      symbol: v.symbol.toUpperCase().trim(),
      date: toIsoOrNull(v.date) as string,
      posChange: Number(v.posChange),
      price: Number(v.price),
      commission: Number(v.commission ?? 0),
      multiplier: Number(v.multiplier ?? 1),
      bestExitPrice: v.bestExitPrice ? Number(v.bestExitPrice) : null,
      bestExitDate: toIsoOrNull(v.bestExitDate),
      notes: v.notes || null,
    };

    const obs = this.isCreating() || !payload.id
      ? this.service.create(payload)
      : this.service.update(payload.id as number, payload);

    obs.subscribe({
      next: () => {
        this.notify.success(this.isCreating() ? 'Trade created' : 'Trade updated');
        this.load();
        this.onCancel();
      },
      error: () => this.notify.error('Failed to save trade'),
    });
  }

  onDelete(): void {
    const sel = this.selected();
    if (!sel) return;
    if (!confirm(`Delete ${sel.symbol} trade from ${new Date(sel.date).toLocaleDateString()}?`)) return;
    this.service.delete(sel.id).subscribe({
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
