import { Component, signal, viewChild, effect, inject } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { DecimalPipe } from '@angular/common';
import { AppDatePipe } from '../shared/app-date.pipe';
import { MatTableModule, MatTableDataSource } from '@angular/material/table';
import { MatSortModule, MatSort } from '@angular/material/sort';
import { MatPaginatorModule, MatPaginator } from '@angular/material/paginator';
import { ContentArea } from '../shared/content-area/content-area';
import { ReactiveFormsModule, FormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatButtonToggleModule } from '@angular/material/button-toggle';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatDatepickerModule } from '@angular/material/datepicker';
import { MatNativeDateModule } from '@angular/material/core';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import {
  OptionPositionsService, OptionPositionDto, OptionPositionUpsert,
  PositionRight, POSITION_RIGHT_LABELS,
} from './option-positions.service';
import { toIsoOrNull, pnlColor } from '../shared/utils';
import { NotificationService } from '../shared/notification.service';

@Component({
  selector: 'app-option-positions',
  standalone: true,
  imports: [
    MatTableModule,
    MatSortModule,
    MatPaginatorModule,
    ContentArea,
    ReactiveFormsModule,
    FormsModule,
    MatFormFieldModule,
    MatInputModule,
    MatSelectModule,
    MatButtonModule,
    MatIconModule,
    MatButtonToggleModule,
    MatTooltipModule,
    MatDatepickerModule,
    MatNativeDateModule,
    MatProgressBarModule,
    AppDatePipe,
    DecimalPipe,
  ],
  templateUrl: './option-positions.html',
  host: { class: 'flex flex-col flex-1' },
})
export class OptionPositions {
  private service = inject(OptionPositionsService);
  private route = inject(ActivatedRoute);
  private router = inject(Router);
  private fb = inject(FormBuilder);
  private notify = inject(NotificationService);

  sort = viewChild(MatSort);
  paginator = viewChild(MatPaginator);

  loading = signal(false);

  dataSource = new MatTableDataSource<OptionPositionDto>([]);
  displayedColumns = [
    'symbol', 'contractId', 'right', 'strike', 'opened', 'expiry', 'closed',
    'pos', 'cost', 'lastPrice', 'unrealizedPnl', 'unrealizedPnlPct',
    'realizedPnl', 'delta', 'theta', 'logCount', 'durationPct', 'roic', 'actions',
  ];

  // Sidebar state
  showSidebar = signal(false);
  form!: FormGroup;
  isCreating = signal(false);
  editMode = signal(false);
  selected = signal<OptionPositionDto | null>(null);

  // Filters
  statusFilter = signal('open');
  filterSymbol = signal('');
  filterRight = signal('');
  filterExpiry = signal('');
  rights = Object.values(PositionRight);
  rightLabel = POSITION_RIGHT_LABELS;

  constructor() {
    const qp = this.route.snapshot.queryParams;
    if (qp['symbol']) this.filterSymbol.set(qp['symbol']);
    this.form = this.fb.group({
      id: [{ value: null, disabled: true }],
      symbol: ['', [Validators.required]],
      contractId: ['', [Validators.required]],
      opened: [null, [Validators.required]],
      expiry: [null, [Validators.required]],
      closed: [null],
      pos: [null, [Validators.required]],
      right: [null, [Validators.required]],
      strike: [null, [Validators.required]],
      cost: [null, [Validators.required]],
      closePrice: [null],
      commission: [0],
      multiplier: [100],
      closeReasons: [null],
      bestExitPrice: [null],
      bestExitDate: [null],
    });
    this.load();

    effect(() => {
      const sort = this.sort();
      const paginator = this.paginator();
      if (sort) {
        this.dataSource.sort = sort;
        this.dataSource.filterPredicate = (row: OptionPositionDto, filter: string) => {
          const f = JSON.parse(filter);
          if (f.symbol && !row.symbol.toLowerCase().includes(f.symbol)) return false;
          if (f.right && row.right !== f.right) return false;
          if (f.expiry && row.expiry?.slice(0, 7) !== f.expiry) return false;
          return true;
        };
      }
      if (paginator) this.dataSource.paginator = paginator;
    });
  }

  load(): void {
    this.loading.set(true);
    const filters: any = { status: this.statusFilter() === 'all' ? undefined : this.statusFilter() };
    if (this.filterSymbol().trim()) filters.symbol = this.filterSymbol().trim().toUpperCase();
    this.service.getAll(filters).subscribe({
      next: (data) => {
        this.dataSource.data = data ?? [];
        this.loading.set(false);
        // Auto-select first row if nothing currently selected
        if (!this.selected() && (data?.length ?? 0) > 0) {
          this.onRowSelect(data![0]);
        }
      },
      error: () => {
        this.notify.error('Failed to load option positions');
        this.loading.set(false);
      },
    });
  }

  applyFilter(): void {
    this.dataSource.filter = JSON.stringify({
      symbol: this.filterSymbol().trim().toLowerCase(),
      right: this.filterRight(),
      expiry: this.filterExpiry(),
    });
  }

  onStatusFilterChange(value: string): void {
    this.statusFilter.set(value);
    this.load();
  }

  onRowSelect(row: OptionPositionDto): void {
    this.isCreating.set(false);
    this.editMode.set(false);
    this.selected.set(row);
    this.form.reset({
      id: row.id,
      symbol: row.symbol,
      contractId: row.contractId,
      opened: row.opened ? new Date(row.opened) : null,
      expiry: row.expiry ? new Date(row.expiry) : null,
      closed: row.closed ? new Date(row.closed) : null,
      pos: row.pos,
      right: row.right,
      strike: row.strike,
      cost: row.cost,
      closePrice: row.closePrice ?? null,
      commission: row.commission,
      multiplier: row.multiplier,
      closeReasons: row.closeReasons ?? null,
      bestExitPrice: row.bestExitPrice ?? null,
      bestExitDate: row.bestExitDate ? new Date(row.bestExitDate) : null,
    });
    this.form.disable({ emitEvent: false });
    this.showSidebar.set(true);
  }

  onEdit(): void {
    this.editMode.set(true);
    this.form.enable({ emitEvent: false });
    // ID always read-only
    this.form.get('id')?.disable({ emitEvent: false });
  }

  onCancelEdit(): void {
    const row = this.selected();
    if (this.isCreating()) {
      this.onCancel();
      return;
    }
    if (row) {
      this.onRowSelect(row);
    }
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
      contractId: '',
      opened: new Date(),
      expiry: null,
      closed: null,
      pos: 1,
      right: null,
      strike: null,
      cost: null,
      closePrice: null,
      commission: 0,
      multiplier: 100,
      closeReasons: null,
      bestExitPrice: null,
      bestExitDate: null,
    });
    this.showSidebar.set(true);
  }

  onCancel(): void {
    this.showSidebar.set(false);
    this.selected.set(null);
    this.isCreating.set(false);
    this.editMode.set(false);
  }

  openGreeksHistory(contractId: string, event: Event): void {
    event.stopPropagation();
    this.router.navigate(['/greeks-history'], { queryParams: { contractId } });
  }

  onSave(): void {
    if (this.form.invalid) {
      this.form.markAllAsTouched();
      return;
    }
    const v = this.form.getRawValue();
    const payload: OptionPositionUpsert = {
      id: v.id ?? undefined,
      symbol: v.symbol.toUpperCase().trim(),
      contractId: v.contractId.trim(),
      opened: toIsoOrNull(v.opened) as string,
      expiry: toIsoOrNull(v.expiry) as string,
      closed: toIsoOrNull(v.closed),
      pos: Number(v.pos),
      right: v.right,
      strike: Number(v.strike),
      cost: Number(v.cost),
      closePrice: v.closePrice ? Number(v.closePrice) : null,
      commission: Number(v.commission ?? 0),
      multiplier: Number(v.multiplier ?? 100),
      closeReasons: v.closeReasons ? Number(v.closeReasons) : null,
      bestExitPrice: v.bestExitPrice ? Number(v.bestExitPrice) : null,
      bestExitDate: toIsoOrNull(v.bestExitDate),
    };

    const obs = this.isCreating() || !payload.id
      ? this.service.create(payload)
      : this.service.update(payload.id as number, payload);

    obs.subscribe({
      next: (saved: any) => {
        this.notify.success(this.isCreating() ? 'Position created' : 'Position updated');
        const wasCreating = this.isCreating();
        this.isCreating.set(false);
        this.editMode.set(false);
        this.load();
        // Re-select the saved row to refresh view-mode display
        if (!wasCreating && saved) {
          setTimeout(() => {
            const row = this.dataSource.data.find(r => r.id === saved.id);
            if (row) this.onRowSelect(row);
          });
        }
      },
      error: () => this.notify.error('Failed to save position'),
    });
  }

  onDelete(): void {
    const sel = this.selected();
    if (!sel) return;
    if (!confirm(`Delete ${sel.symbol} ${sel.right} ${sel.strike} position?`)) return;
    this.service.delete(sel.id).subscribe({
      next: () => {
        this.notify.success('Position deleted');
        this.load();
        this.onCancel();
      },
      error: () => this.notify.error('Failed to delete position'),
    });
  }

  pnlColor = pnlColor;
}
