import { Component, OnInit, inject } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { CommonModule, DatePipe, DecimalPipe } from '@angular/common';
import { MatTableModule } from '@angular/material/table';
import { ContentArea } from '../shared/content-area/content-area';
import { ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
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
    CommonModule,
    MatTableModule,
    ContentArea,
    ReactiveFormsModule,
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
    DatePipe,
    DecimalPipe,
  ],
  templateUrl: './option-positions.html',
  host: { class: 'flex flex-col flex-1' },
})
export class OptionPositions implements OnInit {
  private service = inject(OptionPositionsService);
  private route = inject(ActivatedRoute);
  private router = inject(Router);
  private fb = inject(FormBuilder);
  private notify = inject(NotificationService);

  loading = false;

  positions: OptionPositionDto[] = [];
  displayedColumns = [
    'symbol', 'contractId', 'right', 'strike', 'expiry', 'pos', 'cost',
    'lastPrice', 'unrealizedPnl', 'unrealizedPnlPct', 'realizedPnl',
    'delta', 'theta', 'durationPct', 'roic', 'actions',
  ];

  // Sidebar state
  showSidebar = false;
  form!: FormGroup;
  isCreating = false;
  selected: OptionPositionDto | null = null;

  // Filter
  statusFilter = 'open';
  filterSymbol = '';
  rights = Object.values(PositionRight);
  rightLabel = POSITION_RIGHT_LABELS;

  ngOnInit(): void {
    const qp = this.route.snapshot.queryParams;
    if (qp['symbol']) this.filterSymbol = qp['symbol'];
    this.load();
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
  }

  load(): void {
    this.loading = true;
    const filters: any = { status: this.statusFilter === 'all' ? undefined : this.statusFilter };
    if (this.filterSymbol.trim()) filters.symbol = this.filterSymbol.trim().toUpperCase();
    this.service.getAll(filters).subscribe({
      next: (data) => {
        this.positions = data ?? [];
        this.loading = false;
      },
      error: () => {
        this.notify.error('Failed to load option positions');
        this.loading = false;
      },
    });
  }

  onStatusFilterChange(value: string): void {
    this.statusFilter = value;
    this.load();
  }

  onRowSelect(row: OptionPositionDto): void {
    this.isCreating = false;
    this.selected = row;
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
    this.showSidebar = true;
  }

  onNew(): void {
    this.isCreating = true;
    this.selected = null;
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
    this.showSidebar = true;
  }

  onCancel(): void {
    this.showSidebar = false;
    this.selected = null;
    this.isCreating = false;
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

    const obs = this.isCreating || !payload.id
      ? this.service.create(payload)
      : this.service.update(payload.id as number, payload);

    obs.subscribe({
      next: () => {
        this.notify.success(this.isCreating ? 'Position created' : 'Position updated');
        this.load();
        this.onCancel();
      },
      error: () => this.notify.error('Failed to save position'),
    });
  }

  onDelete(): void {
    if (!this.selected) return;
    if (!confirm(`Delete ${this.selected.symbol} ${this.selected.right} ${this.selected.strike} position?`)) return;
    this.service.delete(this.selected.id).subscribe({
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
