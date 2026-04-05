import { Component, inject, signal } from '@angular/core';
import { DatePipe, DecimalPipe } from '@angular/common';
import { MatTableModule } from '@angular/material/table';
import { ContentArea } from '../shared/content-area/content-area';
import { ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { MatDatepickerModule } from '@angular/material/datepicker';
import { MatNativeDateModule } from '@angular/material/core';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { QuillModule } from 'ngx-quill';
import {
  TradesService, Trade, TradeUpsert, OptionLegDto, StockLegDto,
  Budget, Strategy, TypeOfTrade, DirectionalBias, Timeframe, ManagementRating,
  STRATEGY_LABELS, TYPE_OF_TRADE_LABELS, TIMEFRAME_LABELS, MANAGEMENT_RATING_LABELS,
} from './trades.service';
import { forkJoin, Observable } from 'rxjs';
import { toIsoOrNull } from '../shared/utils';
import { NotificationService } from '../shared/notification.service';

@Component({
  selector: 'app-trades',
  standalone: true,
  imports: [
    MatTableModule,
    ContentArea,
    ReactiveFormsModule,
    MatFormFieldModule,
    MatInputModule,
    MatSelectModule,
    MatButtonModule,
    MatIconModule,
    MatCheckboxModule,
    MatDatepickerModule,
    MatNativeDateModule,
    MatProgressBarModule,
    QuillModule,
    DatePipe,
    DecimalPipe,
  ],
  templateUrl: './trades.html',
  host: { class: 'flex flex-col flex-1' },
})
export class Trades {
  private service = inject(TradesService);
  private fb = inject(FormBuilder);
  private notify = inject(NotificationService);

  loading = signal(false);

  trades = signal<Trade[]>([]);
  displayedColumns = ['symbol', 'date', 'typeOfTrade', 'directional', 'budget', 'strategy', 'managementRating'];

  // Sidebar state
  showSidebar = signal(false);
  form!: FormGroup;
  isCreating = signal(false);
  selected = signal<Trade | null>(null);

  // Filter state
  filterBudget = signal<Budget | null>(null);
  filterStrategy = signal<Strategy | null>(null);

  // Enum values for dropdowns
  budgets = Object.values(Budget);
  strategies = Object.values(Strategy);
  typesOfTrade = Object.values(TypeOfTrade);
  directionalOptions = Object.values(DirectionalBias);
  timeframes = Object.values(Timeframe);
  managementRatings = Object.values(ManagementRating);

  // Labels (cast to index-friendly type for template access)
  strategyLabel: Record<string, string> = STRATEGY_LABELS;
  typeOfTradeLabel: Record<string, string> = TYPE_OF_TRADE_LABELS;
  timeframeLabel: Record<string, string> = TIMEFRAME_LABELS;
  managementRatingLabel: Record<string, string> = MANAGEMENT_RATING_LABELS;

  // Legs (assigned positions)
  optionLegs = signal<OptionLegDto[]>([]);
  stockLegs = signal<StockLegDto[]>([]);

  // Leg picker
  showLegPicker = signal(false);
  unassignedOptions = signal<OptionLegDto[]>([]);
  unassignedStocks = signal<StockLegDto[]>([]);
  selectedLegIds = signal<{ options: Set<number>; stocks: Set<number> }>({ options: new Set(), stocks: new Set() });

  // Quill config
  showToolbar = signal(false);
  quillModules: any = {
    toolbar: [
      ['bold', 'italic', 'underline'],
      [{ header: [1, 2, 3, false] }],
      [{ list: 'ordered' }, { list: 'bullet' }],
      ['link', 'blockquote'],
      ['clean'],
    ],
  };
  quillModulesOff: any = { toolbar: false };

  constructor() {
    this.load();
    this.form = this.fb.group({
      id: [{ value: null, disabled: true }],
      symbol: ['', [Validators.required]],
      date: [null, [Validators.required]],
      typeOfTrade: [null, [Validators.required]],
      notes: [''],
      directional: [null],
      timeframe: [null],
      budget: [null, [Validators.required]],
      strategy: [null, [Validators.required]],
      newsCatalyst: [false],
      recentEarnings: [false],
      sectorSupport: [false],
      ath: [false],
      rvol: [null],
      institutionalSupport: [''],
      gapPct: [null],
      xAtrMove: [null],
      taFaNotes: [''],
      intendedManagement: [''],
      actualManagement: [''],
      managementRating: [null],
      learnings: [''],
    });
  }

  load(): void {
    this.loading.set(true);
    const filters: any = {};
    if (this.filterBudget()) filters.budget = this.filterBudget();
    if (this.filterStrategy()) filters.strategy = this.filterStrategy();
    this.service.getAll(filters).subscribe({
      next: (data) => {
        this.trades.set(data ?? []);
        this.loading.set(false);
      },
      error: () => {
        this.notify.error('Failed to load trades');
        this.loading.set(false);
      },
    });
  }

  onFilterChange(): void {
    this.load();
  }

  onRowSelect(row: Trade): void {
    this.isCreating.set(false);
    this.selected.set(row);
    this.form.reset({
      id: row.id,
      symbol: row.symbol,
      date: row.date ? new Date(row.date) : null,
      typeOfTrade: row.typeOfTrade,
      notes: row.notes ?? '',
      directional: row.directional ?? null,
      timeframe: row.timeframe ?? null,
      budget: row.budget,
      strategy: row.strategy,
      newsCatalyst: row.newsCatalyst,
      recentEarnings: row.recentEarnings,
      sectorSupport: row.sectorSupport,
      ath: row.ath,
      rvol: row.rvol ?? null,
      institutionalSupport: row.institutionalSupport ?? '',
      gapPct: row.gapPct ?? null,
      xAtrMove: row.xAtrMove ?? null,
      taFaNotes: row.taFaNotes ?? '',
      intendedManagement: row.intendedManagement ?? '',
      actualManagement: row.actualManagement ?? '',
      managementRating: row.managementRating ?? null,
      learnings: row.learnings ?? '',
    });
    this.showSidebar.set(true);
    this.showLegPicker.set(false);
    this.loadLegs(row.id);
  }

  onNew(): void {
    this.isCreating.set(true);
    this.selected.set(null);
    this.form.reset({
      id: null,
      symbol: '',
      date: new Date(),
      typeOfTrade: null,
      notes: '',
      directional: null,
      timeframe: Timeframe.OneDay,
      budget: null,
      strategy: null,
      newsCatalyst: false,
      recentEarnings: false,
      sectorSupport: false,
      ath: false,
      rvol: null,
      institutionalSupport: '',
      gapPct: null,
      xAtrMove: null,
      taFaNotes: '',
      intendedManagement: '',
      actualManagement: '',
      managementRating: null,
      learnings: '',
    });
    this.showSidebar.set(true);
    this.optionLegs.set([]);
    this.stockLegs.set([]);
    this.showLegPicker.set(false);
  }

  onCancel(): void {
    this.showSidebar.set(false);
    this.selected.set(null);
    this.isCreating.set(false);
  }

  toggleQuillToolbar(): void {
    this.showToolbar.update(v => !v);
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
      typeOfTrade: v.typeOfTrade,
      notes: v.notes || null,
      directional: v.directional || null,
      timeframe: v.timeframe || null,
      budget: v.budget,
      strategy: v.strategy,
      newsCatalyst: v.newsCatalyst ?? false,
      recentEarnings: v.recentEarnings ?? false,
      sectorSupport: v.sectorSupport ?? false,
      ath: v.ath ?? false,
      rvol: v.rvol ? Number(v.rvol) : null,
      institutionalSupport: v.institutionalSupport || null,
      gapPct: v.gapPct ? Number(v.gapPct) : null,
      xAtrMove: v.xAtrMove ? Number(v.xAtrMove) : null,
      taFaNotes: v.taFaNotes || null,
      intendedManagement: v.intendedManagement || null,
      actualManagement: v.actualManagement || null,
      managementRating: v.managementRating || null,
      learnings: v.learnings || null,
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
    if (!this.selected()) return;
    if (!confirm(`Delete trade for ${this.selected()!.symbol}?`)) return;
    this.service.delete(this.selected()!.id).subscribe({
      next: () => {
        this.notify.success('Trade deleted');
        this.load();
        this.onCancel();
      },
      error: () => this.notify.error('Failed to delete trade'),
    });
  }

  // --- Leg management ---

  loadLegs(tradeId: number): void {
    this.service.getById(tradeId).subscribe({
      next: (detail) => {
        this.optionLegs.set(detail.optionPositions ?? []);
        this.stockLegs.set(detail.stockPositions ?? []);
      },
      error: () => this.notify.error('Failed to load legs'),
    });
  }

  onOpenLegPicker(): void {
    const symbol = this.selected()?.symbol;
    this.selectedLegIds.set({ options: new Set(), stocks: new Set() });
    this.showLegPicker.set(true);

    forkJoin([
      this.service.getUnassignedOptionPositions(symbol),
      this.service.getUnassignedStockPositions(symbol),
    ]).subscribe({
      next: ([options, stocks]) => {
        this.unassignedOptions.set(options);
        this.unassignedStocks.set(stocks);
      },
      error: () => this.notify.error('Failed to load unassigned positions'),
    });
  }

  toggleLegSelection(type: 'options' | 'stocks', id: number): void {
    this.selectedLegIds.update(prev => {
      const set = new Set(prev[type]);
      if (set.has(id)) set.delete(id); else set.add(id);
      return { ...prev, [type]: set };
    });
  }

  onAssignSelected(): void {
    const tradeId = this.selected()?.id;
    if (!tradeId) return;

    const ids = this.selectedLegIds();
    const assigns: Observable<void>[] = [];
    ids.options.forEach(id => assigns.push(this.service.assignOptionPosition(id, tradeId)));
    ids.stocks.forEach(id => assigns.push(this.service.assignStockPosition(id, tradeId)));

    if (assigns.length === 0) return;

    forkJoin(assigns).subscribe({
      next: () => {
        this.notify.success(`${assigns.length} position(s) assigned`);
        this.showLegPicker.set(false);
        this.loadLegs(tradeId);
      },
      error: () => this.notify.error('Failed to assign positions'),
    });
  }

  onUnassignOption(positionId: number): void {
    this.service.assignOptionPosition(positionId, null).subscribe({
      next: () => {
        this.optionLegs.update(legs => legs.filter(l => l.id !== positionId));
        this.notify.success('Position unassigned');
      },
      error: () => this.notify.error('Failed to unassign'),
    });
  }

  onUnassignStock(positionId: number): void {
    this.service.assignStockPosition(positionId, null).subscribe({
      next: () => {
        this.stockLegs.update(legs => legs.filter(l => l.id !== positionId));
        this.notify.success('Position unassigned');
      },
      error: () => this.notify.error('Failed to unassign'),
    });
  }
}
