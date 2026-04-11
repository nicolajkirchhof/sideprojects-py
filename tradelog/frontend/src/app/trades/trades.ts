import { Component, inject, signal } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { AppDatePipe } from '../shared/app-date.pipe';
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
  TradesService, Trade, TradeUpsert, OptionLegDto, StockLegDto, TradeEventDto,
  TradeEventType, TRADE_EVENT_TYPE_LABELS,
} from './trades.service';
import { forkJoin, Observable } from 'rxjs';
import { toIsoOrNull } from '../shared/utils';
import { NotificationService } from '../shared/notification.service';
import { LookupService } from '../shared/lookup.service';

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
    AppDatePipe,
    DecimalPipe,
  ],
  templateUrl: './trades.html',
  host: { class: 'flex flex-col flex-1' },
})
export class Trades {
  private service = inject(TradesService);
  private fb = inject(FormBuilder);
  private notify = inject(NotificationService);
  protected lookup = inject(LookupService);

  loading = signal(false);

  trades = signal<Trade[]>([]);
  displayedColumns = ['symbol', 'date', 'typeOfTrade', 'directional', 'budget', 'strategy', 'managementRating'];

  // Sidebar state
  showSidebar = signal(false);
  form!: FormGroup;
  isCreating = signal(false);
  editMode = signal(false);
  selected = signal<Trade | null>(null);

  // Filter state
  filterBudget = signal<number | null>(null);
  filterStrategy = signal<number | null>(null);

  // Follow-up chain
  childTradeIds = signal<number[]>([]);

  // Legs (assigned positions)
  optionLegs = signal<OptionLegDto[]>([]);
  stockLegs = signal<StockLegDto[]>([]);

  // Leg picker
  showLegPicker = signal(false);
  unassignedOptions = signal<OptionLegDto[]>([]);
  unassignedStocks = signal<StockLegDto[]>([]);
  selectedLegIds = signal<{ options: Set<number>; stocks: Set<number> }>({ options: new Set(), stocks: new Set() });

  // Trade events
  events = signal<TradeEventDto[]>([]);
  showEventForm = signal(false);
  eventTypes = Object.values(TradeEventType);
  eventTypeLabel: Record<string, string> = TRADE_EVENT_TYPE_LABELS;
  eventForm!: FormGroup;

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
      parentTradeId: [null],
    });
    this.eventForm = this.fb.group({
      type: [null, [Validators.required]],
      date: [null, [Validators.required]],
      notes: [''],
      pnlImpact: [null],
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

  onFilterChange(): void {
    this.load();
  }

  onRowSelect(row: Trade): void {
    this.isCreating.set(false);
    this.editMode.set(false);
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
      parentTradeId: row.parentTradeId ?? null,
    });
    this.form.disable({ emitEvent: false });
    this.showSidebar.set(true);
    this.showLegPicker.set(false);
    this.loadLegs(row.id);
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
      typeOfTrade: null,
      notes: '',
      directional: null,
      timeframe: null,
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
      parentTradeId: null,
    });
    this.showSidebar.set(true);
    this.optionLegs.set([]);
    this.stockLegs.set([]);
    this.events.set([]);
    this.showLegPicker.set(false);
    this.showEventForm.set(false);
  }

  onCancel(): void {
    this.showSidebar.set(false);
    this.selected.set(null);
    this.isCreating.set(false);
    this.editMode.set(false);
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
      parentTradeId: v.parentTradeId || null,
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
        this.childTradeIds.set(detail.childTradeIds ?? []);
        this.events.set(detail.events ?? []);
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

  onCreateFollowUp(): void {
    const parent = this.selected();
    if (!parent) return;

    this.isCreating.set(true);
    this.selected.set(null);
    this.form.reset({
      id: null,
      symbol: parent.symbol,
      date: new Date(),
      typeOfTrade: parent.typeOfTrade,
      notes: '',
      directional: parent.directional ?? null,
      timeframe: parent.timeframe ?? null,
      budget: parent.budget,
      strategy: parent.strategy,
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
      parentTradeId: parent.id,
    });
    this.optionLegs.set([]);
    this.stockLegs.set([]);
    this.childTradeIds.set([]);
    this.events.set([]);
    this.showLegPicker.set(false);
    this.showEventForm.set(false);
  }

  onSelectTradeById(tradeId: number): void {
    const trade = this.trades().find(t => t.id === tradeId);
    if (trade) {
      this.onRowSelect(trade);
    } else {
      this.service.getById(tradeId).subscribe({
        next: (detail) => this.onRowSelect(detail),
        error: () => this.notify.error('Trade not found — try clearing filters'),
      });
    }
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

  // --- Trade events ---

  onOpenEventForm(): void {
    this.eventForm.reset({ type: null, date: new Date(), notes: '', pnlImpact: null });
    this.showEventForm.set(true);
  }

  onSaveEvent(): void {
    if (this.eventForm.invalid) {
      this.eventForm.markAllAsTouched();
      return;
    }
    const tradeId = this.selected()?.id;
    if (!tradeId) return;

    const v = this.eventForm.getRawValue();
    this.service.createEvent(tradeId, {
      type: v.type,
      date: toIsoOrNull(v.date) as string,
      notes: v.notes || null,
      pnlImpact: v.pnlImpact ? Number(v.pnlImpact) : null,
    }).subscribe({
      next: (created) => {
        this.events.update(list => [...list, created].sort((a, b) => a.date.localeCompare(b.date)));
        this.showEventForm.set(false);
        this.notify.success('Event added');
      },
      error: () => this.notify.error('Failed to add event'),
    });
  }

  onDeleteEvent(eventId: number): void {
    if (!confirm('Delete this event?')) return;
    this.service.deleteEvent(eventId).subscribe({
      next: () => {
        this.events.update(list => list.filter(e => e.id !== eventId));
        this.notify.success('Event deleted');
      },
      error: () => this.notify.error('Failed to delete event'),
    });
  }
}
