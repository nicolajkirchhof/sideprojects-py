import { Component, OnInit, inject } from '@angular/core';
import { CommonModule, DatePipe } from '@angular/common';
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
  TradesService, Trade, TradeUpsert,
  Budget, Strategy, TypeOfTrade, DirectionalBias, Timeframe, ManagementRating,
  STRATEGY_LABELS, TYPE_OF_TRADE_LABELS, TIMEFRAME_LABELS, MANAGEMENT_RATING_LABELS,
} from './trades.service';
import { toIsoOrNull } from '../shared/utils';
import { NotificationService } from '../shared/notification.service';

@Component({
  selector: 'app-trades',
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
    MatCheckboxModule,
    MatDatepickerModule,
    MatNativeDateModule,
    MatProgressBarModule,
    QuillModule,
    DatePipe,
  ],
  templateUrl: './trades.html',
  host: { class: 'flex flex-col flex-1' },
})
export class Trades implements OnInit {
  private service = inject(TradesService);
  private fb = inject(FormBuilder);
  private notify = inject(NotificationService);

  loading = false;

  trades: Trade[] = [];
  displayedColumns = ['symbol', 'date', 'typeOfTrade', 'directional', 'budget', 'strategy', 'managementRating'];

  // Sidebar state
  showSidebar = false;
  form!: FormGroup;
  isCreating = false;
  selected: Trade | null = null;

  // Filter state
  filterBudget: Budget | null = null;
  filterStrategy: Strategy | null = null;

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

  // Quill config
  showToolbar = false;
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

  ngOnInit(): void {
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
    this.loading = true;
    const filters: any = {};
    if (this.filterBudget) filters.budget = this.filterBudget;
    if (this.filterStrategy) filters.strategy = this.filterStrategy;
    this.service.getAll(filters).subscribe({
      next: (data) => {
        this.trades = data ?? [];
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

  onRowSelect(row: Trade): void {
    this.isCreating = false;
    this.selected = row;
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
    this.showSidebar = true;
  }

  onNew(): void {
    this.isCreating = true;
    this.selected = null;
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
    this.showSidebar = true;
  }

  onCancel(): void {
    this.showSidebar = false;
    this.selected = null;
    this.isCreating = false;
  }

  toggleQuillToolbar(): void {
    this.showToolbar = !this.showToolbar;
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
    if (!confirm(`Delete trade for ${this.selected.symbol}?`)) return;
    this.service.delete(this.selected.id).subscribe({
      next: () => {
        this.notify.success('Trade deleted');
        this.load();
        this.onCancel();
      },
      error: () => this.notify.error('Failed to delete trade'),
    });
  }
}
