import { Component, inject, signal } from '@angular/core';
import { DatePipe } from '@angular/common';
import { MatTableModule } from '@angular/material/table';
import { ContentArea } from '../shared/content-area/content-area';
import { ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatDatepickerModule } from '@angular/material/datepicker';
import { MatNativeDateModule } from '@angular/material/core';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { QuillModule } from 'ngx-quill';
import { WeeklyPrepService, WeeklyPrep as WeeklyPrepModel, WeeklyPrepUpsert } from './weekly-prep.service';
import { toIsoOrNull } from '../shared/utils';
import { NotificationService } from '../shared/notification.service';

@Component({
  selector: 'app-weekly-prep',
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
    MatDatepickerModule,
    MatNativeDateModule,
    MatProgressBarModule,
    QuillModule,
    DatePipe,
  ],
  templateUrl: './weekly-prep.html',
  host: { class: 'flex flex-col flex-1' },
})
export class WeeklyPrepComponent {
  private service = inject(WeeklyPrepService);
  private fb = inject(FormBuilder);
  private notify = inject(NotificationService);

  loading = signal(false);

  entries = signal<WeeklyPrepModel[]>([]);
  displayedColumns = ['date', 'indexBias', 'breadth', 'currentPortfolioRisk', 'scanningFor'];

  showSidebar = signal(false);
  form!: FormGroup;
  isCreating = signal(false);
  selected = signal<WeeklyPrepModel | null>(null);

  filterYear = signal<number | null>(null);
  availableYears: number[] = [];

  quillModules: any = {
    toolbar: [
      ['bold', 'italic', 'underline'],
      [{ list: 'ordered' }, { list: 'bullet' }],
      ['link', 'blockquote'],
      ['clean'],
    ],
  };
  quillModulesOff: any = { toolbar: false };

  constructor() {
    const currentYear = new Date().getFullYear();
    this.availableYears = Array.from({ length: 5 }, (_, i) => currentYear - i);
    this.load();
    this.form = this.fb.group({
      id: [{ value: null, disabled: true }],
      date: [null, [Validators.required]],
      indexBias: [''],
      breadth: [''],
      notableSectors: [''],
      volatilityNotes: [''],
      openPositionsRequiringManagement: [''],
      currentPortfolioRisk: [''],
      portfolioNotes: [''],
      scanningFor: [''],
      indexSectorPreference: [''],
      watchlist: [''],
      learnings: [''],
      focusForImprovement: [''],
      externalComments: [''],
    });
  }

  load(): void {
    this.loading.set(true);
    const filters: any = {};
    if (this.filterYear()) filters.year = this.filterYear();
    this.service.getAll(filters).subscribe({
      next: (data) => {
        this.entries.set(data ?? []);
        this.loading.set(false);
      },
      error: () => {
        this.notify.error('Failed to load weekly prep entries');
        this.loading.set(false);
      },
    });
  }

  onFilterChange(): void {
    this.load();
  }

  onRowSelect(row: WeeklyPrepModel): void {
    this.isCreating.set(false);
    this.selected.set(row);
    this.form.reset({
      id: row.id,
      date: row.date ? new Date(row.date) : null,
      indexBias: row.indexBias ?? '',
      breadth: row.breadth ?? '',
      notableSectors: row.notableSectors ?? '',
      volatilityNotes: row.volatilityNotes ?? '',
      openPositionsRequiringManagement: row.openPositionsRequiringManagement ?? '',
      currentPortfolioRisk: row.currentPortfolioRisk ?? '',
      portfolioNotes: row.portfolioNotes ?? '',
      scanningFor: row.scanningFor ?? '',
      indexSectorPreference: row.indexSectorPreference ?? '',
      watchlist: row.watchlist ?? '',
      learnings: row.learnings ?? '',
      focusForImprovement: row.focusForImprovement ?? '',
      externalComments: row.externalComments ?? '',
    });
    this.showSidebar.set(true);
  }

  onNew(): void {
    this.isCreating.set(true);
    this.selected.set(null);
    // Pre-fill date to the upcoming Monday
    const now = new Date();
    const day = now.getDay();
    const daysUntilMonday = day <= 1 ? (1 - day) : (8 - day);
    const nextMonday = new Date(now);
    nextMonday.setDate(now.getDate() + daysUntilMonday);

    this.form.reset({
      id: null,
      date: nextMonday,
      indexBias: '',
      breadth: '',
      notableSectors: '',
      volatilityNotes: '',
      openPositionsRequiringManagement: '',
      currentPortfolioRisk: '',
      portfolioNotes: '',
      scanningFor: '',
      indexSectorPreference: '',
      watchlist: '',
      learnings: '',
      focusForImprovement: '',
      externalComments: '',
    });
    this.showSidebar.set(true);
  }

  onCancel(): void {
    this.showSidebar.set(false);
    this.selected.set(null);
    this.isCreating.set(false);
  }

  onSave(): void {
    if (this.form.invalid) {
      this.form.markAllAsTouched();
      return;
    }
    const v = this.form.getRawValue();
    const payload: WeeklyPrepUpsert = {
      id: v.id ?? undefined,
      date: toIsoOrNull(v.date) as string,
      indexBias: v.indexBias || null,
      breadth: v.breadth || null,
      notableSectors: v.notableSectors || null,
      volatilityNotes: v.volatilityNotes || null,
      openPositionsRequiringManagement: v.openPositionsRequiringManagement || null,
      currentPortfolioRisk: v.currentPortfolioRisk || null,
      portfolioNotes: v.portfolioNotes || null,
      scanningFor: v.scanningFor || null,
      indexSectorPreference: v.indexSectorPreference || null,
      watchlist: v.watchlist || null,
      learnings: v.learnings || null,
      focusForImprovement: v.focusForImprovement || null,
      externalComments: v.externalComments || null,
    };

    const obs = this.isCreating() || !payload.id
      ? this.service.create(payload)
      : this.service.update(payload.id as number, payload);

    obs.subscribe({
      next: () => {
        this.notify.success(this.isCreating() ? 'Weekly prep created' : 'Weekly prep updated');
        this.load();
        this.onCancel();
      },
      error: () => this.notify.error('Failed to save weekly prep'),
    });
  }

  onDelete(): void {
    if (!this.selected()) return;
    if (!confirm(`Delete weekly prep for ${this.selected()!.date}?`)) return;
    this.service.delete(this.selected()!.id).subscribe({
      next: () => {
        this.notify.success('Weekly prep deleted');
        this.load();
        this.onCancel();
      },
      error: () => this.notify.error('Failed to delete weekly prep'),
    });
  }
}
