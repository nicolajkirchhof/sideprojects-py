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
import { MatDatepickerModule } from '@angular/material/datepicker';
import { MatNativeDateModule } from '@angular/material/core';
import { QuillModule } from 'ngx-quill';
import { WeeklyPrepService, WeeklyPrep as WeeklyPrepModel, WeeklyPrepUpsert } from './weekly-prep.service';
import { toIsoOrNull } from '../shared/utils';

@Component({
  selector: 'app-weekly-prep',
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
    MatDatepickerModule,
    MatNativeDateModule,
    QuillModule,
    DatePipe,
  ],
  templateUrl: './weekly-prep.html',
  host: { class: 'flex flex-col flex-1' },
})
export class WeeklyPrepComponent implements OnInit {
  private service = inject(WeeklyPrepService);
  private fb = inject(FormBuilder);

  entries: WeeklyPrepModel[] = [];
  displayedColumns = ['date', 'indexBias', 'breadth', 'currentPortfolioRisk', 'scanningFor'];

  showSidebar = false;
  form!: FormGroup;
  isCreating = false;
  selected: WeeklyPrepModel | null = null;

  filterYear: number | null = null;
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

  ngOnInit(): void {
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
    const filters: any = {};
    if (this.filterYear) filters.year = this.filterYear;
    this.service.getAll(filters).subscribe({
      next: (data) => (this.entries = data ?? []),
      error: (err) => console.error('Failed to load weekly prep entries', err),
    });
  }

  onFilterChange(): void {
    this.load();
  }

  onRowSelect(row: WeeklyPrepModel): void {
    this.isCreating = false;
    this.selected = row;
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
    this.showSidebar = true;
  }

  onNew(): void {
    this.isCreating = true;
    this.selected = null;
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

    const obs = this.isCreating || !payload.id
      ? this.service.create(payload)
      : this.service.update(payload.id as number, payload);

    obs.subscribe({
      next: () => {
        this.load();
        this.onCancel();
      },
      error: (err) => console.error('Save failed', err),
    });
  }

  onDelete(): void {
    if (!this.selected) return;
    if (!confirm(`Delete weekly prep for ${this.selected.date}?`)) return;
    this.service.delete(this.selected.id).subscribe({
      next: () => {
        this.load();
        this.onCancel();
      },
      error: (err) => console.error('Delete failed', err),
    });
  }
}
