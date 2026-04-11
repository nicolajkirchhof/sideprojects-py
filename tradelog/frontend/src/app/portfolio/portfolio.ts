import { Component, inject, signal } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { MatTableModule } from '@angular/material/table';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { ContentArea } from '../shared/content-area/content-area';
import { PortfolioService, PortfolioDto, PortfolioUpsert } from './portfolio.service';
import { pnlColor } from '../shared/utils';
import { NotificationService } from '../shared/notification.service';
import { LookupService } from '../shared/lookup.service';

@Component({
  selector: 'app-portfolio',
  standalone: true,
  imports: [
    MatTableModule,
    MatButtonModule,
    MatIconModule,
    ContentArea,
    ReactiveFormsModule,
    MatFormFieldModule,
    MatInputModule,
    MatSelectModule,
    MatProgressBarModule,
    DecimalPipe,
  ],
  templateUrl: './portfolio.html',
  host: { class: 'flex flex-col flex-1' },
})
export class PortfolioComponent {
  private service = inject(PortfolioService);
  private fb = inject(FormBuilder);
  private notify = inject(NotificationService);
  protected lookup = inject(LookupService);

  loading = signal(false);

  entries = signal<PortfolioDto[]>([]);
  displayedColumns = ['budget', 'strategy', 'minAllocation', 'maxAllocation', 'currentAllocation', 'pnl'];

  showSidebar = signal(false);
  form!: FormGroup;
  isCreating = signal(false);
  editMode = signal(false);
  selected = signal<PortfolioDto | null>(null);

  pnlColor = pnlColor;

  constructor() {
    this.load();
    this.form = this.fb.group({
      id: [{ value: null, disabled: true }],
      budget: [null, [Validators.required]],
      strategy: [null, [Validators.required]],
      minAllocation: [0, [Validators.required]],
      maxAllocation: [0, [Validators.required]],
    });
  }

  load(): void {
    this.loading.set(true);
    this.service.getAll().subscribe({
      next: (data) => {
        this.entries.set(data ?? []);
        this.loading.set(false);
        if (!this.selected() && (data?.length ?? 0) > 0) {
          this.onRowSelect(data![0]);
        }
      },
      error: () => {
        this.notify.error('Failed to load portfolio');
        this.loading.set(false);
      },
    });
  }

  onRowSelect(row: PortfolioDto): void {
    this.isCreating.set(false);
    this.editMode.set(false);
    this.selected.set(row);
    this.form.reset({
      id: row.id,
      budget: row.budget,
      strategy: row.strategy,
      minAllocation: row.minAllocation,
      maxAllocation: row.maxAllocation,
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

  onNew(): void {
    this.isCreating.set(true);
    this.editMode.set(true);
    this.selected.set(null);
    this.form.reset({ id: null, budget: null, strategy: null, minAllocation: 0, maxAllocation: 0 });
    this.form.enable({ emitEvent: false });
    this.form.get('id')?.disable({ emitEvent: false });
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
    const payload: PortfolioUpsert = {
      id: v.id ?? undefined,
      budget: v.budget,
      strategy: v.strategy,
      minAllocation: Number(v.minAllocation),
      maxAllocation: Number(v.maxAllocation),
    };

    const done = () => {
      this.notify.success(this.isCreating() ? 'Allocation created' : 'Allocation updated');
      this.load();
      this.onCancel();
    };
    const fail = () => this.notify.error('Failed to save allocation');

    if (this.isCreating() || !payload.id) {
      this.service.create(payload).subscribe({ next: done, error: fail });
    } else {
      this.service.update(payload.id as number, payload).subscribe({ next: done, error: fail });
    }
  }

  onDelete(): void {
    if (!this.selected()) return;
    if (!confirm('Delete this portfolio allocation?')) return;
    this.service.delete(this.selected()!.id).subscribe({
      next: () => {
        this.notify.success('Allocation deleted');
        this.load();
        this.onCancel();
      },
      error: () => this.notify.error('Failed to delete allocation'),
    });
  }

  allocationClass(row: PortfolioDto): string {
    if (row.currentAllocation > row.maxAllocation) return 'text-red-400';
    if (row.currentAllocation < row.minAllocation) return 'text-amber-400';
    return '';
  }
}
