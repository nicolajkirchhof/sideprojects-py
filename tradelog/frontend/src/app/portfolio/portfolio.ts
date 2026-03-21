import { Component, OnInit, inject } from '@angular/core';
import { CommonModule, DecimalPipe } from '@angular/common';
import { MatTableModule } from '@angular/material/table';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { ContentArea } from '../shared/content-area/content-area';
import { PortfolioService, PortfolioDto, PortfolioUpsert } from './portfolio.service';
import { Budget, Strategy, STRATEGY_LABELS } from '../trade-entries/trade-entries.service';
import { pnlColor } from '../shared/utils';

@Component({
  selector: 'app-portfolio',
  standalone: true,
  imports: [
    CommonModule,
    MatTableModule,
    MatButtonModule,
    MatIconModule,
    ContentArea,
    ReactiveFormsModule,
    MatFormFieldModule,
    MatInputModule,
    MatSelectModule,
    DecimalPipe,
  ],
  templateUrl: './portfolio.html',
  host: { class: 'flex flex-col flex-1' },
})
export class PortfolioComponent implements OnInit {
  private service = inject(PortfolioService);
  private fb = inject(FormBuilder);

  entries: PortfolioDto[] = [];
  displayedColumns = ['budget', 'strategy', 'minAllocation', 'maxAllocation', 'currentAllocation', 'pnl'];

  showSidebar = false;
  form!: FormGroup;
  isCreating = false;
  selected: PortfolioDto | null = null;

  budgets = Object.values(Budget);
  strategies = Object.values(Strategy);
  strategyLabel: Record<string, string> = STRATEGY_LABELS;
  pnlColor = pnlColor;

  ngOnInit(): void {
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
    this.service.getAll().subscribe({
      next: (data) => (this.entries = data ?? []),
      error: (err) => console.error('Failed to load portfolio', err),
    });
  }

  onRowSelect(row: PortfolioDto): void {
    this.isCreating = false;
    this.selected = row;
    this.form.reset({
      id: row.id,
      budget: row.budget,
      strategy: row.strategy,
      minAllocation: row.minAllocation,
      maxAllocation: row.maxAllocation,
    });
    this.showSidebar = true;
  }

  onNew(): void {
    this.isCreating = true;
    this.selected = null;
    this.form.reset({ id: null, budget: null, strategy: null, minAllocation: 0, maxAllocation: 0 });
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
    const payload: PortfolioUpsert = {
      id: v.id ?? undefined,
      budget: v.budget,
      strategy: v.strategy,
      minAllocation: Number(v.minAllocation),
      maxAllocation: Number(v.maxAllocation),
    };

    const done = () => { this.load(); this.onCancel(); };
    const fail = (err: any) => console.error('Save failed', err);

    if (this.isCreating || !payload.id) {
      this.service.create(payload).subscribe({ next: done, error: fail });
    } else {
      this.service.update(payload.id as number, payload).subscribe({ next: done, error: fail });
    }
  }

  onDelete(): void {
    if (!this.selected) return;
    if (!confirm('Delete this portfolio allocation?')) return;
    this.service.delete(this.selected.id).subscribe({
      next: () => {
        this.load();
        this.onCancel();
      },
      error: (err) => console.error('Delete failed', err),
    });
  }

  allocationClass(row: PortfolioDto): string {
    if (row.currentAllocation > row.maxAllocation) return 'text-red-400';
    if (row.currentAllocation < row.minAllocation) return 'text-amber-400';
    return '';
  }
}
