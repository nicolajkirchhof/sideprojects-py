import { Component, OnInit, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatTableModule } from '@angular/material/table';
import { InstrumentsService, Instrument, InstrumentUpsert, SecTypes } from './instruments.service';
import { ContentArea } from '../shared/content-area/content-area';
import { ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatButtonModule } from '@angular/material/button';

@Component({
  selector: 'app-instruments',
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
  ],
  templateUrl: './instruments.html',
  host: {class: 'flex flex-col flex-1'}
})
export class Instruments implements OnInit {
  private instrumentsService = inject(InstrumentsService);
  private fb = inject(FormBuilder);

  instruments: Instrument[] = [];
  displayedColumns: string[] = [
    'id',
    'secType',
    'symbol',
    'multiplier',
    'sector',
    'subsector',
  ];

  // sidebar state
  showSidebar = false;
  form!: FormGroup;
  isCreating = false;
  selected: Instrument | null = null;

  ngOnInit(): void {
    this.load();
    this.form = this.fb.group({
      id: [{ value: null, disabled: true }],
      secType: [SecTypes.Stock, [Validators.required]],
      symbol: ['', [Validators.required, Validators.maxLength(50)]],
      multiplier: [1, [Validators.required]],
      sector: ['', [Validators.maxLength(50)]],
      subsector: ['', [Validators.maxLength(50)]],
    });
  }

  load(): void {
    this.instrumentsService.getInstruments().subscribe({
      next: (data) => (this.instruments = data ?? []),
      error: (err) => {
        console.error('Failed to load instruments', err);
        this.instruments = [];
      },
    });
  }

  onRowSelect(row: Instrument): void {
    this.isCreating = false;
    this.selected = row;
    this.form.reset({
      id: row.id,
      secType: row.secType,
      symbol: row.symbol,
      multiplier: row.multiplier,
      sector: row.sector,
      subsector: row.subsector ?? '',
    });
    this.showSidebar = true;
  }

  onNew(): void {
    this.isCreating = true;
    this.selected = null;
    this.form.reset({
      id: null,
      secType: SecTypes.Stock,
      symbol: '',
      multiplier: 1,
      sector: '',
      subsector: '',
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
    const value = this.form.getRawValue();
    const payload: InstrumentUpsert = {
      id: value.id ?? undefined,
      secType: value.secType,
      symbol: value.symbol,
      multiplier: Number(value.multiplier),
      sector: value.sector,
      subsector: value.subsector || null,
    };

    const obs = this.isCreating || !payload.id
      ? this.instrumentsService.createInstrument(payload)
      : this.instrumentsService.updateInstrument(payload.id as number, payload);

    obs.subscribe({
      next: () => {
        this.load();
        this.onCancel();
      },
      error: (err) => console.error('Save failed', err)
    });
  }
}
