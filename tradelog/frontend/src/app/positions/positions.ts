import { Component, OnInit, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatTableModule } from '@angular/material/table';
import { PositionsService, Position, PositionUpsert, PositionTypes, CloseReasons } from './positions.service';
import { ContentArea } from '../shared/content-area/content-area';
import { ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatButtonModule } from '@angular/material/button';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { MatDatepickerModule } from '@angular/material/datepicker';
import { MatNativeDateModule } from '@angular/material/core';
import { InstrumentsService, Instrument } from '../instruments/instruments.service';

@Component({
  selector: 'app-positions',
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
    MatCheckboxModule,
    MatDatepickerModule,
    MatNativeDateModule,
  ],
  templateUrl: './positions.html',
  host: {class: 'flex flex-col flex-1'}
})
export class Positions implements OnInit {
  private positionsService = inject(PositionsService);
  private instrumentsService = inject(InstrumentsService);
  private fb = inject(FormBuilder);

  positions: Position[] = [];
  instruments: Instrument[] = [];
  displayedColumns: string[] = [
    'id',
    'instrumentId',
    'contractId',
    'type',
    'opened',
    'expiry',
    'closed',
    'size',
    'strike',
    'cost',
    'close',
    'comission',
    'multiplier',
    'closeReasons',
  ];

  // sidebar state
  showSidebar = false;
  form!: FormGroup;
  isCreating = false;
  selected: Position | null = null;

  // CloseReasons helpers
  readonly CloseReasons = CloseReasons;
  readonly closeReasonsList = [
    { label: 'Take Loss', value: CloseReasons.TakeLoss },
    { label: 'Take Profit', value: CloseReasons.TakeProfit },
    { label: 'Roll', value: CloseReasons.Roll },
    { label: 'Assumption Invalidated', value: CloseReasons.AssumptionInvalidated },
    { label: 'Time Limit', value: CloseReasons.TimeLimit },
    { label: 'Other', value: CloseReasons.Other },
  ];

  hasReason(flag: number): boolean {
    const v = this.form?.get('closeReasons')?.value ?? 0;
    return (Number(v) & flag) === flag;
  }

  toggleReason(flag: number, checked: boolean): void {
    const control = this.form.get('closeReasons');
    const current = Number(control?.value ?? 0);
    const next = checked ? (current | flag) : (current & ~flag);
    control?.setValue(next === 0 ? null : next);
  }

  renderCloseReasons(value?: number | null): string {
    const v = Number(value ?? 0);
    if (!v) return '—';
    return this.closeReasonsList
      .filter(r => (v & r.value) === r.value)
      .map(r => r.label)
      .join(', ');
  }

  ngOnInit(): void {
    this.load();
    this.instrumentsService.getInstruments().subscribe({
      next: (data) => (this.instruments = data ?? []),
      error: (err) => {
        console.error('Failed to load instruments', err);
        this.instruments = [];
      }
    });

    this.form = this.fb.group({
      id: [{ value: null, disabled: true }],
      instrumentId: [null, [Validators.required]],
      instrumentSpecifics: ['', [Validators.maxLength(20)]],
      contractId: ['', [Validators.required, Validators.maxLength(20)]],
      type: [PositionTypes.Underlying, [Validators.required]],
      opened: [null, [Validators.required]],
      expiry: [null, [Validators.required]],
      closed: [null],
      size: [1, [Validators.required]],
      strike: [0, [Validators.required]],
      cost: [0, [Validators.required]],
      close: [null],
      comission: [null],
      multiplier: [1, [Validators.required]],
      closeReasons: [null]
    });
  }

  load(): void {
    this.positionsService.getPositions().subscribe({
      next: (data) => (this.positions = data ?? []),
      error: (err) => {
        console.error('Failed to load positions', err);
        this.positions = [];
      },
    });
  }

  onRowSelect(row: Position): void {
    this.isCreating = false;
    this.selected = row;
    this.form.reset({
      id: row.id,
      instrumentId: row.instrumentId,
      instrumentSpecifics: (row as any).instrumentSpecifics ?? '',
      contractId: row.contractId,
      type: row.type,
      opened: row.opened ? new Date(row.opened) : null,
      expiry: row.expiry ? new Date(row.expiry) : null,
      closed: row.closed ? new Date(row.closed) : null,
      size: row.size,
      strike: row.strike,
      cost: row.cost,
      close: row.close ?? null,
      comission: row.comission ?? null,
      multiplier: row.multiplier,
      closeReasons: row.closeReasons ?? null
    });
    this.showSidebar = true;
  }

  onNew(): void {
    this.isCreating = true;
    this.selected = null;
    this.form.reset({
      id: null,
      instrumentId: null,
      instrumentSpecifics: '',
      contractId: '',
      type: PositionTypes.Underlying,
      opened: null,
      expiry: null,
      closed: null,
      size: 1,
      strike: 0,
      cost: 0,
      close: null,
      comission: null,
      multiplier: 1,
      closeReasons: null
    });
    this.showSidebar = true;
  }

  onCancel(): void {
    this.showSidebar = false;
    this.selected = null;
    this.isCreating = false;
  }

  private toIsoOrNull(d: any): string | null {
    if (!d) return null;
    const date = d instanceof Date ? d : new Date(d);
    return isNaN(date.getTime()) ? null : date.toISOString();
  }

  onSave(): void {
    if (this.form.invalid) {
      this.form.markAllAsTouched();
      return;
    }
    const v = this.form.getRawValue();
    const payload: PositionUpsert = {
      id: v.id ?? undefined,
      instrumentId: Number(v.instrumentId),
      contractId: v.contractId,
      type: v.type,
      opened: this.toIsoOrNull(v.opened) as string, // required
      expiry: this.toIsoOrNull(v.expiry) as string, // required
      closed: this.toIsoOrNull(v.closed),
      size: Number(v.size),
      strike: Number(v.strike),
      cost: Number(v.cost),
      close: v.close !== null && v.close !== undefined && v.close !== '' ? Number(v.close) : null,
      comission: v.comission !== null && v.comission !== undefined && v.comission !== '' ? Number(v.comission) : null,
      multiplier: Number(v.multiplier),
      closeReasons: v.closeReasons !== null && v.closeReasons !== undefined && v.closeReasons !== ''
        ? Number(v.closeReasons)
        : null,
      // instrumentSpecifics is not part of Position interface but may be supported by backend per model; include if present
      ...(v.instrumentSpecifics ? { instrumentSpecifics: v.instrumentSpecifics } as any : {})
    } as any;

    const obs = this.isCreating || !payload.id
      ? this.positionsService.createPosition(payload)
      : this.positionsService.updatePosition(payload.id as number, payload);

    obs.subscribe({
      next: () => {
        this.load();
        this.onCancel();
      },
      error: (err) => console.error('Save failed', err)
    });
  }
}
