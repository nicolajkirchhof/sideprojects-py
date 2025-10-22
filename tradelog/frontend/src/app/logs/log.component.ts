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
import { LogsService, LogEntry, LogUpsert, Sentiments } from './logs.service';
import { InstrumentsService, Instrument } from '../instruments/instruments.service';
import { QuillModule } from 'ngx-quill';

@Component({
  selector: 'app-trade-ideas',
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
    DatePipe
  ],
  templateUrl: './log.component.html',
  host: {class: 'flex flex-col flex-1'}
})
export class Log implements OnInit {
  private logsService = inject(LogsService);
  private instrumentsService = inject(InstrumentsService);
  private fb = inject(FormBuilder);

  logs: LogEntry[] = [];
  instruments: Instrument[] = [];
  displayedColumns: string[] = [
    'id',
    'instrumentId',
    'date',
    'notes',
    'strategy',
    'sentiment'
  ];

  // sidebar state
  showSidebar = false;
  form!: FormGroup;
  isCreating = false;
  selected: LogEntry | null = null;

  // Quill toolbar toggle state and modules
  showToolbar = false;
  toolbarOptions: any = [
    ['bold', 'italic', 'underline', 'strike'],
    [{ header: [1, 2, 3, 4, 5, 6, false] }],
    [{ list: 'ordered' }, { list: 'bullet' }],
    [{ script: 'sub' }, { script: 'super' }],
    [{ indent: '-1' }, { indent: '+1' }],
    [{ direction: 'rtl' }],
    [{ size: ['small', false, 'large', 'huge'] }],
    [{ color: [] }, { background: [] }],
    [{ font: [] }],
    [{ align: [] }],
    ['link', 'image', 'code-block', 'blockquote'],
    ['clean']
  ];
  // Toolbar configurations
  quillModules: any = { toolbar: this.toolbarOptions };
  quillModulesOff: any = { toolbar: false };

  // Default notes template for the Quill editor
  notesTemplate: string = `
    <p><strong>General</strong></p>
    <p><br></p>
    <p><strong>Technical Analysis</strong></p>
    <p><br></p>
    <p><strong>Fundamental Analysis</strong></p>
    <p><br></p>
    <p><strong>Expected Outcome</strong></p>
    <p><br></p>
    <p><strong>Learnings</strong></p>
    <p><br></p>
  `;

  toggleQuillToolbar(): void {
    this.showToolbar = !this.showToolbar;
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
      date: [null, [Validators.required]],
      notes: [''],
      strategy: [''],
      sentiment: [[] as number[]]
    });
  }

  load(): void {
    this.logsService.getLogs().subscribe({
      next: (data) => (this.logs = data ?? []),
      error: (err) => {
        console.error('Failed to load logs', err);
        this.logs = [];
      },
    });
  }

  onRowSelect(row: LogEntry): void {
    this.isCreating = false;
    this.selected = row;
    this.form.reset({
      id: row.id,
      instrumentId: row.instrumentId,
      date: row.date ? new Date(row.date) : null,
      notes: row.notes ?? '',
      strategy: row.strategy ?? '',
      sentiment: this.sentimentToArray(row.sentiment)
    });
    this.showSidebar = true;
  }

  onNew(): void {
    this.isCreating = true;
    this.selected = null;
    this.form.reset({
      id: null,
      instrumentId: null,
      date: null,
      notes: this.notesTemplate,
      strategy: '',
      sentiment: []
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

  private sentimentToArray(value?: number | null): number[] {
    if (!value) return [];
    const vals: number[] = [];
    const all = [Sentiments.Bullish, Sentiments.Neutral, Sentiments.Bearish];
    for (const v of all) if ((value & v) === v) vals.push(v);
    return vals;
  }

  sentimentToLabels(value?: number | null): string {
    if (!value) return 'None';
    const parts: string[] = [];
    if ((value & Sentiments.Bullish) === Sentiments.Bullish) parts.push('Bullish');
    if ((value & Sentiments.Neutral) === Sentiments.Neutral) parts.push('Neutral');
    if ((value & Sentiments.Bearish) === Sentiments.Bearish) parts.push('Bearish');
    return parts.length ? parts.join(', ') : 'None';
  }

  private arrayToSentiment(values: number[] | null | undefined): number | null {
    if (!values || !values.length) return null;
    return values.reduce((acc, v) => acc | Number(v), 0);
  }

  onSave(): void {
    if (this.form.invalid) {
      this.form.markAllAsTouched();
      return;
    }
    const v = this.form.getRawValue() as any;
    const payload: LogUpsert = {
      id: v.id ?? undefined,
      instrumentId: Number(v.instrumentId),
      date: this.toIsoOrNull(v.date) as string, // required
      notes: v.notes || null,
      strategy: v.strategy || null,
      sentiment: this.arrayToSentiment(v.sentiment),
    };

    const obs = this.isCreating || !payload.id
      ? this.logsService.createLog(payload)
      : this.logsService.updateLog(payload.id as number, payload);

    obs.subscribe({
      next: () => {
        this.load();
        this.onCancel();
      },
      error: (err) => console.error('Save failed', err)
    });
  }
}
