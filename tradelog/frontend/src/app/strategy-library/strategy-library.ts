import { Component, inject, signal } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatListModule } from '@angular/material/list';
import { Observable } from 'rxjs';
import { DocumentService, DocumentDto, DocumentUpsert } from '../shared/document.service';
import { NotificationService } from '../shared/notification.service';

@Component({
  selector: 'app-strategy-library',
  standalone: true,
  imports: [
    ReactiveFormsModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatIconModule,
    MatProgressBarModule,
    MatListModule,
  ],
  templateUrl: './strategy-library.html',
  host: { class: 'flex flex-col flex-1' },
})
export class StrategyLibraryComponent {
  private service = inject(DocumentService);
  private route = inject(ActivatedRoute);
  private fb = inject(FormBuilder);
  private notify = inject(NotificationService);

  loading = signal(false);
  documents = signal<DocumentDto[]>([]);
  selected = signal<DocumentDto | null>(null);
  editMode = signal(false);

  form: FormGroup = this.fb.group({
    title: ['', [Validators.required]],
    content: [''],
  });

  constructor() {
    this.load();
  }

  load(): void {
    this.loading.set(true);
    this.service.getAll().subscribe({
      next: (data) => {
        this.documents.set(data ?? []);
        this.loading.set(false);
        // Auto-select from query param (e.g. /strategy-library?doc=5)
        const docId = this.route.snapshot.queryParams['doc'];
        if (docId) {
          const doc = data?.find(d => d.id === +docId);
          if (doc) this.onSelect(doc);
        }
      },
      error: () => {
        this.notify.error('Failed to load documents');
        this.loading.set(false);
      },
    });
  }

  onSelect(doc: DocumentDto): void {
    this.selected.set(doc);
    this.editMode.set(false);
    this.form.reset({ title: doc.title, content: doc.content ?? '' });
    this.form.disable({ emitEvent: false });
  }

  onNew(): void {
    this.selected.set(null);
    this.editMode.set(true);
    this.form.enable({ emitEvent: false });
    this.form.reset({ title: '', content: '' });
  }

  onEdit(): void {
    this.editMode.set(true);
    this.form.enable({ emitEvent: false });
  }

  onCancelEdit(): void {
    const doc = this.selected();
    if (doc) {
      this.onSelect(doc);
    } else {
      this.editMode.set(false);
    }
  }

  onSave(): void {
    if (this.form.invalid) {
      this.form.markAllAsTouched();
      return;
    }
    const v = this.form.getRawValue();
    const payload: DocumentUpsert = {
      title: v.title.trim(),
      content: v.content || null,
    };

    const sel = this.selected();
    const obs: Observable<any> = sel
      ? this.service.update(sel.id, payload)
      : this.service.create(payload);

    obs.subscribe({
      next: (result: any) => {
        this.notify.success(sel ? 'Document updated' : 'Document created');
        this.editMode.set(false);
        this.load();
        // If creating, select the new document after reload
        if (!sel && result?.id) {
          setTimeout(() => {
            const created = this.documents().find(d => d.id === result.id);
            if (created) this.onSelect(created);
          }, 200);
        } else if (sel) {
          // Re-fetch to get updated data
          this.service.getById(sel.id).subscribe({
            next: (updated) => this.onSelect(updated),
          });
        }
      },
      error: () => this.notify.error('Failed to save document'),
    });
  }

  onDelete(): void {
    const sel = this.selected();
    if (!sel) return;
    if (!confirm(`Delete "${sel.title}"?`)) return;
    this.service.delete(sel.id).subscribe({
      next: () => {
        this.notify.success('Document deleted');
        this.selected.set(null);
        this.editMode.set(false);
        this.load();
      },
      error: () => this.notify.error('Failed to delete document'),
    });
  }
}
