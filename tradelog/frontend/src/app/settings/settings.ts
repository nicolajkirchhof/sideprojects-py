import { Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatRadioModule } from '@angular/material/radio';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatInputModule } from '@angular/material/input';
import { MatFormFieldModule } from '@angular/material/form-field';
import { DateFormatService, DATE_FORMATS } from '../shared/date-format.service';
import { LookupService, LookupCategory, LookupValue } from '../shared/lookup.service';
import { AppDatePipe } from '../shared/app-date.pipe';

@Component({
  selector: 'app-settings',
  standalone: true,
  imports: [
    FormsModule, MatRadioModule, MatCardModule, MatButtonModule, MatIconModule,
    MatInputModule, MatFormFieldModule, AppDatePipe,
  ],
  template: `
    <div class="p-4 flex flex-col gap-4 max-w-[32rem]">
      <!-- Date Format -->
      <mat-card appearance="outlined">
        <mat-card-header>
          <mat-card-title class="!text-sm !font-semibold">Date Format</mat-card-title>
        </mat-card-header>
        <mat-card-content class="!pt-3">
          <mat-radio-group
            [value]="dateFormat.format()"
            (change)="dateFormat.set($event.value)"
            class="flex flex-col gap-2">
            @for (opt of formats; track opt.id) {
              <mat-radio-button [value]="opt.id">
                <span>{{ opt.label }}</span>
              </mat-radio-button>
            }
          </mat-radio-group>
          <div class="mt-3 text-xs opacity-60">
            Preview: {{ sampleDate | appDate }}
          </div>
        </mat-card-content>
      </mat-card>

      <!-- Lookup categories -->
      @for (cat of categories; track cat) {
        <mat-card appearance="outlined">
          <mat-card-header>
            <mat-card-title class="!text-sm !font-semibold">{{ categoryLabel(cat) }}</mat-card-title>
          </mat-card-header>
          <mat-card-content class="!pt-3 flex flex-col gap-1">
            <!-- Active values -->
            @for (lv of activeValues(cat); track lv.id; let i = $index; let last = $last) {
              <div class="flex items-center gap-1 group h-8">
                @if (editingId() === lv.id) {
                  <input class="flex-1 text-sm bg-transparent border-b border-white/30 outline-none px-1 py-0.5"
                         [value]="lv.name"
                         (keydown.enter)="onRename(lv.id, $any($event.target).value); editingId.set(null)"
                         (keydown.escape)="editingId.set(null)"
                         (blur)="onRename(lv.id, $any($event.target).value); editingId.set(null)"
                         #renameInput>
                } @else {
                  <span class="flex-1 text-sm cursor-pointer px-1 py-0.5 rounded hover:bg-white/5"
                        (click)="editingId.set(lv.id)">{{ lv.name }}</span>
                }
                <button mat-icon-button type="button" class="!w-6 !h-6 !p-0 opacity-0 group-hover:opacity-60"
                        [disabled]="i === 0"
                        (click)="onMoveUp(cat, lv, i)">
                  <mat-icon class="!text-sm !w-4 !h-4">arrow_upward</mat-icon>
                </button>
                <button mat-icon-button type="button" class="!w-6 !h-6 !p-0 opacity-0 group-hover:opacity-60"
                        [disabled]="last"
                        (click)="onMoveDown(cat, lv, i)">
                  <mat-icon class="!text-sm !w-4 !h-4">arrow_downward</mat-icon>
                </button>
                <button mat-icon-button type="button" class="!w-6 !h-6 !p-0 opacity-0 group-hover:opacity-60"
                        (click)="onDeactivate(lv.id)">
                  <mat-icon class="!text-sm !w-4 !h-4">visibility_off</mat-icon>
                </button>
              </div>
            }

            <!-- Add new value -->
            <div class="flex items-center gap-2 mt-1">
              <mat-form-field appearance="outline" class="flex-1" subscriptSizing="dynamic">
                <input matInput [ngModel]="newValueNames()[cat] || ''"
                       (ngModelChange)="setNewValueName(cat, $event)"
                       placeholder="New value" (keydown.enter)="onAdd(cat)">
              </mat-form-field>
              <button mat-icon-button type="button" [disabled]="!(newValueNames()[cat] || '').trim()" (click)="onAdd(cat)">
                <mat-icon>add</mat-icon>
              </button>
            </div>

            <!-- Inactive values -->
            @if (inactiveValues(cat).length > 0) {
              <div class="mt-2 border-t border-white/10 pt-2">
                <div class="text-xs opacity-40 mb-1">Inactive</div>
                @for (lv of inactiveValues(cat); track lv.id) {
                  <div class="flex items-center gap-1 h-7">
                    <span class="flex-1 text-sm opacity-40 px-1">{{ lv.name }}</span>
                    <button mat-icon-button type="button" class="!w-6 !h-6 !p-0 opacity-40"
                            (click)="onReactivate(lv.id)">
                      <mat-icon class="!text-sm !w-4 !h-4">visibility</mat-icon>
                    </button>
                  </div>
                }
              </div>
            }
          </mat-card-content>
        </mat-card>
      }
    </div>
  `,
  host: { class: 'flex flex-col flex-1 overflow-auto' },
})
export class SettingsComponent {
  protected dateFormat = inject(DateFormatService);
  protected lookup = inject(LookupService);
  protected formats = DATE_FORMATS;
  protected sampleDate = new Date(2026, 3, 10);

  protected categories: LookupCategory[] = [
    'Budget', 'Strategy', 'TypeOfTrade', 'Timeframe', 'Directional', 'ManagementRating',
  ];

  protected editingId = signal<number | null>(null);
  protected newValueNames = signal<Record<string, string>>({});

  private categoryLabels: Record<string, string> = {
    Budget: 'Budgets',
    Strategy: 'Strategies',
    TypeOfTrade: 'Trade Types',
    Timeframe: 'Timeframes',
    Directional: 'Directional Bias',
    ManagementRating: 'Management Ratings',
  };

  categoryLabel(cat: LookupCategory): string {
    return this.categoryLabels[cat] ?? cat;
  }

  activeValues(cat: LookupCategory): LookupValue[] {
    return this.lookup.allByCategory(cat).filter(lv => lv.isActive);
  }

  inactiveValues(cat: LookupCategory): LookupValue[] {
    return this.lookup.allByCategory(cat).filter(lv => !lv.isActive);
  }

  onRename(id: number, newName: string): void {
    const trimmed = newName.trim();
    if (!trimmed) return;
    this.lookup.rename(id, trimmed).subscribe({ next: () => this.lookup.refresh() });
  }

  onMoveUp(cat: LookupCategory, lv: LookupValue, index: number): void {
    const active = this.activeValues(cat);
    if (index <= 0) return;
    const prev = active[index - 1];
    // Swap sort orders
    this.lookup.reorder(lv.id, prev.sortOrder).subscribe({
      next: () => this.lookup.reorder(prev.id, lv.sortOrder).subscribe({
        next: () => this.lookup.refresh(),
      }),
    });
  }

  onMoveDown(cat: LookupCategory, lv: LookupValue, index: number): void {
    const active = this.activeValues(cat);
    if (index >= active.length - 1) return;
    const next = active[index + 1];
    this.lookup.reorder(lv.id, next.sortOrder).subscribe({
      next: () => this.lookup.reorder(next.id, lv.sortOrder).subscribe({
        next: () => this.lookup.refresh(),
      }),
    });
  }

  onDeactivate(id: number): void {
    this.lookup.deactivate(id).subscribe({ next: () => this.lookup.refresh() });
  }

  onReactivate(id: number): void {
    this.lookup.reactivate(id).subscribe({ next: () => this.lookup.refresh() });
  }

  setNewValueName(cat: string, value: string): void {
    this.newValueNames.update(m => ({ ...m, [cat]: value }));
  }

  onAdd(cat: LookupCategory): void {
    const name = (this.newValueNames()[cat] || '').trim();
    if (!name) return;
    this.lookup.create(cat, name).subscribe({
      next: () => {
        this.newValueNames.update(m => ({ ...m, [cat]: '' }));
        this.lookup.refresh();
      },
    });
  }
}
