import { Component, inject } from '@angular/core';
import { MatRadioModule } from '@angular/material/radio';
import { MatCardModule } from '@angular/material/card';
import { DateFormatService, DATE_FORMATS, DateFormatId } from '../shared/date-format.service';
import { AppDatePipe } from '../shared/app-date.pipe';

@Component({
  selector: 'app-settings',
  standalone: true,
  imports: [MatRadioModule, MatCardModule, AppDatePipe],
  template: `
    <div class="p-4 flex flex-col gap-4 max-w-[32rem]">
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
    </div>
  `,
  host: { class: 'flex flex-col flex-1' },
})
export class SettingsComponent {
  protected dateFormat = inject(DateFormatService);
  protected formats = DATE_FORMATS;
  protected sampleDate = new Date(2026, 3, 10); // 2026-04-10
}
