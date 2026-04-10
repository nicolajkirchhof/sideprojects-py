import { Pipe, PipeTransform, inject } from '@angular/core';
import { DatePipe } from '@angular/common';
import { DateFormatService } from './date-format.service';

/**
 * Formats a date value using the globally configured date format.
 * Impure so it re-evaluates when the user changes the format at runtime.
 *
 * Usage: {{ value | appDate }}
 *        {{ value | appDate:'—' }}   (custom fallback for null)
 */
@Pipe({ name: 'appDate', standalone: true, pure: false })
export class AppDatePipe implements PipeTransform {
  private datePipe = new DatePipe('en-US');
  private dateFormat = inject(DateFormatService);

  transform(value: string | Date | null | undefined, fallback = '—'): string {
    if (value == null) return fallback;
    return this.datePipe.transform(value, this.dateFormat.pattern()) ?? fallback;
  }
}
