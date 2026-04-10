import { Injectable, inject } from '@angular/core';
import { NativeDateAdapter } from '@angular/material/core';
import { DateFormatService } from './date-format.service';

/**
 * Custom DateAdapter for MatDatepicker that formats and parses dates
 * according to the globally configured date format.
 */
@Injectable()
export class AppDateAdapter extends NativeDateAdapter {
  private dateFormat = inject(DateFormatService);

  override format(date: Date, _displayFormat: Object): string {
    const d = date.getDate();
    const m = date.getMonth() + 1;
    const y = date.getFullYear();
    const dd = d < 10 ? `0${d}` : `${d}`;
    const mm = m < 10 ? `0${m}` : `${m}`;

    switch (this.dateFormat.format()) {
      case 'us':  return `${mm}/${dd}/${y}`;
      case 'de':  return `${dd}.${mm}.${y}`;
      default:    return `${y}-${mm}-${dd}`;
    }
  }

  override parse(value: string): Date | null {
    if (!value) return null;
    const trimmed = value.trim();

    // Try configured format first, then ISO fallback
    const fmt = this.dateFormat.format();
    let parts: number[] | null = null;

    if (fmt === 'us') {
      // MM/dd/yyyy
      const m = trimmed.match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})$/);
      if (m) parts = [+m[3], +m[1] - 1, +m[2]];
    } else if (fmt === 'de') {
      // dd.MM.yyyy
      const m = trimmed.match(/^(\d{1,2})\.(\d{1,2})\.(\d{4})$/);
      if (m) parts = [+m[3], +m[2] - 1, +m[1]];
    } else {
      // yyyy-MM-dd
      const m = trimmed.match(/^(\d{4})-(\d{1,2})-(\d{1,2})$/);
      if (m) parts = [+m[1], +m[2] - 1, +m[3]];
    }

    if (parts) {
      const d = new Date(parts[0], parts[1], parts[2]);
      if (!isNaN(d.getTime())) return d;
    }

    // ISO fallback for any format
    const iso = Date.parse(trimmed);
    return isNaN(iso) ? null : new Date(iso);
  }
}
