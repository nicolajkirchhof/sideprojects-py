import { Injectable, computed, signal, effect } from '@angular/core';

export type DateFormatId = 'iso' | 'us' | 'de';

export interface DateFormatOption {
  id: DateFormatId;
  label: string;
  pattern: string;
  placeholder: string;
}

export const DATE_FORMATS: DateFormatOption[] = [
  { id: 'iso', label: 'ISO (2026-04-10)',       pattern: 'yyyy-MM-dd',  placeholder: 'YYYY-MM-DD' },
  { id: 'us',  label: 'US (04/10/2026)',        pattern: 'MM/dd/yyyy',  placeholder: 'MM/DD/YYYY' },
  { id: 'de',  label: 'German (10.04.2026)',     pattern: 'dd.MM.yyyy',  placeholder: 'DD.MM.YYYY' },
];

const STORAGE_KEY = 'tradelog.dateFormat';
const VALID_IDS = new Set<string>(DATE_FORMATS.map(f => f.id));

@Injectable({ providedIn: 'root' })
export class DateFormatService {
  readonly format = signal<DateFormatId>(this.loadInitial());

  readonly pattern = computed(() =>
    DATE_FORMATS.find(f => f.id === this.format())!.pattern
  );

  readonly placeholder = computed(() =>
    DATE_FORMATS.find(f => f.id === this.format())!.placeholder
  );

  constructor() {
    effect(() => {
      try {
        localStorage.setItem(STORAGE_KEY, this.format());
      } catch {
        /* ignore (private mode) */
      }
    });
  }

  set(id: DateFormatId): void {
    this.format.set(id);
  }

  private loadInitial(): DateFormatId {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored && VALID_IDS.has(stored)) return stored as DateFormatId;
    } catch {
      /* ignore */
    }
    return 'iso';
  }
}
