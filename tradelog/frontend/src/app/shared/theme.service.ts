import { Injectable, signal, effect } from '@angular/core';

const STORAGE_KEY = 'tradelog.theme';
type Theme = 'dark' | 'light';

@Injectable({ providedIn: 'root' })
export class ThemeService {
  readonly theme = signal<Theme>(this.loadInitial());

  constructor() {
    effect(() => {
      const t = this.theme();
      const html = document.documentElement;
      html.classList.toggle('light-theme', t === 'light');
      try {
        localStorage.setItem(STORAGE_KEY, t);
      } catch {
        /* ignore (private mode) */
      }
    });
  }

  toggle(): void {
    this.theme.update(t => (t === 'dark' ? 'light' : 'dark'));
  }

  isDark(): boolean {
    return this.theme() === 'dark';
  }

  private loadInitial(): Theme {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored === 'light' || stored === 'dark') return stored;
    } catch {
      /* ignore */
    }
    return 'dark';
  }
}
