import { Injectable, inject, signal, computed } from '@angular/core';
import { HttpClient } from '@angular/common/http';

export interface LookupValue {
  id: number;
  accountId: number;
  category: string;
  name: string;
  sortOrder: number;
  isActive: boolean;
}

export type LookupCategory =
  | 'Strategy'
  | 'TypeOfTrade'
  | 'Budget'
  | 'Timeframe'
  | 'Directional'
  | 'ManagementRating';

/**
 * Caches all lookup values from the API. Call refresh() once at app
 * init; components use byCategory() to get filtered, ordered lists.
 */
@Injectable({ providedIn: 'root' })
export class LookupService {
  private http = inject(HttpClient);

  private all = signal<LookupValue[]>([]);
  private byId = computed(() => {
    const map = new Map<number, LookupValue>();
    for (const lv of this.all()) map.set(lv.id, lv);
    return map;
  });

  /** Active values for a given category, ordered by sortOrder. */
  byCategory(category: LookupCategory): LookupValue[] {
    return this.all()
      .filter(lv => lv.category === category && lv.isActive)
      .sort((a, b) => a.sortOrder - b.sortOrder);
  }

  /** All values (incl. inactive) for a category — used by the settings UI. */
  allByCategory(category: LookupCategory): LookupValue[] {
    return this.all()
      .filter(lv => lv.category === category)
      .sort((a, b) => a.sortOrder - b.sortOrder);
  }

  /** Resolve a lookup ID to its display name. Returns '' for unknown IDs. */
  name(id: number | null | undefined): string {
    if (id == null) return '';
    return this.byId().get(id)?.name ?? '';
  }

  refresh(): void {
    this.http.get<LookupValue[]>('/api/lookups').subscribe({
      next: data => this.all.set(data ?? []),
    });
  }

  create(category: LookupCategory, name: string) {
    return this.http.post<LookupValue>(`/api/lookups/${category}`, { name });
  }

  rename(id: number, name: string) {
    return this.http.put<void>(`/api/lookups/${id}`, { name });
  }

  deactivate(id: number) {
    return this.http.patch<void>(`/api/lookups/${id}/deactivate`, {});
  }

  reactivate(id: number) {
    return this.http.patch<void>(`/api/lookups/${id}/reactivate`, {});
  }

  reorder(id: number, sortOrder: number) {
    return this.http.patch<void>(`/api/lookups/${id}/reorder`, { sortOrder });
  }
}
