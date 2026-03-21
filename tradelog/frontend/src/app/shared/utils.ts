export function toIsoOrNull(d: any): string | null {
  if (!d) return null;
  const date = d instanceof Date ? d : new Date(d);
  return isNaN(date.getTime()) ? null : date.toISOString();
}

export function pnlColor(value: number | null | undefined): string {
  if (value == null || value === 0) return '';
  return value > 0 ? 'text-green-400' : 'text-red-400';
}
