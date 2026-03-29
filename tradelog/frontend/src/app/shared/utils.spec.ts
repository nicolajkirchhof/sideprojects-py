import { toIsoOrNull, pnlColor } from './utils';

describe('toIsoOrNull', () => {
  it('should return ISO string for a Date object', () => {
    const date = new Date('2025-06-15T12:00:00Z');
    expect(toIsoOrNull(date)).toBe(date.toISOString());
  });

  it('should return ISO string for a valid date string', () => {
    const result = toIsoOrNull('2025-06-15');
    expect(result).toContain('2025-06-15');
  });

  it('should return null for null', () => {
    expect(toIsoOrNull(null)).toBeNull();
  });

  it('should return null for undefined', () => {
    expect(toIsoOrNull(undefined)).toBeNull();
  });

  it('should return null for empty string', () => {
    expect(toIsoOrNull('')).toBeNull();
  });

  it('should return null for an invalid date string', () => {
    expect(toIsoOrNull('not-a-date')).toBeNull();
  });
});

describe('pnlColor', () => {
  it('should return green class for positive values', () => {
    expect(pnlColor(100)).toBe('text-green-400');
  });

  it('should return red class for negative values', () => {
    expect(pnlColor(-50)).toBe('text-red-400');
  });

  it('should return empty string for zero', () => {
    expect(pnlColor(0)).toBe('');
  });

  it('should return empty string for null', () => {
    expect(pnlColor(null)).toBe('');
  });

  it('should return empty string for undefined', () => {
    expect(pnlColor(undefined)).toBe('');
  });
});
