import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { App } from './app';

describe('App', () => {
  let httpMock: HttpTestingController;

  beforeEach(async () => {
    localStorage.clear();
    vi.useFakeTimers();

    await TestBed.configureTestingModule({
      imports: [App],
      providers: [
        provideRouter([]),
        provideHttpClient(),
        provideHttpClientTesting(),
        provideZonelessChangeDetection(),
      ],
    }).compileComponents();
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.match(() => true).forEach((r) => r.flush({}));
    vi.useRealTimers();
    localStorage.clear();
  });

  it('should create the app', () => {
    const fixture = TestBed.createComponent(App);
    expect(fixture.componentInstance).toBeTruthy();
  });

  it('should emit "Never synced" when lastSync is null', async () => {
    const fixture = TestBed.createComponent(App);
    fixture.detectChanges();
    vi.advanceTimersByTime(0);

    // Flush AccountSwitcher's getAll
    httpMock.match('/api/accounts').forEach((r) => r.flush([]));
    // Flush the last-sync request
    httpMock.expectOne('/api/option-positions-log/last-sync').flush({ lastSync: null });

    const result = fixture.componentInstance.lastSyncInfo();

    expect(result.label).toBe('Never synced');
    expect(result.stale).toBe(true);
  });

  it('should emit minutes-ago label for recent sync', async () => {
    const fixture = TestBed.createComponent(App);
    fixture.detectChanges();
    vi.advanceTimersByTime(0);

    httpMock.match('/api/accounts').forEach((r) => r.flush([]));
    const fiveMinAgo = new Date(Date.now() - 5 * 60_000).toISOString();
    httpMock.expectOne('/api/option-positions-log/last-sync').flush({ lastSync: fiveMinAgo });

    const result = fixture.componentInstance.lastSyncInfo();

    expect(result.label).toBe('Synced 5m ago');
    expect(result.stale).toBe(false);
  });

  it('should mark sync as stale after 24h', async () => {
    const fixture = TestBed.createComponent(App);
    fixture.detectChanges();
    vi.advanceTimersByTime(0);

    httpMock.match('/api/accounts').forEach((r) => r.flush([]));
    const twoDaysAgo = new Date(Date.now() - 2 * 24 * 60 * 60_000).toISOString();
    httpMock.expectOne('/api/option-positions-log/last-sync').flush({ lastSync: twoDaysAgo });

    const result = fixture.componentInstance.lastSyncInfo();

    expect(result.label).toBe('Synced 2d ago');
    expect(result.stale).toBe(true);
  });
});
