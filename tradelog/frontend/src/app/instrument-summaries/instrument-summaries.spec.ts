import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';
import { provideZonelessChangeDetection } from '@angular/core';
import { InstrumentSummaries } from './instrument-summaries';
import { OptionInstrumentSummary } from './instrument-summaries.service';

function makeSummary(overrides: Partial<OptionInstrumentSummary> = {}): OptionInstrumentSummary {
  return {
    symbol: 'SPY', opened: '2025-06-01', closed: null, dit: 10, dte: 20,
    status: 'open', budget: 'Core', currentSetup: 'ShortStrangle', strikes: '540/560',
    pnl: 100, unrealizedPnl: 80, unrealizedPnlPct: 0.05, realizedPnl: 20, realizedPnlPct: 0.01,
    timeValue: 50, delta: -0.10, theta: 5.0, gamma: -0.02, vega: -3.0, avgIv: 0.18,
    margin: 5400, durationPct: 0.33, pnlPerDurationPct: 3.0, roic: 0.02, commissions: 2.10,
    ...overrides,
  };
}

describe('InstrumentSummaries', () => {
  let component: InstrumentSummaries;
  let fixture: ComponentFixture<InstrumentSummaries>;
  let httpMock: HttpTestingController;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [InstrumentSummaries],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([]),
        provideZonelessChangeDetection(),
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(InstrumentSummaries);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  function flushInit(): void {
    fixture.detectChanges();
    // Default filter is 'open'
    const optReq = httpMock.expectOne((r) => r.url === '/api/instrument-summaries/options');
    expect(optReq.request.params.get('status')).toBe('open');
    optReq.flush([
      makeSummary({ symbol: 'SPY', pnl: 200, delta: -0.10, theta: 5, vega: -3, gamma: -0.02, margin: 5400 }),
      makeSummary({ symbol: 'QQQ', pnl: -50, delta: 0.20, theta: 3, vega: -2, gamma: 0.01, margin: 3000 }),
    ]);
    httpMock.expectOne('/api/instrument-summaries/trades').flush([]);
  }

  it('should compute portfolio totals from option summaries', () => {
    flushInit();
    expect(component.totals.pnl).toBe(150);
    expect(component.totals.delta).toBeCloseTo(0.10);
    expect(component.totals.theta).toBe(8);
    expect(component.totals.vega).toBe(-5);
    expect(component.totals.margin).toBe(8400);
  });

  it('should reload with no status param when filter is "all"', () => {
    flushInit();
    component.onStatusFilterChange('all');

    const optReq = httpMock.expectOne((r) => r.url === '/api/instrument-summaries/options');
    expect(optReq.request.params.keys().length).toBe(0);
    optReq.flush([]);
    httpMock.expectOne('/api/instrument-summaries/trades').flush([]);
  });

  it('should filter trade summaries client-side for open status', () => {
    fixture.detectChanges();
    httpMock.expectOne((r) => r.url === '/api/instrument-summaries/options').flush([]);
    httpMock.expectOne('/api/instrument-summaries/trades').flush([
      { symbol: 'AAPL', status: 'open', totalPos: 100, avgPrice: 200, multiplier: 1, pnl: 50, unrealizedPnl: 50, realizedPnl: 0, commissions: 1 },
      { symbol: 'MSFT', status: 'closed', totalPos: 0, avgPrice: 400, multiplier: 1, pnl: 20, unrealizedPnl: 0, realizedPnl: 20, commissions: 1 },
    ]);

    // Default filter is 'open', so only AAPL (totalPos !== 0) should remain
    expect(component.tradeDataSource.data.length).toBe(1);
    expect(component.tradeDataSource.data[0].symbol).toBe('AAPL');
  });
});
