import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { AnalyticsService } from './analytics.service';

describe('AnalyticsService', () => {
  let service: AnalyticsService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(AnalyticsService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('should GET strategies', () => {
    service.getStrategies().subscribe();
    const req = httpMock.expectOne('/api/analytics/strategies');
    expect(req.request.method).toBe('GET');
    req.flush([]);
  });

  it('should GET strategy equity curve with encoded name', () => {
    service.getStrategyEquityCurve('Positive Drift').subscribe();
    const req = httpMock.expectOne('/api/analytics/strategies/Positive%20Drift/equity-curve');
    expect(req.request.method).toBe('GET');
    req.flush([]);
  });

  it('should GET budgets', () => {
    service.getBudgets().subscribe();
    const req = httpMock.expectOne('/api/analytics/budgets');
    expect(req.request.method).toBe('GET');
    req.flush([]);
  });

  it('should GET budget equity curve', () => {
    service.getBudgetEquityCurve('Core').subscribe();
    const req = httpMock.expectOne('/api/analytics/budgets/Core/equity-curve');
    expect(req.request.method).toBe('GET');
    req.flush([]);
  });

  it('should GET overall performance', () => {
    service.getOverall().subscribe();
    const req = httpMock.expectOne('/api/analytics/overall');
    expect(req.request.method).toBe('GET');
    req.flush({});
  });

  it('should GET overall equity curve', () => {
    service.getOverallEquityCurve().subscribe();
    const req = httpMock.expectOne('/api/analytics/overall/equity-curve');
    expect(req.request.method).toBe('GET');
    req.flush([]);
  });
});
