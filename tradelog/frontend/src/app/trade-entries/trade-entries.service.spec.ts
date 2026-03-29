import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TradeEntriesService, Budget, Strategy, TypeOfTrade, TradeEntryUpsert } from './trade-entries.service';

describe('TradeEntriesService', () => {
  let service: TradeEntriesService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(TradeEntriesService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('should GET all entries without filters', () => {
    service.getAll().subscribe();
    const req = httpMock.expectOne('/api/trade-entries');
    expect(req.request.method).toBe('GET');
    expect(req.request.params.keys().length).toBe(0);
    req.flush([]);
  });

  it('should GET entries with symbol filter', () => {
    service.getAll({ symbol: 'AAPL' }).subscribe();
    const req = httpMock.expectOne((r) => r.url === '/api/trade-entries');
    expect(req.request.params.get('symbol')).toBe('AAPL');
    req.flush([]);
  });

  it('should GET entries with budget and strategy filters', () => {
    service.getAll({ budget: Budget.Core, strategy: Strategy.Momentum }).subscribe();
    const req = httpMock.expectOne((r) => r.url === '/api/trade-entries');
    expect(req.request.params.get('budget')).toBe('Core');
    expect(req.request.params.get('strategy')).toBe('Momentum');
    req.flush([]);
  });

  it('should GET entry by id', () => {
    service.getById(5).subscribe();
    const req = httpMock.expectOne('/api/trade-entries/5');
    expect(req.request.method).toBe('GET');
    req.flush({});
  });

  it('should POST to create entry', () => {
    const payload: TradeEntryUpsert = {
      symbol: 'AAPL', date: '2025-06-15', typeOfTrade: TypeOfTrade.ShortStrangle,
      budget: Budget.Core, strategy: Strategy.PositiveDrift,
      newsCatalyst: false, recentEarnings: false, sectorSupport: true, ath: false,
    };
    service.create(payload).subscribe();
    const req = httpMock.expectOne('/api/trade-entries');
    expect(req.request.method).toBe('POST');
    expect(req.request.body.symbol).toBe('AAPL');
    req.flush({});
  });

  it('should PUT to update entry', () => {
    const payload: TradeEntryUpsert = {
      symbol: 'AAPL', date: '2025-06-15', typeOfTrade: TypeOfTrade.ShortStrangle,
      budget: Budget.Core, strategy: Strategy.PositiveDrift,
      newsCatalyst: false, recentEarnings: false, sectorSupport: true, ath: false,
    };
    service.update(5, payload).subscribe();
    const req = httpMock.expectOne('/api/trade-entries/5');
    expect(req.request.method).toBe('PUT');
    req.flush({});
  });

  it('should DELETE entry', () => {
    service.delete(5).subscribe();
    const req = httpMock.expectOne('/api/trade-entries/5');
    expect(req.request.method).toBe('DELETE');
    req.flush(null);
  });
});
