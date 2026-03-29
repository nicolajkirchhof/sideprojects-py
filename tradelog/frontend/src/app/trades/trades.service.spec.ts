import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TradesService, TradeUpsert } from './trades.service';

describe('TradesService', () => {
  let service: TradesService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(TradesService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('should GET all trades without filters', () => {
    service.getAll().subscribe();
    const req = httpMock.expectOne('/api/trades');
    expect(req.request.method).toBe('GET');
    expect(req.request.params.keys().length).toBe(0);
    req.flush([]);
  });

  it('should GET trades with symbol filter', () => {
    service.getAll({ symbol: 'MSFT' }).subscribe();
    const req = httpMock.expectOne((r) => r.url === '/api/trades');
    expect(req.request.params.get('symbol')).toBe('MSFT');
    req.flush([]);
  });

  it('should GET trade by id', () => {
    service.getById(2).subscribe();
    const req = httpMock.expectOne('/api/trades/2');
    expect(req.request.method).toBe('GET');
    req.flush({});
  });

  it('should POST to create trade', () => {
    const payload: TradeUpsert = { symbol: 'MSFT', date: '2025-06-15', posChange: 100, price: 450.25, commission: 1.00, multiplier: 1 };
    service.create(payload).subscribe();
    const req = httpMock.expectOne('/api/trades');
    expect(req.request.method).toBe('POST');
    expect(req.request.body.symbol).toBe('MSFT');
    req.flush({});
  });

  it('should PUT to update trade', () => {
    const payload: TradeUpsert = { symbol: 'MSFT', date: '2025-06-15', posChange: 100, price: 451.00, commission: 1.00, multiplier: 1 };
    service.update(2, payload).subscribe();
    const req = httpMock.expectOne('/api/trades/2');
    expect(req.request.method).toBe('PUT');
    req.flush({});
  });

  it('should DELETE trade', () => {
    service.delete(2).subscribe();
    const req = httpMock.expectOne('/api/trades/2');
    expect(req.request.method).toBe('DELETE');
    req.flush(null);
  });
});
