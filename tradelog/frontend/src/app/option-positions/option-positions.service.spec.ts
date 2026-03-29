import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { OptionPositionsService, OptionPositionUpsert, PositionRight, OptionPositionsLogService, OptionPositionsLog } from './option-positions.service';

describe('OptionPositionsService', () => {
  let service: OptionPositionsService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(OptionPositionsService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('should GET all positions without filters', () => {
    service.getAll().subscribe();
    const req = httpMock.expectOne('/api/option-positions');
    expect(req.request.method).toBe('GET');
    expect(req.request.params.keys().length).toBe(0);
    req.flush([]);
  });

  it('should GET positions with symbol filter', () => {
    service.getAll({ symbol: 'SPY' }).subscribe();
    const req = httpMock.expectOne((r) => r.url === '/api/option-positions');
    expect(req.request.params.get('symbol')).toBe('SPY');
    req.flush([]);
  });

  it('should GET positions with status filter', () => {
    service.getAll({ status: 'open' }).subscribe();
    const req = httpMock.expectOne((r) => r.url === '/api/option-positions');
    expect(req.request.params.get('status')).toBe('open');
    req.flush([]);
  });

  it('should GET position by id', () => {
    service.getById(10).subscribe();
    const req = httpMock.expectOne('/api/option-positions/10');
    expect(req.request.method).toBe('GET');
    req.flush({});
  });

  it('should POST to create position', () => {
    const payload: OptionPositionUpsert = {
      symbol: 'SPY', contractId: 'SPY250620P00540000', opened: '2025-06-01',
      expiry: '2025-06-20', pos: -1, right: PositionRight.Put, strike: 540,
      cost: 3.50, commission: 1.05, multiplier: 100,
    };
    service.create(payload).subscribe();
    const req = httpMock.expectOne('/api/option-positions');
    expect(req.request.method).toBe('POST');
    expect(req.request.body.symbol).toBe('SPY');
    req.flush({});
  });

  it('should PUT to update position', () => {
    const payload: OptionPositionUpsert = {
      symbol: 'SPY', contractId: 'SPY250620P00540000', opened: '2025-06-01',
      expiry: '2025-06-20', pos: -1, right: PositionRight.Put, strike: 540,
      cost: 3.50, commission: 1.05, multiplier: 100,
    };
    service.update(10, payload).subscribe();
    const req = httpMock.expectOne('/api/option-positions/10');
    expect(req.request.method).toBe('PUT');
    req.flush({});
  });

  it('should DELETE position', () => {
    service.delete(10).subscribe();
    const req = httpMock.expectOne('/api/option-positions/10');
    expect(req.request.method).toBe('DELETE');
    req.flush(null);
  });
});

describe('OptionPositionsLogService', () => {
  let service: OptionPositionsLogService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(OptionPositionsLogService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('should GET logs by contractId', () => {
    service.getByContract('SPY250620P00540000').subscribe();
    const req = httpMock.expectOne((r) => r.url === '/api/option-positions-log');
    expect(req.request.params.get('contractId')).toBe('SPY250620P00540000');
    req.flush([]);
  });

  it('should GET latest logs', () => {
    service.getLatest().subscribe();
    const req = httpMock.expectOne('/api/option-positions-log/latest');
    expect(req.request.method).toBe('GET');
    req.flush([]);
  });

  it('should POST bulk insert', () => {
    const entries: OptionPositionsLog[] = [{
      dateTime: '2025-06-15T16:00:00Z', contractId: 'SPY250620P00540000',
      underlying: 542.5, iv: 0.18, price: 3.20, timeValue: 1.10,
      delta: -0.35, theta: -0.08, gamma: 0.02, vega: 0.15, margin: 5400,
    }];
    service.bulkInsert(entries).subscribe();
    const req = httpMock.expectOne('/api/option-positions-log/bulk');
    expect(req.request.method).toBe('POST');
    expect(req.request.body.length).toBe(1);
    req.flush({ inserted: 1, skipped: 0 });
  });

  it('should GET last sync', () => {
    service.getLastSync().subscribe();
    const req = httpMock.expectOne('/api/option-positions-log/last-sync');
    expect(req.request.method).toBe('GET');
    req.flush({ lastSync: '2025-06-15T16:00:00Z' });
  });
});
