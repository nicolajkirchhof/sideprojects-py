import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { CapitalService, CapitalUpsert } from './capital.service';

describe('CapitalService', () => {
  let service: CapitalService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(CapitalService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('should GET all capital entries', () => {
    service.getAll().subscribe();
    const req = httpMock.expectOne('/api/capital');
    expect(req.request.method).toBe('GET');
    req.flush([]);
  });

  it('should GET capital by id', () => {
    service.getById(1).subscribe();
    const req = httpMock.expectOne('/api/capital/1');
    expect(req.request.method).toBe('GET');
    req.flush({});
  });

  it('should POST to create capital', () => {
    const payload: CapitalUpsert = { date: '2025-06-15', netLiquidity: 100000, maintenance: 25000, excessLiquidity: 75000, bpr: 20000 };
    service.create(payload).subscribe();
    const req = httpMock.expectOne('/api/capital');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual(payload);
    req.flush({});
  });

  it('should PUT to update capital', () => {
    const payload: CapitalUpsert = { date: '2025-06-15', netLiquidity: 110000, maintenance: 25000, excessLiquidity: 85000, bpr: 20000 };
    service.update(1, payload).subscribe();
    const req = httpMock.expectOne('/api/capital/1');
    expect(req.request.method).toBe('PUT');
    req.flush({});
  });

  it('should DELETE capital', () => {
    service.delete(1).subscribe();
    const req = httpMock.expectOne('/api/capital/1');
    expect(req.request.method).toBe('DELETE');
    req.flush(null);
  });
});
