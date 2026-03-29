import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { PortfolioService, PortfolioUpsert } from './portfolio.service';

describe('PortfolioService', () => {
  let service: PortfolioService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(PortfolioService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('should GET all portfolio entries', () => {
    service.getAll().subscribe();
    const req = httpMock.expectOne('/api/portfolio');
    expect(req.request.method).toBe('GET');
    req.flush([]);
  });

  it('should POST to create portfolio entry', () => {
    const payload: PortfolioUpsert = { budget: 'Core', strategy: 'PositiveDrift', minAllocation: 10, maxAllocation: 30 };
    service.create(payload).subscribe();
    const req = httpMock.expectOne('/api/portfolio');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual(payload);
    req.flush({});
  });

  it('should PUT to update portfolio entry', () => {
    const payload: PortfolioUpsert = { budget: 'Core', strategy: 'PositiveDrift', minAllocation: 15, maxAllocation: 35 };
    service.update(1, payload).subscribe();
    const req = httpMock.expectOne('/api/portfolio/1');
    expect(req.request.method).toBe('PUT');
    req.flush(null);
  });

  it('should DELETE portfolio entry', () => {
    service.delete(1).subscribe();
    const req = httpMock.expectOne('/api/portfolio/1');
    expect(req.request.method).toBe('DELETE');
    req.flush(null);
  });
});
