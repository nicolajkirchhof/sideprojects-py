import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { InstrumentSummariesService } from './instrument-summaries.service';

describe('InstrumentSummariesService', () => {
  let service: InstrumentSummariesService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(InstrumentSummariesService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('should GET option summaries without filter', () => {
    service.getOptionSummaries().subscribe();
    const req = httpMock.expectOne('/api/instrument-summaries/options');
    expect(req.request.method).toBe('GET');
    expect(req.request.params.keys().length).toBe(0);
    req.flush([]);
  });

  it('should GET option summaries with status filter', () => {
    service.getOptionSummaries('open').subscribe();
    const req = httpMock.expectOne((r) => r.url === '/api/instrument-summaries/options');
    expect(req.request.params.get('status')).toBe('open');
    req.flush([]);
  });

  it('should GET trade summaries', () => {
    service.getTradeSummaries().subscribe();
    const req = httpMock.expectOne('/api/instrument-summaries/trades');
    expect(req.request.method).toBe('GET');
    req.flush([]);
  });
});
