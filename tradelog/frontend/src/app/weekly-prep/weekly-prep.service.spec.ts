import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { WeeklyPrepService, WeeklyPrepUpsert } from './weekly-prep.service';

describe('WeeklyPrepService', () => {
  let service: WeeklyPrepService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(WeeklyPrepService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('should GET all preps without filters', () => {
    service.getAll().subscribe();
    const req = httpMock.expectOne('/api/weekly-prep');
    expect(req.request.method).toBe('GET');
    expect(req.request.params.keys().length).toBe(0);
    req.flush([]);
  });

  it('should GET preps with year filter', () => {
    service.getAll({ year: 2025 }).subscribe();
    const req = httpMock.expectOne((r) => r.url === '/api/weekly-prep');
    expect(req.request.params.get('year')).toBe('2025');
    req.flush([]);
  });

  it('should GET prep by id', () => {
    service.getById(3).subscribe();
    const req = httpMock.expectOne('/api/weekly-prep/3');
    expect(req.request.method).toBe('GET');
    req.flush({});
  });

  it('should POST to create prep', () => {
    const payload: WeeklyPrepUpsert = { date: '2025-06-16', indexBias: 'Bullish' };
    service.create(payload).subscribe();
    const req = httpMock.expectOne('/api/weekly-prep');
    expect(req.request.method).toBe('POST');
    expect(req.request.body.indexBias).toBe('Bullish');
    req.flush({});
  });

  it('should PUT to update prep', () => {
    const payload: WeeklyPrepUpsert = { date: '2025-06-16', indexBias: 'Neutral' };
    service.update(3, payload).subscribe();
    const req = httpMock.expectOne('/api/weekly-prep/3');
    expect(req.request.method).toBe('PUT');
    req.flush({});
  });

  it('should DELETE prep', () => {
    service.delete(3).subscribe();
    const req = httpMock.expectOne('/api/weekly-prep/3');
    expect(req.request.method).toBe('DELETE');
    req.flush(null);
  });
});
