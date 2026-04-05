import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { AccountsService } from './accounts.service';

describe('AccountsService', () => {
  let service: AccountsService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    localStorage.clear();
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(AccountsService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
    localStorage.clear();
  });

  // --- Account selection ---

  it('should default selectedAccountId to 0 when localStorage is empty', () => {
    expect(service.selectedAccountId).toBe(0);
  });

  it('should persist to localStorage and reflect in selectedAccountId', () => {
    service.selectAccount(42);
    expect(localStorage.getItem('tradelog-selected-account-id')).toBe('42');
    expect(service.selectedAccountId).toBe(42);
  });

  it('should persist and emit selected account id', () => {
    const emitted: number[] = [];
    service.selectedAccountId$Obs.subscribe((id) => emitted.push(id));

    service.selectAccount(7);

    expect(service.selectedAccountId).toBe(7);
    expect(localStorage.getItem('tradelog-selected-account-id')).toBe('7');
    expect(emitted).toContain(7);
  });

  // --- CRUD ---

  it('should GET all accounts', () => {
    service.getAll().subscribe();
    const req = httpMock.expectOne('/api/accounts');
    expect(req.request.method).toBe('GET');
    req.flush([]);
  });

  it('should GET account by id', () => {
    service.getById(3).subscribe();
    const req = httpMock.expectOne('/api/accounts/3');
    expect(req.request.method).toBe('GET');
    req.flush({});
  });

  it('should POST to create account', () => {
    const payload = { name: 'Test', ibkrAccountId: 'U123' };
    service.create(payload).subscribe();
    const req = httpMock.expectOne('/api/accounts');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual(payload);
    req.flush({});
  });

  it('should PUT to update account', () => {
    const payload = { name: 'Updated' };
    service.update(5, payload).subscribe();
    const req = httpMock.expectOne('/api/accounts/5');
    expect(req.request.method).toBe('PUT');
    expect(req.request.body).toEqual(payload);
    req.flush(null);
  });

  it('should DELETE account', () => {
    service.delete(5).subscribe();
    const req = httpMock.expectOne('/api/accounts/5');
    expect(req.request.method).toBe('DELETE');
    req.flush(null);
  });

  // --- Sync ---

  it('should GET sync status', () => {
    service.getSyncStatus().subscribe();
    const req = httpMock.expectOne('/api/ibkr/sync/status');
    expect(req.request.method).toBe('GET');
    req.flush({ canSync: true });
  });

  it('should POST to trigger flex sync', () => {
    service.triggerFlexSync().subscribe();
    const req = httpMock.expectOne('/api/ibkr/flex-sync');
    expect(req.request.method).toBe('POST');
    req.flush({});
  });

  it('should POST to trigger live sync', () => {
    service.triggerLiveSync().subscribe();
    const req = httpMock.expectOne('/api/ibkr/live-sync');
    expect(req.request.method).toBe('POST');
    req.flush({});
  });
});
