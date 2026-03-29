import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideRouter, Router } from '@angular/router';
import { AccountSwitcherComponent } from './account-switcher';
import { AccountsService, Account } from '../../accounts/accounts.service';
import { provideAnimations } from '@angular/platform-browser/animations';

const MOCK_ACCOUNTS: Account[] = [
  { id: 1, ibkrAccountId: 'U111', name: 'Main', host: '127.0.0.1', port: 7497, clientId: 1, isDefault: false },
  { id: 2, ibkrAccountId: 'U222', name: 'Paper', host: '127.0.0.1', port: 7497, clientId: 2, isDefault: true },
];

describe('AccountSwitcherComponent', () => {
  let component: AccountSwitcherComponent;
  let fixture: ComponentFixture<AccountSwitcherComponent>;
  let httpMock: HttpTestingController;
  let accountsService: AccountsService;

  beforeEach(async () => {
    localStorage.clear();
    await TestBed.configureTestingModule({
      imports: [AccountSwitcherComponent],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([]),
        provideAnimations(),
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(AccountSwitcherComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
    accountsService = TestBed.inject(AccountsService);
  });

  afterEach(() => {
    httpMock.match(() => true).forEach((r) => r.flush([]));
    httpMock.verify();
    localStorage.clear();
  });

  it('should auto-select default account when none is selected', () => {
    vi.spyOn(accountsService, 'selectAccount');
    const router = TestBed.inject(Router);
    vi.spyOn(router, 'navigateByUrl').mockResolvedValue(true);

    fixture.detectChanges();
    httpMock.expectOne('/api/accounts').flush(MOCK_ACCOUNTS);

    expect(component.accounts.length).toBe(2);
    expect(component.selectedId).toBe(2);
    expect(accountsService.selectAccount).toHaveBeenCalledWith(2);
  });

  it('should select first account if no default exists', () => {
    const router = TestBed.inject(Router);
    vi.spyOn(router, 'navigateByUrl').mockResolvedValue(true);

    const noDefault = MOCK_ACCOUNTS.map((a) => ({ ...a, isDefault: false }));
    fixture.detectChanges();
    httpMock.expectOne('/api/accounts').flush(noDefault);

    expect(component.selectedId).toBe(1);
  });

  it('should update service and reload route on change', () => {
    vi.spyOn(accountsService, 'selectAccount');
    const router = TestBed.inject(Router);
    vi.spyOn(router, 'navigateByUrl').mockResolvedValue(true);

    component.onChange(3);

    expect(component.selectedId).toBe(3);
    expect(accountsService.selectAccount).toHaveBeenCalledWith(3);
    expect(router.navigateByUrl).toHaveBeenCalled();
  });
});
