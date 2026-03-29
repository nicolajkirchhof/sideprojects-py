import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideAnimations } from '@angular/platform-browser/animations';
import { AccountsComponent } from './accounts';

describe('AccountsComponent', () => {
  let component: AccountsComponent;
  let fixture: ComponentFixture<AccountsComponent>;
  let httpMock: HttpTestingController;

  beforeEach(async () => {
    localStorage.clear();
    await TestBed.configureTestingModule({
      imports: [AccountsComponent],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        provideAnimations(),
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(AccountsComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);

    // Flush ngOnInit: getAll + getSyncStatus
    fixture.detectChanges();
    httpMock.expectOne('/api/accounts').flush([]);
    httpMock.expectOne('/api/ibkr/sync/status').flush({
      canSync: true, lastSyncAt: null, lastSyncResult: null, cooldownRemainingSeconds: null,
    });
  });

  afterEach(() => {
    httpMock.verify();
    localStorage.clear();
  });

  it('should create the component', () => {
    expect(component).toBeTruthy();
  });

  it('should require name and ibkrAccountId', () => {
    component.onNew();
    component.form.patchValue({ name: '', ibkrAccountId: '' });
    expect(component.form.get('name')!.valid).toBe(false);
    expect(component.form.get('ibkrAccountId')!.valid).toBe(false);
  });

  it('should validate port range 1-65535', () => {
    component.onNew();
    const portCtrl = component.form.get('port')!;

    portCtrl.setValue(0);
    expect(portCtrl.valid).toBe(false);

    portCtrl.setValue(65536);
    expect(portCtrl.valid).toBe(false);

    portCtrl.setValue(7497);
    expect(portCtrl.valid).toBe(true);

    portCtrl.setValue(1);
    expect(portCtrl.valid).toBe(true);

    portCtrl.setValue(65535);
    expect(portCtrl.valid).toBe(true);
  });

  it('should validate clientId >= 0', () => {
    component.onNew();
    const ctrl = component.form.get('clientId')!;

    ctrl.setValue(-1);
    expect(ctrl.valid).toBe(false);

    ctrl.setValue(0);
    expect(ctrl.valid).toBe(true);
  });

  it('should open sidebar in create mode', () => {
    component.onNew();
    expect(component.showSidebar).toBe(true);
    expect(component.isCreating).toBe(true);
    expect(component.selected).toBeNull();
  });

  it('should open sidebar in edit mode on row select', () => {
    const account = { id: 1, ibkrAccountId: 'U111', name: 'Test', host: '127.0.0.1', port: 7497, clientId: 1, isDefault: false };
    component.onRowSelect(account);
    // Flush the getSyncStatus triggered by onRowSelect
    httpMock.expectOne('/api/ibkr/sync/status').flush({
      canSync: true, lastSyncAt: null, lastSyncResult: null, cooldownRemainingSeconds: null,
    });

    expect(component.showSidebar).toBe(true);
    expect(component.isCreating).toBe(false);
    expect(component.selected).toEqual(account);
    expect(component.form.getRawValue().name).toBe('Test');
  });

  it('should close sidebar on cancel', () => {
    component.onNew();
    component.onCancel();
    expect(component.showSidebar).toBe(false);
    expect(component.isCreating).toBe(false);
  });

  it('should return "Never" for account with no sync', () => {
    expect(component.lastSyncLabel({ lastSyncAt: null } as any)).toBe('Never');
  });

  it('should return minutes-ago label for recent sync', () => {
    const fiveMinAgo = new Date(Date.now() - 5 * 60_000).toISOString();
    expect(component.lastSyncLabel({ lastSyncAt: fiveMinAgo } as any)).toBe('5m ago');
  });

  it('should return hours-ago label', () => {
    const twoHoursAgo = new Date(Date.now() - 2 * 60 * 60_000).toISOString();
    expect(component.lastSyncLabel({ lastSyncAt: twoHoursAgo } as any)).toBe('2h ago');
  });

  it('should return days-ago label', () => {
    const threeDaysAgo = new Date(Date.now() - 3 * 24 * 60 * 60_000).toISOString();
    expect(component.lastSyncLabel({ lastSyncAt: threeDaysAgo } as any)).toBe('3d ago');
  });
});
