import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { Trades } from './trades';

describe('Trades', () => {
  let component: Trades;
  let fixture: ComponentFixture<Trades>;
  let httpMock: HttpTestingController;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [Trades],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        provideZonelessChangeDetection(),
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(Trades);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);

    // Constructor already triggers load — flush the pending request
    httpMock.match('/api/trades').forEach((r) => r.flush([]));
  });

  afterEach(() => {
    httpMock.match(() => true).forEach((r) => r.flush([]));
  });

  it('should create the component', () => {
    expect(component).toBeTruthy();
    expect(component.form).toBeTruthy();
  });

  it('should require symbol, date, typeOfTrade, budget, strategy', () => {
    component.onNew();

    const form = component.form;
    form.patchValue({ symbol: '', date: null, typeOfTrade: null, budget: null, strategy: null });

    expect(form.get('symbol')!.valid).toBe(false);
    expect(form.get('date')!.valid).toBe(false);
    expect(form.get('typeOfTrade')!.valid).toBe(false);
    expect(form.get('budget')!.valid).toBe(false);
    expect(form.get('strategy')!.valid).toBe(false);
  });

  it('should be valid when required fields are filled', () => {
    component.onNew();

    component.form.patchValue({
      symbol: 'SPY',
      date: new Date(),
      typeOfTrade: 'ShortStrangle',
      budget: 'Core',
      strategy: 'PositiveDrift',
    });

    expect(component.form.valid).toBe(true);
  });

  it('should open sidebar in create mode with defaults', () => {
    component.onNew();

    expect(component.showSidebar()).toBe(true);
    expect(component.isCreating()).toBe(true);
    expect(component.selected()).toBeNull();
    expect(component.form.get('timeframe')!.value).toBe('OneDay');
  });

  it('should open sidebar in edit mode on row select', () => {
    const entry = {
      id: 1, symbol: 'AAPL', date: '2025-06-15', typeOfTrade: 'LongCall' as any,
      budget: 'Core' as any, strategy: 'Momentum' as any,
      newsCatalyst: true, recentEarnings: false, sectorSupport: false, ath: false,
    };
    component.onRowSelect(entry as any);

    expect(component.showSidebar()).toBe(true);
    expect(component.isCreating()).toBe(false);
    expect(component.selected()).toEqual(entry as any);
    expect(component.form.getRawValue().symbol).toBe('AAPL');
  });

  it('should not save when form is invalid', () => {
    component.onNew();
    component.form.patchValue({ symbol: '' });

    component.onSave();

    const postPut = httpMock.match((r) => r.method === 'POST' || r.method === 'PUT');
    expect(postPut.length).toBe(0);
  });

  it('should close sidebar on cancel', () => {
    component.onNew();
    component.onCancel();

    expect(component.showSidebar()).toBe(false);
    expect(component.isCreating()).toBe(false);
    expect(component.selected()).toBeNull();
  });
});
