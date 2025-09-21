import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TradeIdeas } from './trade-ideas';

describe('TradeIdeas', () => {
  let component: TradeIdeas;
  let fixture: ComponentFixture<TradeIdeas>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TradeIdeas]
    })
    .compileComponents();

    fixture = TestBed.createComponent(TradeIdeas);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
