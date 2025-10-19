import { ComponentFixture, TestBed } from '@angular/core/testing';

import { Log } from './log.component';

describe('Log', () => {
  let component: Log;
  let fixture: ComponentFixture<Log>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [Log]
    })
    .compileComponents();

    fixture = TestBed.createComponent(Log);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
