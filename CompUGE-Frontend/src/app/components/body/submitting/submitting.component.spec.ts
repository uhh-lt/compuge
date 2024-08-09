import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SubmittingComponent } from './submitting.component';

describe('SubmittingComponent', () => {
  let component: SubmittingComponent;
  let fixture: ComponentFixture<SubmittingComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SubmittingComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(SubmittingComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
