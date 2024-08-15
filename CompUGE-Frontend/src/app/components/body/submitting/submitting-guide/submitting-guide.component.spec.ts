import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SubmittingGuideComponent } from './submitting-guide.component';

describe('SubmittingGuideComponent', () => {
  let component: SubmittingGuideComponent;
  let fixture: ComponentFixture<SubmittingGuideComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SubmittingGuideComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(SubmittingGuideComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
