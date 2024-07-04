import {Component, Input} from '@angular/core';
import {MatCard, MatCardContent} from "@angular/material/card";

@Component({
  selector: 'app-dataset',
  standalone: true,
  imports: [
    MatCard,
    MatCardContent
  ],
  templateUrl: './dataset.component.html',
  styleUrl: './dataset.component.css'
})
export class DatasetComponent {

  @Input()
  public name: string = 'Placeholder';
  @Input()
  public description: string = 'Placeholder';
  @Input()
  public link: string = 'Placeholder';
  @Input()
  public train: any[] = [];
  @Input()
  public test: any[] = [];

}
