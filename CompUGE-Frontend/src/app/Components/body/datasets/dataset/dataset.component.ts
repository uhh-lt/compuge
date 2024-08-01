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
  public task: string = 'Placeholder';
  @Input()
  public name: string = 'Placeholder';
  @Input()
  public description: string = 'Placeholder';
  @Input()
  public link: string = 'Placeholder';
  @Input()
  public paper: string = 'Placeholder';
  @Input()
  public paper_link: string = 'Placeholder';
  @Input()
  public train: string[] = [];
  @Input()
  public test: string[] = [];

  getTextDownloadURL(data: string[]) {
    return window.URL.createObjectURL(new Blob(data, {type: 'text/plain'}));
  }

}
