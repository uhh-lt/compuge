import {Component, Input} from '@angular/core';
import {MatCard, MatCardContent} from "@angular/material/card";
import {MatButton} from "@angular/material/button";
import {MatExpansionPanel, MatExpansionPanelHeader, MatExpansionPanelTitle} from "@angular/material/expansion";

@Component({
  selector: 'app-dataset',
  standalone: true,
  imports: [
    MatCard,
    MatCardContent,
    MatButton,
    MatExpansionPanel,
    MatExpansionPanelHeader,
    MatExpansionPanelTitle
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

  downloadData(data: string[], name: string) {
    const url = this.getTextDownloadURL(data);
    const a = document.createElement('a');
    a.href = url;
    a.download = this.task + '-' + this.name + '-' + name + '.csv';
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
  }

  goToUrl(url: string) {
    window.open(url, '_blank');
  }
}
