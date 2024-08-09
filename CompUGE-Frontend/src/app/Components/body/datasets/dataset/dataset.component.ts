import {Component, Input} from '@angular/core';
import {MatCard, MatCardContent} from "@angular/material/card";
import {MatButton} from "@angular/material/button";
import {MatExpansionPanel, MatExpansionPanelHeader, MatExpansionPanelTitle} from "@angular/material/expansion";
import {DatasetModel} from "../../../../state_management/models/dataset.model";
import {NgIf, NgStyle} from "@angular/common";

@Component({
  selector: 'app-dataset',
  standalone: true,
  imports: [
    MatCard,
    MatCardContent,
    MatButton,
    MatExpansionPanel,
    MatExpansionPanelHeader,
    MatExpansionPanelTitle,
    NgIf,
    NgStyle
  ],
  templateUrl: './dataset.component.html',
  styleUrl: './dataset.component.css'
})
export class DatasetComponent {

  @Input()
  public datasetModel: DatasetModel = {
    task: '',
    name: '',
    description: '',
    link: '',
    paper: '',
    paper_link: '',
    train: [],
    test: [],
    val: []
  };

  getTextDownloadURL(data: string[]) {
    return window.URL.createObjectURL(new Blob(data, {type: 'text/plain'}));
  }

  downloadData(data: string[], name: string) {
    const url = this.getTextDownloadURL(data);
    const a = document.createElement('a');
    a.href = url;
    a.download = this.datasetModel.task + '-' + this.datasetModel.name + '-' + name + '.csv';
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
  }

  goToUrl(url: string) {
    window.open(url, '_blank');
  }
}
