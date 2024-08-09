import {Component, Input, OnInit} from '@angular/core';
import {AppStateService} from "../../../state_management/services/app-state.service";
import {DatasetComponent} from "./dataset/dataset.component";
import {AsyncPipe, NgForOf} from "@angular/common";
import {map, Observable} from "rxjs";

@Component({
  selector: 'app-datasets',
  standalone: true,
  imports: [
    DatasetComponent,
    NgForOf,
    AsyncPipe
  ],
  templateUrl: './datasets.component.html',
  styleUrl: './datasets.component.css'
})
export class DatasetsComponent implements OnInit {

  @Input() datasets!: Observable<any[]>;

  constructor(
    private appState: AppStateService,
  ) {}

  ngOnInit(): void {
    if (!this.datasets) {
      this.datasets = this.appState.state$.pipe(
        map(state => state.datasets)
      );
    }
  }

}
