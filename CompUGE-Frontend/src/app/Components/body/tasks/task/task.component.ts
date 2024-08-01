import {Component, Input, OnChanges, SimpleChanges} from '@angular/core';
import {AboutComponent} from "../../about/about.component";
import {MatTab, MatTabGroup} from "@angular/material/tabs";
import {MatCard, MatCardContent} from "@angular/material/card";
import {AppStateService} from "../../../../state_management/services/app-state.service";
import {filter, map} from "rxjs";
import {DatasetComponent} from "../../datasets/dataset/dataset.component";
import {AsyncPipe, NgForOf, NgIf} from "@angular/common";
import {SubmissionsComponent} from "../../submissions/submissions.component";
import {LeaderboardComponent} from "../../leaderboards/leaderboard/leaderboard.component";
import {LeaderboardsComponent} from "../../leaderboards/leaderboards.component";

@Component({
  selector: 'app-task',
  standalone: true,
  imports: [
    AboutComponent,
    MatTabGroup,
    MatTab,
    MatCard,
    MatCardContent,
    DatasetComponent,
    NgForOf,
    AsyncPipe,
    SubmissionsComponent,
    LeaderboardComponent,
    LeaderboardsComponent,
    NgIf
  ],
  templateUrl: './task.component.html',
  styleUrl: './task.component.css'
})
export class TaskComponent implements OnChanges {

  @Input() task: string = '';

  datasets: any;

  constructor(
    private state: AppStateService
  ) {
    this.updateDatasets();
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['task']) {
      this.updateDatasets();
    }
  }

  private updateDatasets() {
    this.datasets = this.state.state$.pipe(
      map(state => state.datasets.filter(dataset => dataset.task === this.task))
    );
  }
}
