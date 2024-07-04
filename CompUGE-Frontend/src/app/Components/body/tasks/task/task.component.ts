import {Component, Input} from '@angular/core';
import {AboutComponent} from "../../about/about.component";
import {MatTab, MatTabGroup} from "@angular/material/tabs";
import {MatCard, MatCardContent} from "@angular/material/card";
import {AppStateService} from "../../../../state_management/services/app-state.service";
import {filter, map} from "rxjs";
import {DatasetComponent} from "../../datasets/dataset/dataset.component";
import {AsyncPipe, NgForOf} from "@angular/common";
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
    LeaderboardsComponent
  ],
  templateUrl: './task.component.html',
  styleUrl: './task.component.css'
})
export class TaskComponent {

  @Input() task: string = 'QI';

  // all datasets where task is equal to the task of this component
  datasets = this.state.state$.pipe(
    map(state => state.datasets.filter(dataset => dataset.task === this.task))
  );

  constructor(
    private state: AppStateService
  ) { }

}
