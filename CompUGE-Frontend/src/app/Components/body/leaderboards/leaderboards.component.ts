import {Component, Input} from '@angular/core';
import {MatCard, MatCardHeader} from "@angular/material/card";
import {MatTab, MatTabGroup} from "@angular/material/tabs";
import {LeaderboardComponent} from "./leaderboard/leaderboard.component";
import {AppStateService} from "../../../state_management/services/app-state.service";
import {filter, map} from "rxjs";
import {AsyncPipe, NgForOf} from "@angular/common";

@Component({
  selector: 'app-leaderboards',
  standalone: true,
  imports: [
    MatCard,
    MatCardHeader,
    MatTabGroup,
    MatTab,
    LeaderboardComponent,
    AsyncPipe,
    NgForOf
  ],
  templateUrl: './leaderboards.component.html',
  styleUrl: './leaderboards.component.css'
})
export class LeaderboardsComponent {

  @Input()
  task: string = '';

  datasets = this.stateService.state$.pipe(
    map(state => state.datasets.filter(dataset => dataset.task === this.task || this.task === '')),
  );

  constructor(
    private stateService: AppStateService
  ) {
  }


}
