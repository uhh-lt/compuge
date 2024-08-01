import {Component, Input, OnChanges, OnInit, SimpleChanges} from '@angular/core';
import {MatCard, MatCardHeader} from "@angular/material/card";
import {MatTab, MatTabGroup} from "@angular/material/tabs";
import {LeaderboardComponent} from "./leaderboard/leaderboard.component";
import {AppStateService} from "../../../state_management/services/app-state.service";
import {filter, map} from "rxjs";
import {AsyncPipe, NgForOf, NgIf} from "@angular/common";

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
    NgForOf,
    NgIf
  ],
  templateUrl: './leaderboards.component.html',
  styleUrl: './leaderboards.component.css'
})
export class LeaderboardsComponent implements OnChanges {

  @Input()
  task: string = '';

  @Input()
  showTask: boolean = true;

  datasets: any;

  constructor(
    private stateService: AppStateService
  ) {
    this.updateDatasets();
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['task']) {
      this.updateDatasets();
    }
  }

  private updateDatasets() {
    this.datasets = this.stateService.state$.pipe(
      map(state =>
        state.datasets.filter(dataset =>
          dataset.task === this.task || !this.task
        )
      ),
    );
  }

}
