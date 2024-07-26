import { Component } from '@angular/core';
import {MatTab, MatTabGroup} from "@angular/material/tabs";
import {MatCard, MatCardContent, MatCardHeader, MatCardTitle} from "@angular/material/card";
import {LeaderboardsComponent} from "./leaderboards/leaderboards.component";
import {AboutComponent} from "./about/about.component";

import {SubmissionsComponent} from "./submissions/submissions.component";
import {DashboardComponent} from "./dashboard/dashboard.component";
import {TasksComponent} from "./tasks/tasks.component";
import {DatasetsComponent} from "./datasets/datasets.component";
import {MatButton} from "@angular/material/button";
import {AppStateService} from "../../state_management/services/app-state.service";
import {map} from "rxjs";
import {AsyncPipe, NgForOf, NgOptimizedImage} from "@angular/common";
import {RouterLink} from "@angular/router";

@Component({
  selector: 'app-body',
  standalone: true,
  imports: [
    MatTabGroup,
    MatTab,
    MatCard,
    LeaderboardsComponent,
    AboutComponent,
    SubmissionsComponent,
    DashboardComponent,
    TasksComponent,
    DatasetsComponent,
    MatCardContent,
    MatCardTitle,
    MatCardHeader,
    MatButton,
    AsyncPipe,
    NgForOf,
    NgOptimizedImage,
    RouterLink
  ],
  templateUrl: './body.component.html',
  styleUrl: './body.component.css'
})
export class BodyComponent {

  tasks = this.stateService.state$.pipe(
    map(state => state.tasks)
  );

  constructor(
    private stateService: AppStateService
  ) { }

}
