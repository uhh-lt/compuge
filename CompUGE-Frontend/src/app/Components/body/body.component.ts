import { Component } from '@angular/core';
import {MatTab, MatTabGroup} from "@angular/material/tabs";
import {MatCard} from "@angular/material/card";
import {LeaderboardsComponent} from "./leaderboards/leaderboards.component";
import {AboutComponent} from "./about/about.component";

import {SubmissionsComponent} from "./submissions/submissions.component";
import {DashboardComponent} from "./dashboard/dashboard.component";
import {TasksComponent} from "./tasks/tasks.component";
import {DatasetsComponent} from "./datasets/datasets.component";

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
    DatasetsComponent
  ],
  templateUrl: './body.component.html',
  styleUrl: './body.component.css'
})
export class BodyComponent {

}
