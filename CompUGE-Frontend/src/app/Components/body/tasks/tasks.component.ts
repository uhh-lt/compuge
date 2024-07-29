import { Component } from '@angular/core';
import {TaskComponent} from "./task/task.component";
import {MatTab, MatTabGroup} from "@angular/material/tabs";
import {MatCard, MatCardContent} from "@angular/material/card";
import {AppStateService} from "../../../state_management/services/app-state.service";
import {map} from "rxjs";
import {AsyncPipe, NgForOf} from "@angular/common";
import {AboutComponent} from "../about/about.component";
import {RouterLink} from "@angular/router";
import {MatButton} from "@angular/material/button";

@Component({
  selector: 'app-tasks',
  standalone: true,
  imports: [
    TaskComponent,
    MatTab,
    MatTabGroup,
    MatCardContent,
    MatCard,
    AsyncPipe,
    AboutComponent,
    RouterLink,
    MatButton,
    NgForOf
  ],
  templateUrl: './tasks.component.html',
  styleUrl: './tasks.component.css'
})
export class TasksComponent {

  tasks = this.stateService.state$.pipe(
    map(state => state.tasks)
  );

  constructor(private stateService: AppStateService) {
  }

}
