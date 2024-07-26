import { Component } from '@angular/core';
import {TaskComponent} from "./task/task.component";
import {MatTab, MatTabGroup} from "@angular/material/tabs";
import {MatCard, MatCardContent} from "@angular/material/card";

@Component({
  selector: 'app-tasks',
  standalone: true,
  imports: [
    TaskComponent,
    MatTab,
    MatTabGroup,
    MatCardContent,
    MatCard
  ],
  templateUrl: './tasks.component.html',
  styleUrl: './tasks.component.css'
})
export class TasksComponent {



}
