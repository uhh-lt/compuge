import { Component } from '@angular/core';
import {MatCard, MatCardHeader} from "@angular/material/card";
import {MatTab, MatTabGroup} from "@angular/material/tabs";
import {LeaderboardComponent} from "./leaderboard/leaderboard.component";

@Component({
  selector: 'app-leaderboards',
  standalone: true,
  imports: [
    MatCard,
    MatCardHeader,
    MatTabGroup,
    MatTab,
    LeaderboardComponent
  ],
  templateUrl: './leaderboards.component.html',
  styleUrl: './leaderboards.component.css'
})
export class LeaderboardsComponent {





}
