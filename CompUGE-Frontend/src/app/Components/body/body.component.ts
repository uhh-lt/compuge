import { Component } from '@angular/core';
import {MatTab, MatTabGroup} from "@angular/material/tabs";
import {MatCard} from "@angular/material/card";
import {LeaderboardsComponent} from "./leaderboards/leaderboards.component";
import {AboutComponent} from "./about/about.component";
import {ContactComponent} from "./contact/contact.component";
import {SubmissionsComponent} from "./submissions/submissions.component";

@Component({
  selector: 'app-body',
  standalone: true,
  imports: [
    MatTabGroup,
    MatTab,
    MatCard,
    LeaderboardsComponent,
    AboutComponent,
    ContactComponent,
    SubmissionsComponent
  ],
  templateUrl: './body.component.html',
  styleUrl: './body.component.css'
})
export class BodyComponent {

}
