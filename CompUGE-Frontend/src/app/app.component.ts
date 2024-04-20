import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import {HeaderComponent} from "./Components/header/header.component";
import {CommunicationService} from "./StateManagement/Services/communication.service";

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, HeaderComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {

  constructor(
    private communicationService: CommunicationService
  ) {
  }

  title = 'compuge-frontend';
}
