import { Component } from '@angular/core';
import {MatCard} from "@angular/material/card";

@Component({
  selector: 'app-header',
  standalone: true,
  imports: [
    MatCard
  ],
  templateUrl: './header.component.html',
  styleUrl: './header.component.css'
})
export class HeaderComponent {

}
