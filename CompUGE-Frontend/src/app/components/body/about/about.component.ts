import {Component, Input, OnInit} from '@angular/core';
import {MatCard, MatCardContent} from "@angular/material/card";
import {NgIf} from "@angular/common";

@Component({
  selector: 'app-about',
  standalone: true,
  imports: [
    MatCard,
    MatCardContent,
    NgIf
  ],
  templateUrl: './about.component.html',
  styleUrl: './about.component.css'
})
export class AboutComponent implements OnInit{

  @Input()
  whatAbout = 'general';

  constructor() { }

  ngOnInit() {
    if (this.whatAbout === undefined) {
      this.whatAbout = 'general';
    }
  }


}
