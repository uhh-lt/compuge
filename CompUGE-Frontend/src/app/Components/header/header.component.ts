import {Component, OnInit} from '@angular/core';
import {MatCard} from "@angular/material/card";
import {MatToolbar, MatToolbarRow} from "@angular/material/toolbar";
import {AsyncPipe, NgClass, NgForOf} from "@angular/common";
import {MatMenu, MatMenuTrigger} from "@angular/material/menu";
import {MatIcon} from "@angular/material/icon";
import {MatMenuItem} from "@angular/material/menu";
import {RouterLink} from "@angular/router";
import {AppStateService} from "../../state_management/services/app-state.service";
import {MatIconButton} from "@angular/material/button";
import {FlexLayoutModule} from "@angular/flex-layout";

@Component({
  selector: 'app-header',
  standalone: true,
  imports: [
    MatCard,
    MatToolbarRow,
    NgClass,
    MatToolbar,
    MatMenu,
    MatIcon,
    MatMenuTrigger,
    MatMenuItem,
    RouterLink,
    NgForOf,
    AsyncPipe,
    MatIconButton,
    FlexLayoutModule
  ],
  templateUrl: './header.component.html',
  styleUrl: './header.component.css'
})
export class HeaderComponent implements OnInit{

  stateObservable = this.state.state$;


  constructor(private state: AppStateService) { }

  ngOnInit() {
  }

}
