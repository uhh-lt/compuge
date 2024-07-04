import {Component, OnInit} from '@angular/core';
import {MatCard} from "@angular/material/card";
import {MatToolbar, MatToolbarRow} from "@angular/material/toolbar";
import {AsyncPipe, NgClass, NgForOf} from "@angular/common";
import {MatMenu, MatMenuTrigger} from "@angular/material/menu";
import {MatIcon} from "@angular/material/icon";
import {MatMenuItem} from "@angular/material/menu";
import {RouterLink} from "@angular/router";
import {AppStateService} from "../../state_management/services/app-state.service";

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
    AsyncPipe
  ],
  templateUrl: './header.component.html',
  styleUrl: './header.component.css'
})
export class HeaderComponent implements OnInit{

  selectedTab = 'home';

  stateObservable = this.state.state$;


  constructor(private state: AppStateService) { }

  ngOnInit() {
    this.setSelectedTabFromUrl();
  }

  /**
   * Selects a tab according to the given information in the url (see Tabs object for the allocation)
   */
  setSelectedTabFromUrl() {
    const url = window.location.href;
    console.log('1' + url);

    const routeStart = url.lastIndexOf('/') + 1;
    this.selectedTab = url.substr(routeStart);

    if (this.selectedTab === '') {
      this.selectedTab = 'cam';
    }
    console.log('2' + this.selectedTab);

  }

  setSelectedTab(tab: string) {
    this.selectedTab = tab;
  }
}
