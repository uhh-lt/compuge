import {AfterViewInit, Component, Input, OnInit, ViewChild} from '@angular/core';
import {MatCard} from "@angular/material/card";
import {
  MatCell,
  MatCellDef, MatColumnDef,
  MatHeaderCell,
  MatHeaderCellDef, MatHeaderRow, MatHeaderRowDef, MatRow, MatRowDef,
  MatTable,
  MatTableDataSource
} from "@angular/material/table";
import {LeaderboardEntry} from "../../../../state_management/models/leaderboard-entry.model";
import {MatButton} from "@angular/material/button";
import {AppStateService} from "../../../../state_management/services/app-state.service";
import {map} from "rxjs";
import {DecimalPipe, NgIf} from "@angular/common";
import {MatSort, MatSortModule} from "@angular/material/sort";

@Component({
  selector: 'app-leaderboard',
  standalone: true,
  imports: [
    MatCard,
    MatTable,
    MatHeaderCell,
    MatHeaderCellDef,
    MatCell,
    MatCellDef,
    MatColumnDef,
    MatHeaderRow,
    MatHeaderRowDef,
    MatRow,
    MatRowDef,
    MatButton,
    NgIf,
    DecimalPipe,
    MatSort,
    MatSortModule
  ],
  templateUrl: './leaderboard.component.html',
  styleUrl: './leaderboard.component.css'
})
export class LeaderboardComponent implements OnInit, AfterViewInit {

  displayedColumns: string[] = [
    'index',
    'model',
    'team',
    'predictions',
    'accuracy',
    'precision',
    'recall',
    'f1_score'
  ];

  @Input()
  task : string = 'Question Identification';

  @Input()
  dataset : string = "CIFAR10";

  dataSource = new MatTableDataSource<LeaderboardEntry>();
  leaderboards = this.stateService.state$.pipe(map(state => state.leaderboards));

  @ViewChild(MatSort) sort: MatSort | undefined;

  ngAfterViewInit() {
    if (this.sort)
      this.dataSource.sort = this.sort;
  }

  constructor(private stateService: AppStateService) {

  }

  ngOnInit() {
    this.leaderboards.subscribe(
      data => {
        // choose only entries where task == this.task and dataset == this.dataset.
        // assign the data to the dataSource
        this.dataSource.data = data.filter(
          entry => (entry.task == this.task && entry.dataset == this.dataset)
        )
      }
    );
    this.stateService.updateLeaderboards();
  }

  getTextDownloadURL(prediction: string) {
    return window.URL.createObjectURL(new Blob([prediction], {type: 'text/plain'}));
  }

  refresh() {
    this.stateService.updateLeaderboards();
  }
}
