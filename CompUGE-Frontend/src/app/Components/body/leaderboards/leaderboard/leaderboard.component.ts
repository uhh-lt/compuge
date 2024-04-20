import {Component, Input, OnInit} from '@angular/core';
import {MatCard} from "@angular/material/card";
import {
  MatCell,
  MatCellDef, MatColumnDef,
  MatHeaderCell,
  MatHeaderCellDef, MatHeaderRow, MatHeaderRowDef, MatRow, MatRowDef,
  MatTable,
  MatTableDataSource
} from "@angular/material/table";
import {LeaderboardEntry} from "../../../../StateManagement/Models/leaderboard-entry.model";
import {CommunicationService} from "../../../../StateManagement/Services/communication.service";
import {MatButton} from "@angular/material/button";

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
    MatButton
  ],
  templateUrl: './leaderboard.component.html',
  styleUrl: './leaderboard.component.css'
})
export class LeaderboardComponent implements OnInit {

  displayedColumns: string[] = [
    'model',
    'size',
    'accuracy',
    'precision',
    'recall',
    'f1',
    'overall'
  ];

  @Input()
  task : String = "QI";

  @Input()
  dataset : String = "CIFAR10";

  dataSource = new MatTableDataSource<LeaderboardEntry>();

  constructor(private communicationService: CommunicationService) {

  }

  ngOnInit() {
    this.loadData();
  }

  refresh() {
    this.loadData();
  }

  loadData(){
    this.communicationService.getLeaderboard(this.task).subscribe(
      (data: any) => {
        this.dataSource.data = data;
      }
    )
  }

}
