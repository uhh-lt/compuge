import {Component, Input, OnInit} from '@angular/core';
import {MatCard, MatCardActions, MatCardContent} from "@angular/material/card";
import {MatFormField, MatLabel} from "@angular/material/form-field";
import {MatInput} from "@angular/material/input";
import {MatButton} from "@angular/material/button";
import {NgIf} from "@angular/common";
import {MatOption} from "@angular/material/autocomplete";
import {MatSelect} from "@angular/material/select";
import {FormsModule} from "@angular/forms";
import {AppStateService} from "../../../state_management/services/app-state.service";
import {map} from "rxjs";
import {
  MatCell,
  MatCellDef,
  MatColumnDef,
  MatHeaderCell, MatHeaderCellDef,
  MatHeaderRow,
  MatHeaderRowDef,
  MatRow, MatRowDef, MatTable, MatTableDataSource
} from "@angular/material/table";
import {LeaderboardEntry} from "../../../state_management/models/leaderboard-entry.model";
import {SubmissionEntry} from "../../../state_management/models/submission-entry.model";
import {MatTab, MatTabGroup} from "@angular/material/tabs";

@Component({
  selector: 'app-submissions',
  standalone: true,
  imports: [
    MatCard,
    MatCardContent,
    MatFormField,
    MatInput,
    MatButton,
    MatCardActions,
    MatLabel,
    NgIf,
    MatOption,
    MatSelect,
    FormsModule,
    MatCell,
    MatCellDef,
    MatColumnDef,
    MatHeaderCell,
    MatHeaderRow,
    MatHeaderRowDef,
    MatRow,
    MatRowDef,
    MatTable,
    MatHeaderCellDef,
    MatTabGroup,
    MatTab
  ],
  templateUrl: './submissions.component.html',
  styleUrl: './submissions.component.css'
})
export class SubmissionsComponent implements OnInit {

  displayedColumns: string[] = [
    'team',
    'model',
    'task',
    'dataset',
    'status',
    'predictions',
    'time',
  ];

  @Input()
  task: string = '';

  constructor(private state: AppStateService) {
  }

  submissions = this.state.state$.pipe(
    map(
      state =>
        state.submissions.filter(
          submission =>
            (submission.task == this.task || this.task == undefined)))
  );

  dataSource = new MatTableDataSource<SubmissionEntry>();

  ngOnInit() {
    this.submissions.subscribe(
      data => {
        this.dataSource.data = data;
      }
    )
  }

  getTextDownloadURL(prediction: string) {
    return window.URL.createObjectURL(new Blob([prediction], {type: 'text/plain'}));
  }

  refresh() {
    this.state.updateSubmissions();
  }
}
