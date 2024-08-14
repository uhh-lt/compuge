import {Component, OnInit} from '@angular/core';
import {AppStateService} from "../../../state_management/services/app-state.service";
import {map} from "rxjs";
import {AdminLoginComponent} from "./admin-login/admin-login.component";
import {AsyncPipe, DecimalPipe, NgIf} from "@angular/common";
import {MatButton} from "@angular/material/button";
import {MatCard, MatCardContent} from "@angular/material/card";
import {ControlPanelEntry} from "../../../state_management/models/control-panel-entry.model";
import {
  MatCell,
  MatCellDef,
  MatColumnDef,
  MatHeaderCell, MatHeaderCellDef,
  MatHeaderRow,
  MatHeaderRowDef,
  MatRow, MatRowDef, MatTable, MatTableDataSource
} from "@angular/material/table";
import {FormsModule} from "@angular/forms";

@Component({
  selector: 'app-control-panel',
  standalone: true,
  imports: [
    AdminLoginComponent,
    NgIf,
    AsyncPipe,
    MatButton,
    MatCard,
    MatCardContent,
    MatCell,
    MatCellDef,
    MatColumnDef,
    MatHeaderCell,
    MatHeaderRow,
    MatHeaderRowDef,
    MatRow,
    MatRowDef,
    MatTable,
    FormsModule,
    MatHeaderCellDef,
    DecimalPipe
  ],
  templateUrl: './control-panel.component.html',
  styleUrl: './control-panel.component.css'
})
export class ControlPanelComponent implements OnInit {

  authenticationState = this.stateService.state$.pipe(
    map(state => state.adminSessionStatus)
  );

  submissions = this.stateService.state$.pipe(
    map(
      state => state.controlPanelSubmissions
    )
  );

  dataSource = new MatTableDataSource<ControlPanelEntry>();

  displayedColumns: string[] = [
    'team', 'task', 'dataset', 'model', 'link', 'email',
    'status', 'time', 'is_public', 'accuracy', 'precision',
    'recall', 'f1_score', 'actions' // Including 'actions' for edit/delete/save buttons
  ];

  editingState: { [key: number]: boolean } = {};

  constructor(private stateService: AppStateService) {
  }

  ngOnInit() {
    this.submissions.subscribe(
      data => {
        this.dataSource.data = data;
      }
    )
    this.refresh();
  }

  refresh() {
    this.stateService.refreshControlPanel();
  }

  editRow(entry: ControlPanelEntry) {
    this.editingState[entry.id] = true;
  }

  saveEdit(entry: ControlPanelEntry) {
    this.editingState[entry.id] = false;
    // Modify the entry in the local data source
    const index = this.dataSource.data.findIndex(e => e.id === entry.id);
    this.dataSource.data[index] = entry;
    this.stateService.updateSubmission(entry);
  }

  deleteRow(id: number) {
    // remove the entry from the local data source
    this.dataSource.data = this.dataSource.data.filter(entry => entry.id !== id);
    this.stateService.deleteSubmission(id);
  }

  isEditing(entry: ControlPanelEntry): boolean {
    return this.editingState[entry.id] || false;
  }

  cancelEdit(entry: ControlPanelEntry) {
    this.editingState[entry.id] = false;
  }

}
