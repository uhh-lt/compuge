import {Component, OnInit} from '@angular/core';
import {MatButton} from "@angular/material/button";
import {MatCard, MatCardActions, MatCardContent} from "@angular/material/card";
import {MatFormField, MatLabel} from "@angular/material/form-field";
import {MatInput} from "@angular/material/input";
import {MatOption} from "@angular/material/autocomplete";
import {MatSelect} from "@angular/material/select";
import {AsyncPipe, NgIf} from "@angular/common";
import {FormsModule, ReactiveFormsModule} from "@angular/forms";
import {AppStateService} from "../../../state_management/services/app-state.service";
import {map} from "rxjs";

@Component({
  selector: 'app-submitting',
  standalone: true,
  imports: [
    MatButton,
    MatCard,
    MatCardActions,
    MatCardContent,
    MatFormField,
    MatInput,
    MatLabel,
    MatOption,
    MatSelect,
    NgIf,
    ReactiveFormsModule,
    FormsModule,
    AsyncPipe
  ],
  templateUrl: './submitting.component.html',
  styleUrl: './submitting.component.css'
})
export class SubmittingComponent implements OnInit{
  constructor(
    private stateService: AppStateService
  ) {}

  tasks = this.stateService.state$.pipe(map(state => state.tasks));
  datasets = this.stateService.state$.pipe(map(state => state.datasets));

  modelName = '';
  modelLink = '';
  task = '';
  dataset = '';
  fileName = '';
  file?: any;
  fileContent = '';

  ngOnInit(){
  }

  submit = () => {
    console.log('Submitting');
    console.log(this.fileContent);
    console.log('after file content');

    this.stateService.submit(this.modelName, this.modelLink, this.task, this.dataset, this.fileContent).subscribe(
(data: any) => {
        console.log('Submitted');
        console.log(data);
      }
    );
  }

  onFileSelected(event : any) {
    console.log(event);
    console.log(event.target.files[0]);
    const file: File = event.target.files[0];
    if (file) {
      this.fileName = file.name;
      this.file = file;
    }
    // read file as text
    const reader = new FileReader();
    reader.onload = (e: any) => {
      this.fileContent = e.target.result;
    };
    console.log(this.file.name);
    reader.readAsText(this.file);
  }
}
