import {Component} from '@angular/core';
import {MatCard, MatCardActions, MatCardContent} from "@angular/material/card";
import {MatFormField, MatLabel} from "@angular/material/form-field";
import {MatInput} from "@angular/material/input";
import {MatButton} from "@angular/material/button";
import {NgIf} from "@angular/common";
import {MatOption} from "@angular/material/autocomplete";
import {MatSelect} from "@angular/material/select";
import {FormsModule} from "@angular/forms";
import {CommunicationService} from "../../../StateManagement/Services/communication.service";

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
    FormsModule
  ],
  templateUrl: './submissions.component.html',
  styleUrl: './submissions.component.css'
})
export class SubmissionsComponent {

  constructor(
    private communicationService: CommunicationService
  ) {}

  tasks = [
    "QI",
    "OAI",
    "SC",
    "SG"
    ];

  modelName = '';
  modelLink = '';
  task = '';
  fileName = '';
  file?: any;
  fileContent = '';

  submit = () => {
    // read file as text
    const reader = new FileReader();
    reader.onload = (e: any) => {
      this.fileContent = e.target.result;
    };
    reader.readAsText(this.file);
    this.communicationService.submit(this.modelName, this.modelLink, this.task, this.fileContent).subscribe(
      (data: any) => {
        console.log(data);
      },
      (error: any) => {
        console.log(error);
      }
    );
  }

  onFileSelected(event : any) {
    const file: File = event.target.files[0];
    if (file) {
      this.file = file;
    }
  }
}
