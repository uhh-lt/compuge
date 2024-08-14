import {Component, OnInit} from '@angular/core';
import {MatButton} from "@angular/material/button";
import {MatCard, MatCardActions, MatCardContent} from "@angular/material/card";
import {MatError, MatFormField, MatLabel} from "@angular/material/form-field";
import {MatInput} from "@angular/material/input";
import {MatOption} from "@angular/material/autocomplete";
import {MatSelect} from "@angular/material/select";
import {AsyncPipe, NgIf} from "@angular/common";
import {FormsModule, ReactiveFormsModule} from "@angular/forms";
import {AppStateService} from "../../../state_management/services/app-state.service";
import {map, Observer} from "rxjs";
import {FormGroup, FormBuilder, Validators} from "@angular/forms";
import {error} from "@angular/compiler-cli/src/transformers/util";


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
    AsyncPipe,
    MatError
  ],
  templateUrl: './submitting.component.html',
  styleUrl: './submitting.component.css'
})
export class SubmittingComponent implements OnInit {

  tasks = this.stateService.state$.pipe(map(state => state.tasks));
  datasets = this.stateService.state$.pipe(map(state => state.datasets));

  chosenFileName = '';
  message = '';

  form: FormGroup;
  fileContent = '';

  constructor(
    private fb: FormBuilder,
    private stateService: AppStateService
  ) {
    this.form = this.fb.group({
      modelName: ['', Validators.required],
      modelLink: [''],
      task: ['', Validators.required],
      dataset: ['', Validators.required],
      file: ['', Validators.required],
      teamName: [''],
      contactEmail: ['', Validators.email],
      isPublic: [false, Validators.required]
    });
  }


  ngOnInit() {
  }

  onSubmit() {
    this.message = '';

    if (this.form.invalid) {
      this.message = 'Invalid form';
      return;
    }
    this.message = 'Submitting...';
    this.stateService.submit(
      this.form.value.modelName,
      this.form.value.modelLink,
      this.form.value.teamName,
      this.form.value.contactEmail,
      this.form.value.task,
      this.form.value.dataset,
      this.form.value.isPublic,
      this.fileContent
    ).subscribe(
      {
        next: (response: any) => {
          console.log(response);
          this.message = response[0].message;
        },
        error: (error: any) => {
          switch (error.status) {
            case 400:
              this.message = 'Submission rejected';
              break;
            case 500:
              this.message = 'Internal server error';
              break;
            default:
              this.message = 'An unexpected error occurred. Please try again later.';
              break;
          }
        }
      }
    );
  }

  onFileSelected(event: any) {
    const file: File = event.target.files[0];
    if (file) {
      this.chosenFileName = file.name;
      // read file as text
      const reader = new FileReader();
      reader.onload = (e: any) => {
        this.fileContent = e.target.result;
      };
      reader.readAsText(file);
    } else {
      this.chosenFileName = 'Invalid file';
    }
  }
}
