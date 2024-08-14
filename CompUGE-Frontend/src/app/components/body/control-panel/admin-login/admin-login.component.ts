import { Component } from '@angular/core';
import {MatCard} from "@angular/material/card";
import {FormsModule} from "@angular/forms";
import {MatFormField, MatLabel} from "@angular/material/form-field";
import {MatButton} from "@angular/material/button";
import {MatInput} from "@angular/material/input";
import {AppStateService} from "../../../../state_management/services/app-state.service";
import {map} from "rxjs";
import {AsyncPipe, NgIf} from "@angular/common";
import {Router, RouterModule} from "@angular/router";

@Component({
  selector: 'app-admin-login',
  standalone: true,
  imports: [
    MatCard,
    FormsModule,
    MatFormField,
    MatButton,
    MatInput,
    MatLabel,
    NgIf,
    AsyncPipe,
    RouterModule
  ],
  templateUrl: './admin-login.component.html',
  styleUrl: './admin-login.component.css'
})
export class AdminLoginComponent {
  password: string = '';

  constructor(private stateService: AppStateService,
              private router: Router
  ) {}

  authenticationState = this.stateService.state$.pipe(
    map(state => state.adminSessionStatus)
  );

  onSubmit() {
    this.stateService.authenticate(this.password).subscribe(
      next => {
        console.log('Authentication successful');
        this.router.navigate(['/control-panel']);
      }
    );
  }
}
