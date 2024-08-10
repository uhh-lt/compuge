import { Injectable } from '@angular/core';
import {environment} from "../../../environments/environment";
import {BehaviorSubject, catchError, Observable} from "rxjs";
import {HttpClient} from "@angular/common/http";
import {StateModel} from "../models/state.model";
import {AuthenticationService} from "./authentication.service";

@Injectable({
  providedIn: 'root'
})
export class AppStateService {
  private readonly _apiUrl = environment.apiUrl;

  private readonly _state = new BehaviorSubject<StateModel>(
    {
      leaderboards: [],
      submissions: [],
      datasets: [],
      tasks: [],
      controlPanelSubmissions: [],
      adminSessionStatus: ''
      }
  );

  public readonly state$ = this._state.asObservable();

  constructor(public http: HttpClient,
              public authService: AuthenticationService) {
    console.log('AppStateService created');
    console.log(this._apiUrl)
    this.updateTasks();
    this.updateDatasets();
    this.updateLeaderboards();
    this.updateSubmissions();
    authService.$authStatus.subscribe(
      (status: string) => {
        this._setState({
          ...this.getState(),
          adminSessionStatus: status
        });
      }
    );
  }

  private _setState(state: StateModel): void {
    this._state.next(state);
  }

  public getState() : StateModel {
    return this._state.getValue();
  }

  public submit(
    modelName: string,
    modelLink: string,
    teamName: string,
    contactEmail: string,
    task: string,
    dataset: string,
    isPublic: boolean,
    fileContent: string
  ) : Observable<Object> {
    console.log(fileContent);
    return this.http.post(this._apiUrl + '/submission/' + task + '/' + dataset, {
      modelName: modelName,
      modelLink: modelLink,
      teamName: teamName,
      contactEmail: contactEmail,
      isPublic: isPublic,
      fileContent: fileContent
    });
  }

  public updateTasks() {
    this.http.get(this._apiUrl + '/tasks').subscribe(
      (data: any) => {
        this._setState({
          ...this.getState(),
          tasks: data
        });
      }
    );
  }

  public updateDatasets() {
      this.http.get(this._apiUrl + '/datasets').subscribe(
        (data: any) => {
          this._setState({
            ...this.getState(),
            datasets: data
          });
        }
      );
  }

  public updateLeaderboards() {
      this.http.get(this._apiUrl + '/leaderboards').subscribe(
        (data: any) => {
          this._setState({
            ...this.getState(),
            leaderboards: data
          });
        }
      );
  }

  public updateSubmissions() {
    this.http.get(this._apiUrl + '/submissions').subscribe(
      (data: any) => {
        this._setState({
          ...this.getState(),
          submissions: data
        });
      }
    );
  }

  public authenticate(password : string) {
    this.authService.login(password).subscribe(
      () => {
        this.updateControlPanel();
      }
    );
  }

  public updateControlPanel() {
    const headers = this.authService.getAuthHeaders();
    this.http.get(`${this._apiUrl}/controlPanelSubmissions`, { headers }).pipe(
      catchError(error => {
          if (error.status === 401) {
            this.authService.logout();
          }
          console.log(error);
          return error;
        }
      ))
      .subscribe(
      (data: any) => {
        this._setState({
          ...this.getState(),
          controlPanelSubmissions: data
        });
      }
    );
  }

  public forceUpdateSubmission(entry: any){
    const headers = this.authService.getAuthHeaders();
    this.http.put(`${this._apiUrl}/controlPanelSubmission/${entry.id}`, entry, { headers }).pipe(
      catchError(error => {
          if (error.status === 401) {
            this.authService.logout();
          }
          console.log(error);
          return error;
        }
      ))
      .subscribe(
        () => {
          this.updateControlPanel();
        }
      );
  }

  public deleteSubmission(id: number){
    const headers = this.authService.getAuthHeaders();
    console.log('Deleting submission with id: ' + id);
    this.http.delete(`${this._apiUrl}/controlPanelSubmission/` + id, { headers })
      .pipe(
        catchError(error => {
          if (error.status === 401) {
            this.authService.logout();
          }
          console.log(error);
          return error;
        }
      ))
      .subscribe(
      () => {
        this.updateControlPanel();
      }
    );
  }
}
