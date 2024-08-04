import { Injectable } from '@angular/core';
import {environment} from "../../../environments/environment";
import {BehaviorSubject, Observable} from "rxjs";
import {HttpClient} from "@angular/common/http";
import {StateModel} from "../models/state.model";

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
      tasks: []
      }
  );

  public readonly state$ = this._state.asObservable();

  constructor(public http: HttpClient) {
    console.log('AppStateService created');
    console.log(this._apiUrl)
    this.updateTasks();
    this.updateDatasets();
    this.updateLeaderboards();
    this.updateSubmissions();
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
}
