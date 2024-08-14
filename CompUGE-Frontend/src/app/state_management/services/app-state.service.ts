import { Injectable } from '@angular/core';
import { environment } from '../../../environments/environment';
import {BehaviorSubject, Observable, OperatorFunction, shareReplay, throwError, timer} from 'rxjs';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { StateModel } from '../models/state.model';
import { AuthenticationService } from './authentication.service';
import { catchError, retry } from 'rxjs/operators';

@Injectable({
  providedIn: 'root',
})
export class AppStateService {
  private readonly _apiUrl = environment.apiUrl;

  private readonly _state = new BehaviorSubject<StateModel>({
    leaderboards: [],
    submissions: [],
    datasets: [],
    tasks: [],
    controlPanelSubmissions: [],
    adminSessionStatus: '',
  });

  public readonly state$ = this._state.asObservable();

  constructor(public http: HttpClient, public authService: AuthenticationService) {
    console.log(this._apiUrl);
    this.refreshTasks();
    this.refreshDatasets();
    this.refreshLeaderboards();
    this.refreshSubmissions();
    authService.$authStatus.subscribe((status: string) => {
      this._setState({
        ...this.getState(),
        adminSessionStatus: status,
      });
    });
  }

  private _setState(state: StateModel): void {
    this._state.next(state);
  }

  public getState(): StateModel {
    return this._state.getValue();
  }

  private makeRequest<T>(request: Observable<T>, stateKey?: keyof StateModel, callback?: (data: any) => void) {
     let obs = request.pipe(
       this.retryStrategy(),
       shareReplay(1),
       catchError(this.handleRequestError.bind(this))
      );
     obs.subscribe(
       (data) => {
         if (callback) {
           callback(data);
         } else if (stateKey) {
           this._setState({
             ...this.getState(),
             [stateKey]: data
           });
         }
       }
     );
      return obs;
  }

  private handleRequestError(error: any): Observable<never> {
    let errorMessage = 'An unknown error occurred!';

    if (error.error instanceof ErrorEvent) {
      errorMessage = `Client Error: ${error.error.message}`;
    } else {
      switch (error.status) {
        case 400:
          errorMessage = `Bad Request: ${error.message}`;
          break;
        case 401:
          errorMessage = 'Unauthorized: Please log in again.';
          this.authService.logout();
          break;
        case 404:
          errorMessage = `Not Found: The requested resource was not found.`;
          break;
        case 500:
          errorMessage = `Internal Server Error: Please try again later.`;
          break;
        case 503:
          errorMessage = `Service Unavailable: Please try again later.`;
          break;
        default:
          errorMessage = `Error Code: ${error.status}\nMessage: ${error.message}`;
          break;
      }
    }
    // Rethrow the error so it can be handled globally
    return throwError(() => new Error(errorMessage));
  }

  // Improved retry strategy with exponential backoff and jitter
  private retryStrategy<T>() {
    return <OperatorFunction<T, T>>((source) =>
        source.pipe(
          retry({
            count: 3, // Maximum of 3 retry attempts
            delay: (error, retryCount) => {
              if (![500, 503].includes(error.status)) {
                // Do not retry for errors other than 500 and 503
                return throwError(() => error);
              }
              // Exponential backoff with jitter
              const jitter = Math.random() * 500; // Jitter value between 0-500ms
              const backoffDelay = Math.pow(2, retryCount) * 1000 + jitter; // Exponential backoff
              console.log(`Retrying request after ${backoffDelay}ms (attempt #${retryCount})`);
              return timer(backoffDelay);
            }
          })
        )
    );
  }

  // ========================
  // Public API
  // ========================
  public submit(
    modelName: string,
    modelLink: string,
    teamName: string,
    contactEmail: string,
    task: string,
    dataset: string,
    isPublic: boolean,
    fileContent: string
  ) {
    return this.makeRequest(
      this.http.post<Object>(`${this._apiUrl}/submission/${task}/${dataset}`, {
        modelName,
        modelLink,
        teamName,
        contactEmail,
        isPublic,
        fileContent,
      })
    );
  }

  public refreshTasks() {
    return this.makeRequest(this.http.get(this._apiUrl + '/tasks'), 'tasks');
  }

  public refreshDatasets() {
    return this.makeRequest(this.http.get(this._apiUrl + '/datasets'), 'datasets');
  }

  public refreshLeaderboards() {
    return this.makeRequest(this.http.get(this._apiUrl + '/leaderboards'), 'leaderboards');
  }

  public refreshSubmissions() {
    return this.makeRequest(this.http.get(this._apiUrl + '/submissions'), 'submissions');
  }

  public authenticate(password: string) {
    let obs = this.authService.login(password);
    obs.subscribe(
      next => {
        console.log('Login successful');
        this._setState({
          ...this.getState(),
          adminSessionStatus: 'authenticated',
        });
      });
    return obs;
  }

  public refreshControlPanel() {
    const headers = this.authService.getAuthHeaders();
    return this.makeRequest(
      this.http.get(`${this._apiUrl}/control-panel-submissions`, { headers }),
      'controlPanelSubmissions'
    );
  }

  public updateSubmission(entry: any) {
    const headers = this.authService.getAuthHeaders();
    return this.makeRequest(
      this.http.put(`${this._apiUrl}/submission/${entry.id}`, entry, { headers }),
      undefined,
      () => this.refreshControlPanel() // Refresh control panel after update
    );
  }

  public deleteSubmission(id: number) {
    const headers = this.authService.getAuthHeaders();
    return this.makeRequest(
      this.http.delete(`${this._apiUrl}/submission/${id}`, { headers }),
      undefined,
      () => this.refreshControlPanel() // Refresh control panel after deletion
    );
  }
}
