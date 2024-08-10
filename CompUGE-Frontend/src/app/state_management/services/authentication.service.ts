import {Injectable} from '@angular/core';
import {HttpClient, HttpHeaders} from "@angular/common/http";
import {BehaviorSubject, catchError, Observable, tap, throwError} from "rxjs";
import {environment} from "../../../environments/environment";

@Injectable({
  providedIn: 'root'
})
export class AuthenticationService {

  private readonly apiUrl = environment.apiUrl;
  private token: string | null = this.getTokenFromLocalStorage();
  public $authStatus = new BehaviorSubject<string>(
    this.token !== null ? 'authenticated' : ''
  );


  constructor(
    private http: HttpClient
  ) {

  }

  private getTokenFromLocalStorage(): string | null {
    try {
      return localStorage.getItem('auth_token');
    } catch (e) {
      return null;
    }
  }

  public getAuthHeaders(): HttpHeaders {
    return new HttpHeaders({
      'Content-Type': 'application/json',
      Authorization: `Bearer ${this.token}`,
    });
  }

  public login(password: string): Observable<{ access_token: string; token_type: string }> {
    // Create the request payload
    const payload = new URLSearchParams();
    payload.set('username', 'admin'); // Username is not used in this case
    payload.set('password', password);

    // Send POST request to the backend
    return this.http.post<{ access_token: string; token_type: string }>
    (
      this.apiUrl + '/token', payload.toString(),
      {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded', // Set content type for form data
        },
      }).pipe(
      tap(response => {
        this.token = response.access_token;
        localStorage.setItem('auth_token', this.token);
        this.$authStatus.next('authenticated');
        console.log(this.token);
      }),
      catchError(error => {
        if (error.status === 401) {
          this.$authStatus.next('wrong password');
        } else {
          this.$authStatus.next('server error');
        }
        // Return an observable to continue the stream
        return throwError(() => error);
      })
    );
  }


  public logout(): void {
    this.token = null;
    localStorage.removeItem('auth_token');
    this.$authStatus.next('');
  }
}
