import { Injectable, ErrorHandler } from '@angular/core';

@Injectable({
  providedIn: 'root',
})
export class GlobalErrorHandlerService implements ErrorHandler {
  handleError(error: any): void {
    // Log the error to the console or send it to a server
    console.error('An error occurred:', error);

    // Optionally, display a user-friendly message
    alert('An unexpected error occurred. Please try again later.');
  }
}
