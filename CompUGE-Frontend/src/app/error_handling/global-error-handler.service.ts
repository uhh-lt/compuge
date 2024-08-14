import { Injectable, ErrorHandler, Injector } from '@angular/core';
import { HttpErrorResponse } from '@angular/common/http';
import { Router } from '@angular/router';

@Injectable({
  providedIn: 'root',
})
export class GlobalErrorHandlerService implements ErrorHandler {

  constructor(private injector: Injector) {}

  handleError(error: any): void {
    // Log the error to the console
    console.error('An error occurred:', error);

    // Differentiate between error types
    if (error instanceof HttpErrorResponse) {
      // Handle HTTP errors
      this.handleHttpError(error);
    } else if (error instanceof TypeError) {
      // Handle client-side errors (TypeErrors)
      this.handleClientError(error);
    } else if (error instanceof Error) {
      // Handle generic JavaScript errors
      this.handleGenericError(error);
    } else {
      // Handle any other type of error
      this.handleUnknownError(error);
    }

    // Always log the error, could be to a server or external logging service
    this.logError(error);
  }

  private handleHttpError(error: HttpErrorResponse): void {
    // Customize the message for different status codes
    let errorMessage = 'An unexpected error occurred. Please try again later.';
    switch (error.status) {
      case 400:
        errorMessage = 'Bad Request: Please check your input.';
        break;
      case 401:
        errorMessage = 'Unauthorized: Please log in again.';
        break;
      case 404:
        errorMessage = 'Not Found: The requested resource was not found.';
        break;
      case 500:
        errorMessage = 'Internal Server Error: Please try again later.';
        break;
      case 503:
        errorMessage = 'Service Unavailable: Please try again later.';
        break;
      default:
        errorMessage = `Unexpected Error: ${error.message}`;
        break;
    }
    // Optionally, display the error to the user
    alert(errorMessage);
  }

  private handleClientError(error: TypeError): void {
    // Handle client-side errors
    alert('A client-side error occurred. Please try again.');
  }

  private handleGenericError(error: Error): void {
    // Handle generic JavaScript errors
    alert('An unexpected error occurred. Please try again later.');
  }

  private handleUnknownError(error: any): void {
    // Handle unknown errors
    alert('An unknown error occurred. Please try again later.');
  }

  private logError(error: any): void {
    // Implement your logging logic here, e.g., send the error to a server
    console.error('Logging the error:', error);
  }
}
