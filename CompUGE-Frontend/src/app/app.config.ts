import {ApplicationConfig, ErrorHandler} from '@angular/core';
import {provideRouter, withComponentInputBinding} from '@angular/router';

import { routes } from './app.routes';
import { provideAnimationsAsync } from '@angular/platform-browser/animations/async';
import {provideHttpClient, withFetch} from "@angular/common/http";
import {GlobalErrorHandlerService} from "./error_handling/global-error-handler.service";

export const appConfig: ApplicationConfig = {
  providers: [
    provideRouter(routes, withComponentInputBinding()),
    provideAnimationsAsync(),
    provideAnimationsAsync(),
    provideHttpClient(withFetch()),
    {provide: ErrorHandler, useClass: GlobalErrorHandlerService}
  ]
};
