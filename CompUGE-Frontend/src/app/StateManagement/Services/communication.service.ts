import { Injectable } from '@angular/core';
import {environment} from "../../../environments/environment";
import {HttpClient} from "@angular/common/http";

@Injectable({
  providedIn: 'root'
})
export class CommunicationService {

  api = environment.apiUrl;

  constructor(
    private http: HttpClient
  ) {
  }

  public getLeaderboard(task: String) {
    return this.http.get(this.api + '/api/leaderboard/' + task);
  }


  public submit(modelName: string, modelLink: string, task: string, fileContent: string) {
    return this.http.post(this.api + '/api/submission', {
      modelName: modelName,
      modelLink: modelLink,
      task: task,
      fileContent: fileContent
    });
  }
}
