import {Routes} from '@angular/router';
import {BodyComponent} from "./components/body/body.component";
import {TasksComponent} from "./components/body/tasks/tasks.component";
import {LeaderboardsComponent} from "./components/body/leaderboards/leaderboards.component";
import {SubmissionsComponent} from "./components/body/submissions/submissions.component";
import {DatasetsComponent} from "./components/body/datasets/datasets.component";
import {AboutComponent} from "./components/body/about/about.component";
import {TaskComponent} from "./components/body/tasks/task/task.component";
import {SubmittingComponent} from "./components/body/submitting/submitting.component";
import {ControlPanelComponent} from "./components/body/control-panel/control-panel.component";
import {AdminLoginComponent} from "./components/body/control-panel/admin-login/admin-login.component";
import {NotFoundComponent} from "./error_handling/not-found/not-found.component";

export const routes: Routes = [
  {path: '', component : BodyComponent},
  {path: 'tasks', component: TasksComponent},
  {path: 'tasks/:task', component: TaskComponent},
  {path: 'submissions', component: SubmissionsComponent},
  {path: 'submitting', component: SubmittingComponent},
  {path: 'datasets', component: DatasetsComponent},
  {path: 'about', component: AboutComponent},
  {path: 'leaderboards', component: LeaderboardsComponent},
  {path: 'leaderboards/:task', component: LeaderboardsComponent},
  {path: 'control', component: AdminLoginComponent},
  {path: 'control-panel', component: ControlPanelComponent},
  {path: '404', component: NotFoundComponent},
  {path: '**', redirectTo: '/404'}
];
