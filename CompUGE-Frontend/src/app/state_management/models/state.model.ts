import {TaskModel} from "./task.mode";
import {DatasetModel} from "./dataset.model";
import {SubmissionEntry} from "./submission-entry.model";
import {LeaderboardEntry} from "./leaderboard-entry.model";
import {ControlPanelEntry} from "./control-panel-entry.model";
import {Observable} from "rxjs";

export interface StateModel{
    leaderboards: LeaderboardEntry[];
    submissions: SubmissionEntry[];
    datasets: DatasetModel[];
    tasks: TaskModel[];
    controlPanelSubmissions: ControlPanelEntry[];
    adminSessionStatus: string;
}
