export interface LeaderboardEntry {
    task: string;
    dataset: string;
    model: string;
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
}
