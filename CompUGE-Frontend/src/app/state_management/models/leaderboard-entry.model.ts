export interface LeaderboardEntry {
    task: string;
    dataset: string;
    model: string;
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    predictions: string;
    team: string;
    is_public: boolean;
    blob_url: string;
}
