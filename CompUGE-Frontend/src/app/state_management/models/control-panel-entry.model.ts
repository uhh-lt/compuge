export interface ControlPanelEntry {
  id: number;
  task: string;
  dataset: string;
  model: string;
  link: string;
  team: string;
  email: string;
  status: string;
  time: string;
  is_public: boolean;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
}
