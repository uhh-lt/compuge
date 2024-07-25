export interface DatasetModel {
  task: string;
  name: string;
  description: string;
  link: string;
  paper: string;
  paper_link: string;
  train: any[];
  test: any[];
}
