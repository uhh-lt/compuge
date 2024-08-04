class DatasetDTO:
    def __init__(self, task: str,
                 name: str,
                 description: str,
                 link: str,
                 paper: str,
                 paper_link: str,
                 train: list,
                 test: list):
        self.task = task
        self.name = name
        self.description = description
        self.link = link
        self.paper = paper
        self.paper_link = paper_link
        self.train = train
        self.test = test


class TaskDTO:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description


class SubmissionDTO:
    def __init__(self, task: str, dataset: str, model: str, link: str, predictions: str, status: str, time: str):
        self.task = task
        self.dataset = dataset
        self.model = model
        self.link = link
        self.predictions = predictions
        self.status = status
        self.time = time


class LeaderboardDTO:
    def __init__(self, task: str, dataset: str, model: str, accuracy: float, precision: float, recall: float,
                 f1_score: float):
        self.task = task
        self.dataset = dataset
        self.model = model
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
