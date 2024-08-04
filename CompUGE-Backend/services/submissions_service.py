import datetime

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import repositories.leaderboards_repository as ld_repo
import repositories.submissions_repository as sub_repo
from repositories.db_engine import Leaderboard, Submission
from services.dataset_service import get_test_labels, get_labels_from_csv


def submit_solution(task, dataset, submission_dict):
    # Extract submission information
    model = submission_dict["modelName"]
    link = submission_dict["modelLink"]
    team = submission_dict["teamName"]
    email = submission_dict["contactEmail"]
    predictions = get_labels_from_csv(submission_dict["fileContent"])
    is_public = bool(submission_dict["isPublic"])

    # Get dataset split
    labels = get_test_labels(task, dataset)

    # Handle accepted submission
    submission = Submission(
        time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        task=task,
        dataset=dataset,
        model=model,
        link=link,
        team=team,
        email=email,
        predictions=str(predictions),
        status="accepted",
        is_public=is_public,
    )
    sub_repo.insert_data(submission)

    # Perform model evaluation
    accuracy, precision, recall, f1 = evaluate_model(labels, predictions)
    record = Leaderboard(
        submission_id=submission.id,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
    )

    if ld_repo.insert_data(record):
        return "Submission successful"
    return "Submission failed due to persistence error"


# a method that receives a list of labels and predictions and returns :
def evaluate_model(labels, predictions):
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')

    return float(accuracy), float(precision), float(recall), float(f1)


def get_submissions():
    return sub_repo.query_all()
