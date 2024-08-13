import datetime
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import repositories.leaderboards_repository as ld_repo
import repositories.submissions_repository as sub_repo
from repositories.db_engine import Leaderboard, Submission
from services.dataset_service import get_test_labels, get_labels_from_csv

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def submit_solution(task, dataset, submission_dict):
    # Extract submission information
    model = submission_dict["modelName"]
    link = submission_dict["modelLink"]
    team = submission_dict["teamName"]
    email = submission_dict["contactEmail"]
    is_public = bool(submission_dict["isPublic"])

    try:
        # Get dataset labels and predictions
        labels = get_test_labels(task, dataset)
        predictions = get_labels_from_csv(submission_dict["fileContent"])
        accuracy, precision, recall, f1 = evaluate_model(labels, predictions)

        # Create and persist the submission
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

        # Create and persist the leaderboard entry
        record = Leaderboard(
            submission_id=submission.id,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
        )
        ld_repo.insert_data(record)

        return "Submission successful"

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")

        # Handle rejected submission
        submission = Submission(
            time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            task=task,
            dataset=dataset,
            model=model,
            link=link,
            team=team,
            email=email,
            status="rejected",
            is_public=is_public,
        )
        sub_repo.insert_data(submission)

        record = Leaderboard(
            submission_id=submission.id,
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
        )
        ld_repo.insert_data(record)

        return "Submission failed due to model evaluation error"


def evaluate_model(labels, predictions):
    # Evaluate the model and return the metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')

    return float(accuracy), float(precision), float(recall), float(f1)


def get_submissions():
    return sub_repo.query_all()


def get_submissions_with_their_leaderboard_entries():
    result = sub_repo.query_all_with_leaderboard_entries()
    result_dict = [
        {
            'id': r.id,
            'accuracy': r.accuracy,
            'precision': r.precision,
            'recall': r.recall,
            'f1_score': r.f1_score,
            'task': r.task,
            'dataset': r.dataset,
            'model': r.model,
            'link': r.link,
            'team': r.team,
            'email': r.email,
            'is_public': r.is_public,
            'status': r.status,
            'time': r.time
        }
        for r in result
    ]
    return result_dict


def force_update_submission(sub_id, submission_dict):
    submission = sub_repo.query_data_by_id(sub_id)
    if submission is None:
        return "Submission not found"

    leaderboardEntry = ld_repo.query_data_by_submission_id(sub_id)

    # Update the submission and leaderboard entry with new data
    for key in submission_dict:
        if hasattr(submission, key):
            setattr(submission, key, submission_dict[key])
        elif hasattr(leaderboardEntry, key):
            setattr(leaderboardEntry, key, submission_dict[key])

    sub_repo.update_submission(submission)
    ld_repo.update_entry(leaderboardEntry)

    return "Submission updated successfully"


def delete_submission(sub_id):
    submission = sub_repo.query_data_by_id(sub_id)
    if submission is None:
        return "Submission not found"

    ld_repo.delete_data(sub_id)
    sub_repo.delete_data(sub_id)

    return "Submission deleted successfully"
