import datetime
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import repositories.submissions_repository as sub_repo
from repositories.db_engine import Leaderboard, Submission
from services.dataset_service import get_test_labels, get_predictions_from_csv

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def submit_solution(task, dataset, submission_dict):
    """
    Submit a solution for evaluation and persist the submission data and evaluation metrics if successful,
    or persist the rejected submission data if unsuccessful.
    :param task: the task name for the submission
    :param dataset: the dataset name for the submission
    :param submission_dict: a dictionary containing the submission data
                            format: {
                                "modelName": str,
                                "modelLink": str,
                                "teamName": str,
                                "contactEmail": str,
                                "fileContent": str,
                                "isPublic": bool
                            }
    :return: a message indicating the outcome of the submission operation
            Options:
            - "Submission successful"
            - "Submission rejected"
            - "Error getting test labels"
            - "Error getting predictions from CSV"
            - "Error persisting accepted submission data"
            - "Error persisting rejected submission data"
    :raises: Exception if input is invalid
    """
    model = submission_dict["modelName"]
    link = submission_dict["modelLink"]
    team = submission_dict["teamName"]
    email = submission_dict["contactEmail"]
    is_public = bool(submission_dict["isPublic"] == "true")
    labels = get_test_labels(task, dataset)
    if labels is None:
        return "Error getting test labels"

    predictions = get_predictions_from_csv(submission_dict["fileContent"])
    if predictions is None:
        return "Error getting predictions from CSV"

    try:
        accuracy, precision, recall, f1 = evaluate_model(labels, predictions)
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
        record = Leaderboard(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
        )
        if sub_repo.insert_data(submission, record):
            return "Submission successful"
        return "Error persisting accepted submission data"
    except Exception as e:
        logger.error(f"An unexpected error occurred in submit_solution: {e}")
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

        record = Leaderboard(
            submission_id=submission.id,
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
        )
        if sub_repo.insert_data(submission, record):
            return "Submission rejected"
        return "Error persisting rejected submission data"


def evaluate_model(labels, predictions):
    """
    Evaluate the model using the provided labels and predictions.
    :param labels: the true labels
    :param predictions: the predicted labels
    :return: a tuple containing the evaluation metrics (accuracy, precision, recall, f1 score) as floats
    :raise ValueError: if the labels and predictions have different lengths
    :raise Exception: if an unexpected error occurs
    """
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')
    return float(accuracy), float(precision), float(recall), float(f1)


def get_submissions():
    """
    Get all submissions.
    :return: A list of all submissions or an empty list if an error occurs.
    :return: A list containing all submissions or an empty list if an error occurs.
    """
    submissions = sub_repo.query_all()
    return submissions if submissions else []


def get_submission_for_control_panel():
    """
    Get all submissions with evaluation metrics and ID for the control panel.
    :return: A list of dictionaries containing submission data or an empty list if an error occurs.
    """
    try:
        result = sub_repo.query_all_with_eval_metrics()
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

    except Exception as e:
        logger.error(f"An unexpected error occurred in get_submission_for_control_panel: {e}")
        return []


def get_leaderboards():
    """
    Get all leaderboards with evaluation metrics.
    :return: A list of dictionaries containing leaderboard data or an empty list if an error occurs.
    """
    try:
        leaderboard = sub_repo.query_all_with_eval_metrics()
        results_dict = [
            {
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
                'predictions': r.predictions,
                'is_public': r.is_public,
            }
            for r in leaderboard
        ]

        return results_dict

    except Exception as e:
        logger.error(f"An unexpected error occurred in get_leaderboards: {e}")
        return []


def update_submission(sub_id, submission_dict):
    """
    Update a submission and its corresponding leaderboard entry.
    :param sub_id: the ID of the submission to update
    :param submission_dict: a dictionary containing the updated submission data
    :return: a message indicating the outcome of the update operation
            Options:
            - "Submission updated successfully"
            - "Submission not found"
            - "Error updating submission data in the database"
    :raises: Exception if an unexpected error occurs
    """
    try:
        submission = sub_repo.query_submission_by_id(sub_id)
        if submission is None:
            return "Submission not found"

        leaderboardEntry = sub_repo.query_leaderboard_by_submission_id(sub_id)

        for key in submission_dict:
            if hasattr(submission, key):
                setattr(submission, key, submission_dict[key])
            elif hasattr(leaderboardEntry, key):
                setattr(leaderboardEntry, key, submission_dict[key])

        if sub_repo.update_data(submission, leaderboardEntry):
            return "Submission updated successfully"
        return "Error updating submission data in the database"

    except Exception as e:
        logger.error(f"An unexpected error occurred in update_submission: {e}")
        raise  # Re-raise to be handled by the controller


def delete_submission(sub_id):
    """
    Delete a submission and its corresponding leaderboard entry.
    :param sub_id: the ID of the submission to delete
    :return: a message indicating the outcome of the delete operation
            Options:
            - "Submission deleted successfully"
            - "Submission not found"
            - "Error deleting submission data from the database"
    :raises: Exception if an unexpected error occurs
    """
    try:
        submission = sub_repo.query_submission_by_id(sub_id)
        if submission is None:
            return "Submission not found"
        if sub_repo.delete_submission(sub_id):
            return "Submission deleted successfully"
        return "Error deleting submission data from the database"
    except Exception as e:
        logger.error(f"An unexpected error occurred in delete_submission: {e}")
        raise  # Re-raise to be handled by the controller
