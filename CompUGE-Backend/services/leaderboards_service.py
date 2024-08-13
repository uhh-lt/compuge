import repositories.leaderboards_repository as lb_repo
from fastapi import HTTPException
import logging

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_leaderboard(task: str, dataset: str):
    # Query the leaderboard for the given task and dataset
    leaderboard = lb_repo.query_data_by_task_and_dataset(task, dataset)

    # If no leaderboard entry is found, raise a 404 HTTP exception
    if not leaderboard:
        raise HTTPException(status_code=404, detail="Leaderboard not found")

    return leaderboard


def get_leaderboards():
    # Query all leaderboards
    leaderboard = lb_repo.query_all()

    # Convert the results to a list of dictionaries
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
