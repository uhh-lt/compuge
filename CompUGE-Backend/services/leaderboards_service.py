import repositories.leaderboards_repository as lb_repo


def get_leaderboard(task: str, dataset: str):
    return lb_repo.query_data_by_task_and_dataset(task, dataset)


def get_leaderboards():
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
