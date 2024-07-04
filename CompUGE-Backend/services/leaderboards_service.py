import os
import repositories.submissions_repository as sub_repo
import repositories.leaderboards_repository as lb_repo
from repositories.db_engine import Leaderboard, Submission
import pandas as pd
import numpy as np
from io import StringIO


def get_leaderboard(task: str, dataset: str):
    return lb_repo.query_data_by_task_and_dataset(task, dataset)


# TODO: Untested
def get_leaderboards():
    leaderboard = lb_repo.query_all()
    return leaderboard

'''
# split the leaderboard into a list of leaderboards, one for each task
    leaderboards = {}
    for lb in leaderboard:
        if lb.task not in leaderboards:
            leaderboards[lb.task] = []
        leaderboards[lb.task].append(lb)
    # for each leaderboard, split the leaderboard into a list of leaderboards, one for each dataset
    for task in leaderboards:
        datasets = {}
        for lb in leaderboards[task]:
            if lb.dataset not in datasets:
                datasets[lb.dataset] = []
            datasets[lb.dataset].append(lb)
        leaderboards[task] = datasets
'''