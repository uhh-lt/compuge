import datetime
import os
import repositories.submissions_repository as sub_repo
import repositories.leaderboards_repository as ld_repo
from repositories.db_engine import Leaderboard, Submission
import pandas as pd
import numpy as np
from io import StringIO


def get_submissions():
    return sub_repo.query_all()


def submit_solution(task, dataset, submission_dict):
    accuracy, precision, recall, f1, overall = evaluate_model(
        submission_dict["fileContent"]
    )
    submission = Submission(
        time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        task=task,
        dataset=dataset,
        model=submission_dict["modelName"],
        link=submission_dict["modelLink"],
        predictions=submission_dict["fileContent"],
        status="accepted"
    )

    record = Leaderboard(
        task=task,
        dataset=dataset,
        model=submission_dict["modelName"],
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        overall_score=overall
    )

    return sub_repo.insert_data(submission) and ld_repo.insert_data(record)


# a method that recieves a list of lables and predictions and returns :
def evaluate_model(fileContent):
    # file content is a string that contains the csv file
    df = pd.read_csv(StringIO(fileContent))
    labels = df["labels"]
    predictions = df["preds"]

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    overall = (accuracy + precision + recall + f1) / 4  # TODO: think about the overall score calculation
    return accuracy, precision, recall, f1, overall


# a method that calculates the accuracy of the model
def accuracy_score(labels, predictions):
    return sum([1 for i in range(len(labels)) if labels[i] == predictions[i]]) / len(labels)


def precision_score(labels, predictions):
    tp = sum([1 for i in range(len(labels)) if labels[i] == 1 and predictions[i] == 1])
    fp = sum([1 for i in range(len(labels)) if labels[i] == 0 and predictions[i] == 1])
    return tp / (tp + fp)


def recall_score(labels, predictions):
    tp = sum([1 for i in range(len(labels)) if labels[i] == 1 and predictions[i] == 1])
    fn = sum([1 for i in range(len(labels)) if labels[i] == 1 and predictions[i] == 0])
    return tp / (tp + fn)


def f1_score(labels, predictions):
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    return 2 * (precision * recall) / (precision + recall)

