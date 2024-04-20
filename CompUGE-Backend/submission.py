import os
import database_service
import pandas as pd
import numpy as np
from io import StringIO

#modelName: modelName,
#modelLink: modelLink,
#task: task,
#fileContent: fileContent

def submit_solution(submission_dict):
    accuracy, precision, recall, f1, overall = evaluate_model(
        submission_dict["fileContent"]
    )

    record = database_service.Leaderboard(
        model=submission_dict["modelName"],
        task=submission_dict["task"],
        size=0,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        overall_score=overall
    )
    return database_service.insert_data(record)


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
