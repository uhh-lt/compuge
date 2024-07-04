from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.exc import IntegrityError
import pandas as pd
import os
import repositories.db_engine as engine
from repositories.db_engine import Leaderboard


# Insert a leaderboard record
def insert_data(data):
    try:
        engine.session.add(data)
        engine.session.commit()
    except IntegrityError as e:
        print(f"Error: {e}")
        engine.session.rollback()
        return False
    print("Data inserted successfully!")
    return True


# Query data by task
def query_data_by_task(task):
    return engine.session.query(Leaderboard).filter(Leaderboard.task == task).all()


# Query data by model
def query_data_by_model(model):
    return engine.session.query(Leaderboard).filter(Leaderboard.model == model).all()


# Query data by task and model
def query_data_by_task_and_model(task, model):
    return engine.session.query(Leaderboard).filter(Leaderboard.task == task, Leaderboard.model == model).all()


# Query data by task and dataset
def query_data_by_task_and_dataset(task, dataset):
    return engine.session.query(Leaderboard).filter(Leaderboard.task == task, Leaderboard.dataset == dataset).all()


# Query data by model, task and dataset
def query_data_by_task_and_dataset_and_model(task, dataset, model):
    return engine.session.query(Leaderboard).filter(Leaderboard.task == task, Leaderboard.dataset == dataset,
                                                    Leaderboard.model == model).all()


def query_all():
    return engine.session.query(Leaderboard).all()