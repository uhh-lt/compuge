from sqlalchemy.exc import IntegrityError

import repositories.db_engine as engine
from repositories.db_engine import Submission


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
    return engine.session.query(Submission).filter(Submission.task == task).all()


# Query data by model
def query_data_by_model(model):
    return engine.session.query(Submission).filter(Submission.model == model).all()


# Query data by task and model
def query_data_by_task_and_model(task, model):
    return engine.session.query(Submission).filter(Submission.task == task, Submission.model == model).first()


# Query data by task and dataset
def query_data_by_task_and_dataset(task, dataset):
    return engine.session.query(Submission).filter(Submission.task == task, Submission.dataset == dataset).all()


# Query data by model, task and dataset
def query_data_by_task_and_dataset_and_model(task, dataset, model):
    return engine.session.query(Submission).filter(Submission.task == task, Submission.dataset == dataset,
                                                   Submission.model == model).all()


def query_all():
    return engine.session.query(Submission).all()
