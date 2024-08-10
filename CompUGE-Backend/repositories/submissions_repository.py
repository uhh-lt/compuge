from sqlalchemy.exc import IntegrityError

import repositories.db_engine as engine
from repositories.db_engine import Submission, Leaderboard


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


def query_all_with_leaderboard_entries():
    results = (engine.session.query(
            Submission.id,
            Leaderboard.accuracy,
            Leaderboard.precision,
            Leaderboard.recall,
            Leaderboard.f1_score,
            Submission.task,
            Submission.dataset,
            Submission.model,
            Submission.link,
            Submission.team,
            Submission.email,
            Submission.is_public,
            Submission.status,
            Submission.time
        ).
               join(Submission, Leaderboard.submission_id == Submission.id).
               all())
    return results


# Update a submission record in the database with the new data provided in the submission object parameter
def update_submission(submission):
    try:
        engine.session.add(submission)
        engine.session.commit()
    except IntegrityError as e:
        print(f"Error: {e}")
        engine.session.rollback()
        return False
    print("Data updated successfully!")
    return True


def query_data_by_id(sub_id):
    return engine.session.query(Submission).filter(Submission.id == sub_id).first()


def delete_data(sub_id):
    submission = query_data_by_id(sub_id)
    if submission is None:
        return "Submission not found"
    engine.session.delete(submission)
    engine.session.commit()
    return "Submission deleted successfully"
