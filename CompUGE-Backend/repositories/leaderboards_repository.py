from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import joinedload
import repositories.db_engine as engine
from repositories.db_engine import Leaderboard, Submission


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
    return (engine.session.query(Leaderboard)
            .join(Submission)
            .filter(Submission.task == task)
            .all())


# Query data by model
def query_data_by_model(model):
    return (engine.session.query(Leaderboard)
            .join(Submission)
            .filter(Submission.model == model)
            .all())


# Query data by task and model
def query_data_by_task_and_model(task, model):
    return (engine.session.query(Leaderboard)
            .join(Submission)
            .filter(Submission.task == task,
                    Submission.model == model)
            .all())


# Query data by task and dataset
def query_data_by_task_and_dataset(task, dataset):
    return (engine.session.query(Leaderboard)
            .join(Submission)
            .filter(Submission.task == task,
                    Submission.dataset == dataset)
            .all())


# Query data by task, dataset, and model
def query_data_by_task_and_dataset_and_model(task, dataset, model):
    return (engine.session.query(Leaderboard)
            .join(Submission)
            .filter(Submission.task == task,
                    Submission.dataset == dataset,
                    Submission.model == model)
            .all())


# Query all data
def query_all():
    # Query the joined tables and select the desired columns
    results = (
        engine.session.query(
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
            Submission.predictions,
            Submission.is_public
        )
        .join(Submission, Leaderboard.submission_id == Submission.id)
        .all()
    )

    return results
