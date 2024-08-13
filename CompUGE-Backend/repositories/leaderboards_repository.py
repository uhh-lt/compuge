import logging
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

import repositories.db_engine as engine
from repositories.db_engine import Leaderboard, Submission

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Insert a leaderboard record
def insert_data(data):
    try:
        engine.session.add(data)
        engine.session.commit()
        logger.info("Data inserted successfully!")
        return True
    except IntegrityError as e:
        logger.error(f"IntegrityError while inserting data: {e}")
        engine.session.rollback()
        return False
    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemyError while inserting data: {e}")
        engine.session.rollback()
        return False
    except Exception as e:
        logger.error(f"Unexpected error while inserting data: {e}")
        engine.session.rollback()
        return False


# Query data by task
def query_data_by_task(task):
    try:
        results = (engine.session.query(Leaderboard)
                   .join(Submission)
                   .filter(Submission.task == task)
                   .all())
        logger.info(f"Queried data by task: {task}")
        return results
    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemyError while querying data by task {task}: {e}")
        return []


# Query data by model
def query_data_by_model(model):
    try:
        results = (engine.session.query(Leaderboard)
                   .join(Submission)
                   .filter(Submission.model == model)
                   .all())
        logger.info(f"Queried data by model: {model}")
        return results
    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemyError while querying data by model {model}: {e}")
        return []


# Query data by task and model
def query_data_by_task_and_model(task, model):
    try:
        results = (engine.session.query(Leaderboard)
                   .join(Submission)
                   .filter(Submission.task == task,
                           Submission.model == model)
                   .all())
        logger.info(f"Queried data by task and model: task={task}, model={model}")
        return results
    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemyError while querying data by task and model: task={task}, model={model}: {e}")
        return []


# Query data by task and dataset
def query_data_by_task_and_dataset(task, dataset):
    try:
        results = (engine.session.query(Leaderboard)
                   .join(Submission)
                   .filter(Submission.task == task,
                           Submission.dataset == dataset)
                   .all())
        logger.info(f"Queried data by task and dataset: task={task}, dataset={dataset}")
        return results
    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemyError while querying data by task and dataset: task={task}, dataset={dataset}: {e}")
        return []


# Query data by task, dataset, and model
def query_data_by_task_and_dataset_and_model(task, dataset, model):
    try:
        results = (engine.session.query(Leaderboard)
                   .join(Submission)
                   .filter(Submission.task == task,
                           Submission.dataset == dataset,
                           Submission.model == model)
                   .all())
        logger.info(f"Queried data by task, dataset, and model: task={task}, dataset={dataset}, model={model}")
        return results
    except SQLAlchemyError as e:
        logger.error(
            f"SQLAlchemyError while querying data by task, dataset, and model: task={task}, dataset={dataset}, model={model}: {e}")
        return []


# Query all data
def query_all():
    try:
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
        logger.info("Queried all data")
        return results
    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemyError while querying all data: {e}")
        return []


# Update a leaderboard record
def update_entry(data):
    try:
        engine.session.add(data)
        engine.session.commit()
        logger.info("Data updated successfully!")
        return True
    except IntegrityError as e:
        logger.error(f"IntegrityError while updating data: {e}")
        engine.session.rollback()
        return False
    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemyError while updating data: {e}")
        engine.session.rollback()
        return False
    except Exception as e:
        logger.error(f"Unexpected error while updating data: {e}")
        engine.session.rollback()
        return False


def query_data_by_submission_id(sub_id):
    try:
        result = engine.session.query(Leaderboard).filter(Leaderboard.submission_id == sub_id).first()
        logger.info(f"Queried data by submission ID: {sub_id}")
        return result
    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemyError while querying data by submission ID {sub_id}: {e}")
        return None


def delete_data(sub_id):
    try:
        record = query_data_by_submission_id(sub_id)
        if record is None:
            logger.warning(f"Record not found for submission ID: {sub_id}")
            return "Record not found"
        engine.session.delete(record)
        engine.session.commit()
        logger.info(f"Record deleted successfully for submission ID: {sub_id}")
        return "Record deleted successfully"
    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemyError while deleting data for submission ID {sub_id}: {e}")
        engine.session.rollback()
        return "Error occurred while deleting the record"
    except Exception as e:
        logger.error(f"Unexpected error while deleting data for submission ID {sub_id}: {e}")
        engine.session.rollback()
        return "Error occurred while deleting the record"
