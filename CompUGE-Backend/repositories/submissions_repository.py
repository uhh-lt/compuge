import logging
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from repositories.db_engine import Submission, session, Leaderboard

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def query_all():
    """
    Query all submissions
    :return: List of all submissions or None if an error occurred
    """
    try:
        results = session.query(Submission).all()
        logger.info("Queried all submissions")
        return results
    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemyError while querying all submissions: {e}")
        return None


def query_all_with_eval_metrics():
    """
    Query all submissions with leaderboard entries (evaluation metrics) included
    :return: List of all submissions with leaderboard entries or None if an error occurred
    """
    try:
        results = (session.query(
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
        )
                   .join(Leaderboard, Leaderboard.submission_id == Submission.id)
                   .all())
        logger.info("Queried all submissions with leaderboard entries")
        return results
    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemyError while querying all submissions with leaderboard entries: {e}")
        return None


def query_submission_by_id(sub_id):
    """
    Query submission by ID
    :param sub_id: the submission ID
    :return: a submission object with the same ID or None if an error occurred
    """
    try:
        result = session.execute(
            f"SELECT * FROM submission WHERE id = {sub_id}"
        )
        logger.info(f"Queried data by submission ID: {sub_id}")
        return result
    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemyError while querying data by submission ID {sub_id}: {e}")
        return None


def query_leaderboard_by_submission_id(sub_id):
    """
    Query leaderboard entry by submission ID
    :param sub_id: the submission ID
    :return: a leaderboard object with the same submission ID or None if an error occurred
    """
    try:
        result = session.execute(
            f"SELECT * FROM leaderboard WHERE submission_id = {sub_id}"
        )
        logger.info(f"Queried leaderboard by submission ID: {sub_id}")
        return result
    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemyError while querying leaderboard by submission ID {sub_id}: {e}")
        return None


def insert_data(submission, leaderboard):
    """
    Insert submission and leaderboard data into the database
    :param submission: the submission object
    :param leaderboard: the leaderboard object
    :return: True if the data was inserted successfully, False otherwise
    """
    try:
        session.add(submission)
        leaderboard.submission_id = submission.id
        session.add(leaderboard)
        session.commit()
        logger.info("Data inserted successfully!")
        return True
    except IntegrityError as e:
        logger.error(f"IntegrityError while inserting data: {e}")
        session.rollback()
        return False
    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemyError while inserting data: {e}")
        session.rollback()
        return False
    except Exception as e:
        logger.error(f"Unexpected error while inserting data: {e}")
        session.rollback()
        return False


def update_data(submission, leaderboard):
    """
    Update submission and leaderboard data in the database
    :param submission: the updated submission object
    :param leaderboard: the updated leaderboard object
    :return: True if the data was updated successfully, False otherwise
    """
    try:
        session.add(submission)
        session.add(leaderboard)
        session.commit()
        logger.info("Data updated successfully!")
        return True
    except IntegrityError as e:
        logger.error(f"IntegrityError while updating data: {e}")
        session.rollback()
        return False
    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemyError while updating data: {e}")
        session.rollback()
        return False
    except Exception as e:
        logger.error(f"Unexpected error while updating data: {e}")
        session.rollback()
        return False


def delete_submission(sub_id):
    """
    Delete submission and associated leaderboard entry
    :param sub_id: the submission ID
    :return: True if the submission was deleted successfully, False otherwise
    """
    try:
        submission = query_submission_by_id(sub_id)
        if submission is None:
            logger.warning(f"Submission not found for ID: {sub_id}")
            return False
        # Delete the submission and associated leaderboard entry due to cascade delete
        session.delete(submission)
        session.commit()
        logger.info(f"Submission deleted successfully for ID: {sub_id}")
        return True
    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemyError while deleting submission ID {sub_id}: {e}")
        session.rollback()
        return False
    except Exception as e:
        logger.error(f"Unexpected error while deleting submission ID {sub_id}: {e}")
        session.rollback()
        return False
