from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.exc import IntegrityError
import pandas as pd
import os

DB_URL = os.environ.get('DB_USER', 'postgresql://postgres:@localhost:5433/postgres')

# Create an engine
engine = create_engine(DB_URL)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()


# Define the schema
# Model,Accuracy,Precision,Recall,F1 Score,Evaluation Time,Overall Score
class Leaderboard(Base):
    __tablename__ = 'leaderboard'
    task = Column(String, primary_key=True)
    model = Column(String, primary_key=True)
    size = Column(Float)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    overall_score = Column(Float)

    def __repr__(self):
        return f"<Leaderboard(task='{self.task}', model='{self.model}', accuracy='{self.accuracy}', precision='{self.precision}', recall='{self.recall}', f1_score='{self.f1_score}', evaluation_time='{self.evaluation_time}', overall_score='{self.overall_score}')>"

    def __str__(self):
        return f"<Leaderboard(task='{self.task}', model='{self.model}', accuracy='{self.accuracy}', precision='{self.precision}', recall='{self.recall}', f1_score='{self.f1_score}', evaluation_time='{self.evaluation_time}', overall_score='{self.overall_score}')>"


# Insert a leaderboard record
def insert_data(data):
    try:
        session.add(data)
        session.commit()
    except IntegrityError as e:
        print(f"Error: {e}")
        session.rollback()
        return False
    print("Data inserted successfully!")
    return True


def insert_data_from_dataframe(data, task):
    data['task'] = task
    data.to_sql('leaderboard', con=engine, if_exists='append', index=False)
    print("Data inserted successfully!")


# Query data by task
def query_data_by_task(task):
    return session.query(Leaderboard).filter(Leaderboard.task == task).all()


# Query data by model
def query_data_by_model(model):
    return session.query(Leaderboard).filter(Leaderboard.model == model).all()


# Query data by task return as dataframe
def query_data_by_task_as_dataframe(task):
    return pd.read_sql(session.query(Leaderboard).filter(Leaderboard.task == task).statement, session.bind)


# Query data by model return as dataframe
def query_data_by_model_as_dataframe(model):
    return pd.read_sql(session.query(Leaderboard).filter(Leaderboard.model == model).statement, session.bind)


# Query data by task and model
def query_data_by_task_and_model(task, model):
    return session.query(Leaderboard).filter(Leaderboard.task == task, Leaderboard.model == model).first()


# Query data by task and model return as dataframe
def query_data_by_task_and_model_as_dataframe(task, model):
    return pd.read_sql(
        session.query(Leaderboard).filter(Leaderboard.task == task, Leaderboard.model == model).statement,
        session.bind)


# Update data by task and model
def update_data_by_task_and_model(task, model, data):
    try:
        session.query(Leaderboard).filter(Leaderboard.task == task, Leaderboard.model == model).update(data)
        session.commit()
    except IntegrityError as e:
        print(f"Error: {e}")
        session.rollback()
    print("Data updated successfully!")


# Delete data by task and model
def delete_data_by_task_and_model(task, model):
    try:
        session.query(Leaderboard).filter(Leaderboard.task == task, Leaderboard.model == model).delete()
        session.commit()
    except IntegrityError as e:
        print(f"Error: {e}")
        session.rollback()
    print("Data deleted successfully!")


# Create the table
Base.metadata.create_all(engine)
