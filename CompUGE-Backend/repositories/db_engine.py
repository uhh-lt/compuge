from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Float
import os

DB_URL = os.environ.get('DB_USER', 'postgresql://postgres:@localhost:5432/postgres')

# Create an engine
engine = create_engine(DB_URL)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()


class Leaderboard(Base):
    __tablename__ = 'leaderboard'
    task = Column(String, primary_key=True)
    dataset = Column(String, primary_key=True)
    model = Column(String, primary_key=True)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    overall_score = Column(Float)

    def __repr__(self):
        return f"<Leaderboard(task='{self.task}', model='{self.model}', accuracy='{self.accuracy}', precision='{self.precision}', recall='{self.recall}', f1_score='{self.f1_score}', evaluation_time='{self.evaluation_time}', overall_score='{self.overall_score}')>"

    def __str__(self):
        return f"<Leaderboard(task='{self.task}', model='{self.model}', accuracy='{self.accuracy}', precision='{self.precision}', recall='{self.recall}', f1_score='{self.f1_score}', evaluation_time='{self.evaluation_time}', overall_score='{self.overall_score}')>"


class Submission(Base):
    __tablename__ = 'submission'
    time = Column(String)
    task = Column(String, primary_key=True)
    dataset = Column(String, primary_key=True)
    model = Column(String, primary_key=True)
    link = Column(String)
    predictions = Column(String)
    status = Column(String)

    def __repr__(self):
        return f"<Submission(time='{self.time}', task='{self.task}', dataset='{self.dataset}', model='{self.model}', link='{self.link}', predictions='{self.predictions}')>"

    def __str__(self):
        return f"<Submission(time='{self.time}', task='{self.task}', dataset='{self.dataset}', model='{self.model}', link='{self.link}', predictions='{self.predictions}')>"


Base.metadata.create_all(engine)
