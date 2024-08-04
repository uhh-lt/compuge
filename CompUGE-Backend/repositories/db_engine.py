import os

from sqlalchemy import create_engine, Column, String, Float, Integer, ForeignKey, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

DB_URL = os.environ.get('DB_URL', 'postgresql://postgres:@localhost:5432/postgres')

# Create an engine
engine = create_engine(DB_URL)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()


class Submission(Base):
    __tablename__ = 'submission'

    id = Column(Integer, primary_key=True, autoincrement=True)
    time = Column(String, nullable=False)
    task = Column(String, nullable=False)
    dataset = Column(String, nullable=False)
    model = Column(String, nullable=False)
    link = Column(String, nullable=True)
    team = Column(String, nullable=True)
    email = Column(String, nullable=True)
    predictions = Column(Text, nullable=False)
    status = Column(String, nullable=False)
    is_public = Column(Boolean, nullable=False)

    def __repr__(self):
        return f"<Submission(id='{self.id}', time='{self.time}', task='{self.task}', dataset='{self.dataset}', model='{self.model}', link='{self.link}', team='{self.team}', email='{self.email}', predictions='{self.predictions}', status='{self.status}', is_public='{self.is_public}')>"

    def __str__(self):
        return f"<Submission(id='{self.id}', time='{self.time}', task='{self.task}', dataset='{self.dataset}', model='{self.model}', link='{self.link}', team='{self.team}', email='{self.email}', status='{self.status}', is_public='{self.is_public}')>"


class Leaderboard(Base):
    __tablename__ = 'leaderboard'

    id = Column(Integer, primary_key=True, autoincrement=True)
    submission_id = Column(Integer, ForeignKey('submission.id'), nullable=False)
    accuracy = Column(Float, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)

    # Define the relationship to the Submission table
    submission = relationship('Submission', back_populates='leaderboard_entry')

    def __repr__(self):
        return f"<Leaderboard(id='{self.id}', submission_id='{self.submission_id}', accuracy='{self.accuracy}', precision='{self.precision}', recall='{self.recall}', f1_score='{self.f1_score}')>"

    def __str__(self):
        return f"<Leaderboard(id='{self.id}', submission_id='{self.submission_id}', accuracy='{self.accuracy}', precision='{self.precision}', recall='{self.recall}', f1_score='{self.f1_score}')>"


# Define the back_populates relationship in Submission
Submission.leaderboard_entry = relationship('Leaderboard', order_by=Leaderboard.id, back_populates='submission')

Base.metadata.create_all(engine)
