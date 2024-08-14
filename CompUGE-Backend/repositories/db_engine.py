import os
import logging
from sqlalchemy import create_engine, Column, String, Float, Integer, ForeignKey, Boolean, Text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get database URL from environment variable or use a default value
DB_URL = os.environ.get('DB_URL', 'postgresql+psycopg2://postgres:@localhost:5432/postgres')

try:
    # Create an engine
    engine = create_engine(DB_URL)
    logger.info("Database engine created successfully.")
except Exception as e:
    logger.error(f"Failed to create database engine: {e}")
    raise

try:
    # Create a configured "Session" class
    Session = sessionmaker(bind=engine)
    # Create a Session
    session = Session()
    logger.info("Database session created successfully.")
except Exception as e:
    logger.error(f"Failed to create database session: {e}")
    raise

# Base class for declarative models
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

    # Define the relationship to the Leaderboard table
    leaderboard_entry = relationship('Leaderboard', back_populates='submission', cascade="all, delete-orphan")

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

    # Define the relationship back to the Submission table
    submission = relationship('Submission', back_populates='leaderboard_entry')

    def __repr__(self):
        return f"<Leaderboard(id='{self.id}', submission_id='{self.submission_id}', accuracy='{self.accuracy}', precision='{self.precision}', recall='{self.recall}', f1_score='{self.f1_score}')>"

    def __str__(self):
        return f"<Leaderboard(id='{self.id}', submission_id='{self.submission_id}', accuracy='{self.accuracy}', precision='{self.precision}', recall='{self.recall}', f1_score='{self.f1_score}')>"

try:
    # Create all tables in the database which are defined by Base's subclasses
    Base.metadata.create_all(engine)
    logger.info("All tables created successfully.")
except Exception as e:
    logger.error(f"Failed to create tables: {e}")
    raise


# A method to check the database connection
def check_db_connection():
    try:
        session.execute('SELECT 1')
        logger.info("Database connection check succeeded.")
        return "pong"
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return f"Database connection failed: {str(e)}"