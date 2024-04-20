import fastapi
import pandas as pd
import submission
import database_service
from fastapi.middleware.cors import CORSMiddleware

app = fastapi.FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/")
def root():
    return "Welcome to CompUGE API! This API is private and not intended for public use."


@app.get("/")
def root():
    return "Welcome to CompUGE backend! Please visit /api/ for the API documentation."


@app.get("/api/leaderboard/{task}")
def leaderboard(task: str):
    return database_service.query_data_by_task(task)


@app.post("/api/submission")
def submit(submission_dict: dict):
    return submission.submit_solution(submission_dict)

