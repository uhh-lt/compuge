import fastapi
import pandas as pd

import database_service

app = fastapi.FastAPI()

database_service.insert_data_from_dataframe(pd.read_csv("CQI_Leaderboard.csv"), "Question Identification")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/")
def root():
    return "Welcome to CompUGE API! This API is private and not intended for public use."


@app.get("/")
def root():
    return "Welcome to CompUGE backend! Please visit /api/ for the API documentation."


@app.get("/api/leaderboard/QI")
def leaderboard():
    return database_service.query_data_by_task_as_dataframe("Question Identification")