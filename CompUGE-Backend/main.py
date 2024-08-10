import json
from datetime import timedelta

import fastapi
from fastapi.middleware.cors import CORSMiddleware

import repositories.db_engine as db_engine
import services.dataset_service as ds_service
import services.leaderboards_service as lb_service
import services.submissions_service as sub_service
from dtos import TaskDTO
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import services.authentication_service as auth_service

app = fastapi.FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load tasks from tasks.json under key "tasks"
tasks = []
with open("tasks.json", "r") as tasks_file:
    tasks_dict = json.load(tasks_file)
    for task in tasks_dict["tasks"]:
        tasks.append(TaskDTO(name=task["name"], description=task["description"]))


# Ping
@app.get("/api/ping")
def ping():
    return db_engine.check_db_connection()


@app.get("/api/")
def root():
    return "Welcome to CompUGE API! This API is private and not intended for public use."


@app.get("/")
def root():
    return "Welcome to CompUGE backend! Please visit /api/ for the API documentation."


@app.get("/api/leaderboards")
def leaderboards():
    lbs = lb_service.get_leaderboards()
    return lbs


@app.get("/api/leaderboard/{task}/{dataset}")
def leaderboard(task: str, dataset: str):
    return lb_service.get_leaderboard(task, dataset)


@app.get("/api/submissions")
def submissions():
    subs = sub_service.get_submissions()
    return subs


@app.get("/api/tasks")
def get_tasks():
    return tasks


@app.get("/api/datasets")
def get_datasets():
    return ds_service.get_dataset_dtos()


@app.get("/api/dataset/{task}/{dataset}")
def get_dataset(task: str, dataset: str):
    return ds_service.get_dataset_dto(task, dataset)


@app.get("/api/datasets/{task}")
def get_datasets_per_task(task: str):
    return ds_service.get_dataset_dtos_per_task(task)


@app.post("/api/submission/{task}/{dataset}")
def submit(task: str, dataset: str, submission_dict: dict):
    response = sub_service.submit_solution(task, dataset, submission_dict)
    if response == "Submission failed due to mismatch in the test set":
        return fastapi.Response(content=response, status_code=400)
    elif response == "Submission failed due to persistence error":
        return fastapi.Response(content=response, status_code=500)
    return response


# ====================== Authentication ==========================

# The OAuth2PasswordBearer instance is needed for the token authentication flow
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/token")


def get_current_user(token: str = Depends(oauth2_scheme)):
    """Dependency that extracts and verifies the current user from the token."""
    payload = auth_service.decode_token(token)
    user = payload.get("sub")
    if user != "admin":  # The username is always "admin"
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )
    return user  # This will return "admin"


@app.post("/api/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    if not auth_service.authenticate_password(form_data.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=30)
    access_token = auth_service.create_access_token(data={"sub": "admin"},
                                                    expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/api/controlPanelSubmissions")
async def control_panel_submissions(current_user: str = Depends(get_current_user)):
    return sub_service.get_submissions_with_their_leaderboard_entries()


@app.put("/api/controlPanelSubmission/{sub_id}")
async def force_update_submission(sub_id: int,
                                  submission: dict,
                                  current_user: str = Depends(get_current_user)):
    return sub_service.force_update_submission(sub_id, submission)


@app.delete("/api/controlPanelSubmission/{sub_id}")
async def delete_submission(sub_id: int, current_user: str = Depends(get_current_user)):
    return sub_service.delete_submission(sub_id)
