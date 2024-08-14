import json
from datetime import timedelta

import fastapi
from fastapi import Depends, HTTPException, status, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

import repositories.db_engine as db_engine
import services.authentication_service as auth_service
import services.dataset_service as ds_service
import services.submissions_service as sub_service
from dtos import TaskDTO

app = fastapi.FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load tasks from JSON file
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
    return sub_service.get_leaderboards()


@app.get("/api/submissions")
def submissions():
    return sub_service.get_submissions()


@app.get("/api/tasks")
def get_tasks():
    return tasks


@app.get("/api/datasets")
def get_datasets():
    return ds_service.get_dataset_dtos()


@app.post("/api/submission/{task}/{dataset}")
def submit(task: str, dataset: str, submission_dict: dict = Body(...)):
    # Input validation
    if not submission_dict.get("modelName") or not submission_dict.get("modelLink"):
        raise HTTPException(status_code=400, detail="Invalid input: 'modelName' and 'modelLink' are required fields.")
    if not submission_dict.get("teamName") or not submission_dict.get("contactEmail"):
        raise HTTPException(status_code=400, detail="Invalid input: 'teamName' and 'contactEmail' are required fields.")
    if not submission_dict.get("fileContent"):
        raise HTTPException(status_code=400, detail="Invalid input: 'fileContent' is required.")
    if not submission_dict.get("isPublic"):
        raise HTTPException(status_code=400, detail="Invalid input: 'isPublic' is required.")

    ret = sub_service.submit_solution(task, dataset, submission_dict)
    switcher = {
        "Submission successful": status.HTTP_201_CREATED,
        "Submission rejected": status.HTTP_400_BAD_REQUEST,
        "Error getting test labels": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "Error getting predictions from CSV": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "Error persisting accepted submission data": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "Error persisting rejected submission data": status.HTTP_500_INTERNAL_SERVER_ERROR,
    }
    return {"message": ret}, switcher.get(ret, status.HTTP_500_INTERNAL_SERVER_ERROR)


# ====================== Authentication ==========================

# The OAuth2PasswordBearer instance is needed for the token authentication flow
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/token")


def get_current_user(token: str = Depends(oauth2_scheme)):
    """Dependency that extracts and verifies the current user from the token."""
    try:
        payload = auth_service.decode_token(token)
        user = payload.get("sub")
        if user != "admin":  # The username is always "admin"
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
            )
        return user  # This will return "admin"
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token or credentials",
        )


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


@app.get("/api/control-panel-submissions")
async def control_panel_submissions(current_user: str = Depends(get_current_user)):
    return sub_service.get_submission_for_control_panel()


@app.put("/api/submission/{sub_id}")
async def update_submission(sub_id: int,
                            submission: dict,
                            current_user: str = Depends(get_current_user)):
    # Validate submission update input
    if not submission:
        raise HTTPException(status_code=400, detail="Invalid input: update data is required.")
    try:
        ret = sub_service.update_submission(sub_id, submission)
        switcher = {
            "Submission updated successfully": status.HTTP_200_OK,
            "Submission not found": status.HTTP_404_NOT_FOUND,
            "Error updating submission data in the database": status.HTTP_500_INTERNAL_SERVER_ERROR,
        }
        return {"message": ret}, switcher.get(ret, status.HTTP_500_INTERNAL_SERVER_ERROR)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred in update_submission")


@app.delete("/api/submission/{sub_id}")
async def delete_submission(sub_id: int, current_user: str = Depends(get_current_user)):
    ret = sub_service.delete_submission(sub_id)
    switcher = {
        "Submission deleted successfully": status.HTTP_200_OK,
        "Submission not found": status.HTTP_404_NOT_FOUND,
        "Error deleting submission data from the database": status.HTTP_500_INTERNAL_SERVER_ERROR,
    }
    return {"message": ret}, switcher.get(ret, status.HTTP_500_INTERNAL_SERVER_ERROR)
