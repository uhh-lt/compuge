import json

import fastapi
from fastapi.middleware.cors import CORSMiddleware

import services.dataset_service as ds_service
import services.leaderboards_service as lb_service
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

# load tasks from tasks.json under key "tasks"
tasks = []
with open("tasks.json", "r") as tasks_file:
    tasks_dict = json.load(tasks_file)
    for task in tasks_dict["tasks"]:
        tasks.append(TaskDTO(name=task["name"], description=task["description"]))


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
