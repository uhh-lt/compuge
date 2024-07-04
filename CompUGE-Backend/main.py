import fastapi
import services.submissions_service as sub_service
import services.leaderboards_service as lb_service
import services.dataset_service as ds_service
from fastapi.middleware.cors import CORSMiddleware
from dtos import SubmissionDTO, LeaderboardDTO, TaskDTO, DatasetDTO
import json

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
    return lb_service.get_leaderboards()


@app.get("/api/leaderboard/{task}/{dataset}")
def leaderboard(task: str, dataset: str):
    return lb_service.get_leaderboard(task, dataset)


@app.get("/api/submissions")
def submissions():
    return sub_service.get_submissions()


@app.post("/api/submission/{task}/{dataset}")
def submit(task: str, dataset: str, submission_dict: dict):
    return sub_service.submit_solution(task, dataset, submission_dict)


@app.get("/api/tasks")
def get_tasks():
    return tasks


@app.get("/api/datasets")
def get_datasets():
    return ds_service.get_datasets()


@app.get("/api/dataset/{task}/{dataset}")
def get_dataset(task: str, dataset: str):
    return ds_service.get_dataset(task, dataset)


@app.get("/api/datasets/{task}")
def get_datasets_per_task(task: str):
    return ds_service.get_datasets_per_task(task)