import json
from dtos import DatasetDTO
import pandas as pd

class DatasetMetadata:
    def __init__(self, task: str, name: str, description: str, link: str, folder: str, paper: str, paper_link: str):
        self.task = task
        self.name = name
        self.description = description
        self.link = link
        self.folder = folder
        self.paper = paper
        self.paper_link = paper_link



datasets_metadata = []
# in datasets-metadata.json we have a list of datasets in the key "datasets"
with open("./datasets/datasets-metadata.json", "r") as metadata_file:
    metadata_dict = json.load(metadata_file)
    for dataset in metadata_dict["datasets"]:
        datasets_metadata.append(
            DatasetMetadata(task=dataset["task"], name=dataset["name"], description=dataset["description"],
                            link=dataset["link"], folder=dataset["folder"], paper=dataset["paper"], paper_link=dataset["paper_link"]))

datasetsDTOs = []
# train and test data are saved as csv files in the dataset's folder
for dataset in datasets_metadata:
    # add exception handling to avoid errors when the file is not found
    train = []
    test = []
    try :
        train = pd.read_csv(f"./datasets/{dataset.folder}/train.csv")
        test = pd.read_csv(f"./datasets/{dataset.folder}/test.csv")
        # remove the 2nd columns from test
        test = test.drop(test.columns[1], axis=1)
        # convert the dataframes to lists of strings including the headers
        train = train.to_string(index=False).split("\n")
        test = test.to_string(index=False).split("\n")
    except FileNotFoundError:
        print(f"File not found for dataset {dataset.name}")
    finally:
        datasetsDTOs.append(DatasetDTO(task=dataset.task, name=dataset.name, description=dataset.description,
                                       link=dataset.link, train=train, test=test, paper=dataset.paper, paper_link=dataset.paper_link))

def get_dataset(task: str, dataset: str):
    for dataset_dto in datasetsDTOs:
        if dataset_dto.task == task and dataset_dto.name == dataset:
            return dataset
    return None


def get_datasets_per_task(task: str):
    return [dataset for dataset in datasetsDTOs if dataset.task == task]


def get_datasets():
    return datasetsDTOs

