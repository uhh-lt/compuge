import json
from io import StringIO

import pandas as pd

from dtos import DatasetDTO


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
datasetsDTOs = []

# in datasets-metadata.json we have a list of datasets in the key "datasets"
with open("./datasets/datasets-metadata.json", "r") as metadata_file:
    metadata_dict = json.load(metadata_file)
    for dataset in metadata_dict["datasets"]:
        datasets_metadata.append(
            DatasetMetadata(task=dataset["task"], name=dataset["name"], description=dataset["description"],
                            link=dataset["link"], folder=dataset["folder"], paper=dataset["paper"],
                            paper_link=dataset["paper_link"]))

# train and test data are saved as csv files in the dataset's folder
for dataset in datasets_metadata:
    # add exception handling to avoid errors when the file is not found
    train = None
    test = None
    val = None
    # try to read the data
    try:
        train = pd.read_csv(f"./datasets/{dataset.folder}/train.csv")
    except FileNotFoundError:
        print(f"Train or test set not found for dataset {dataset.name}")

    try:
        test = pd.read_csv(f"./datasets/{dataset.folder}/test.csv")
    except FileNotFoundError:
        print(f"Test set not found for dataset {dataset.name}")

    try:
        val = pd.read_csv(f"./datasets/{dataset.folder}/val.csv")
    except FileNotFoundError:
        print(f"Validation set not found for dataset {dataset.name}")

    # remove the 2nd columns from test
    test = test.drop(test.columns[1], axis=1)
    # convert the dataframes to lists of strings including the headers
    if train is not None:
        train = train.to_string(index=False).split("\n")
    if test is not None:
        test = test.to_string(index=False).split("\n")
    if val is not None:
        val = val.to_string(index=False).split("\n")

    datasetsDTOs.append(DatasetDTO(task=dataset.task, name=dataset.name, description=dataset.description,
                                   link=dataset.link, train=train, test=test, val=val, paper=dataset.paper,
                                   paper_link=dataset.paper_link))


# ======================================================================================================================
# Methods
# ======================================================================================================================


def get_dataset_dto(task: str, dataset: str):
    for dataset_dto in datasetsDTOs:
        if dataset_dto.task == task and dataset_dto.name == dataset:
            return dataset_dto
    return None


def get_dataset_dtos_per_task(task: str):
    return [dataset for dataset in datasetsDTOs if dataset.task == task]


def get_dataset_dtos():
    return datasetsDTOs


def get_dataset_split(task: str, dataset: str):
    # get the dataset metadata
    dataset_metadata = None
    for dset in datasets_metadata:
        if dset.task == task and dset.name == dataset:
            dataset_metadata = dset
            break
    if dataset_metadata is None:
        return None, None
    # read the train and test data
    try:
        train = pd.read_csv(f"./datasets/{dataset_metadata.folder}/train.csv")
        test = pd.read_csv(f"./datasets/{dataset_metadata.folder}/test.csv")
        return train, test
    except FileNotFoundError:
        print(f"File not found for dataset {dataset_metadata.name}")
        return None, None


def get_test_labels(task: str, dataset: str):
    # get the dataset metadata
    dataset_metadata = None
    for dset in datasets_metadata:
        if dset.task == task and dset.name == dataset:
            dataset_metadata = dset
            break
    if dataset_metadata is None:
        return None
    # read the train data
    try:
        test = pd.read_csv(f"./datasets/{dataset_metadata.folder}/test.csv")
        return test[test.columns[1]].tolist()
    except FileNotFoundError:
        print(f"File not found for dataset {dataset_metadata.name}")
        return None


def get_labels_from_csv(csv_str: str):
    try:
        df = pd.read_csv(StringIO(csv_str))
        return df[df.columns[1]].tolist()
    except Exception as e:
        print(f"Error: {e}")
        return None
