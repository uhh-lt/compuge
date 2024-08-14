import json
import logging
from io import StringIO

import pandas as pd

from dtos import DatasetDTO

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ======================================================================================================================
# Load datasets
# ======================================================================================================================

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

# Load datasets metadata
try:
    with open("./datasets/datasets-metadata.json", "r") as metadata_file:
        metadata_dict = json.load(metadata_file)
        for dataset in metadata_dict["datasets"]:
            datasets_metadata.append(
                DatasetMetadata(task=dataset["task"], name=dataset["name"], description=dataset["description"],
                                link=dataset["link"], folder=dataset["folder"], paper=dataset["paper"],
                                paper_link=dataset["paper_link"]))
except FileNotFoundError as e:
    logger.error(f"Metadata file not found: {e}")
except json.JSONDecodeError as e:
    logger.error(f"Error decoding JSON from metadata file: {e}")
except Exception as e:
    logger.error(f"Unexpected error loading datasets metadata: {e}")

# Load datasets
for dataset in datasets_metadata:
    train = None
    test = None
    val = None

    try:
        train = pd.read_csv(f"./datasets/{dataset.folder}/train.csv")
    except FileNotFoundError:
        logger.warning(f"Train set not found for dataset {dataset.name}")
    except Exception as e:
        logger.error(f"Unexpected error reading train set for dataset {dataset.name}: {e}")

    try:
        test = pd.read_csv(f"./datasets/{dataset.folder}/test.csv")
    except FileNotFoundError:
        logger.warning(f"Test set not found for dataset {dataset.name}")
    except Exception as e:
        logger.error(f"Unexpected error reading test set for dataset {dataset.name}: {e}")

    try:
        val = pd.read_csv(f"./datasets/{dataset.folder}/val.csv")
    except FileNotFoundError:
        logger.warning(f"Validation set not found for dataset {dataset.name}")
    except Exception as e:
        logger.error(f"Unexpected error reading validation set for dataset {dataset.name}: {e}")

    if train is not None:
        train = train.to_string(index=False).split("\n")
    if test is not None:
        test = test.drop(test.columns[1], axis=1)
        test = test.to_string(index=False).split("\n")
    if val is not None:
        val = val.to_string(index=False).split("\n")

    datasetsDTOs.append(DatasetDTO(task=dataset.task, name=dataset.name, description=dataset.description,
                                   link=dataset.link, train=train, test=test, val=val, paper=dataset.paper,
                                   paper_link=dataset.paper_link))


# ======================================================================================================================
# Methods
# ======================================================================================================================


def get_dataset_dtos():
    """
    Get all datasets DTOs
    :return: a list of datasets DTOs
    """
    return datasetsDTOs


def get_test_labels(task: str, dataset: str):
    """
    Get the test labels for a dataset belonging to a task
    :param task: the task name
    :param dataset: the dataset name
    :return: a list of test labels or None if an error occurred
    """
    dataset_metadata = next((dset for dset in datasets_metadata if dset.task == task and dset.name == dataset), None)

    if dataset_metadata is None:
        logger.error(f"Dataset metadata not found: Task={task}, Dataset={dataset}")
        return None

    try:
        test = pd.read_csv(f"./datasets/{dataset_metadata.folder}/test.csv")
        return test[test.columns[1]].tolist()
    except FileNotFoundError:
        logger.error(f"Test file not found for dataset: {dataset_metadata.name}")
        return None
    except Exception as e:
        logger.error(f"Error reading test labels for dataset {dataset_metadata.name}: {e}")
        return None


def get_predictions_from_csv(csv_str: str):
    """
    Extract the predictions from a CSV string
    :param csv_str: the CSV string
    :return: a list of predictions or None if an error occurred
    """
    try:
        df = pd.read_csv(StringIO(csv_str))
    except pd.errors.EmptyDataError:
        logger.error("The provided CSV string is empty")
        return None
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV string: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error processing CSV string: {e}")
        return None

    # check if the CSV has a predictions column
    if "predictions" not in df.columns:
        logger.error("No 'predictions' column found in the CSV")
        return None
    return df["predictions"].tolist()
