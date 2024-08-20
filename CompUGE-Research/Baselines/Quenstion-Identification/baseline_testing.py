import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
import pandas as pd

def load_test_data(folder_path):
    """
    Loads test data from the folder and prepares a Dataset.
    Expects folder structure:
    - folder_path/
      - test.csv
    The CSV should have at least two columns: 'text' and 'label'.
    """
    test_csv_path = os.path.join(folder_path, "test.csv")
    test_data = pd.read_csv(test_csv_path)
    test_dataset = Dataset.from_pandas(test_data)

    return test_dataset

def save_test_results(results_folder, test_dataset, predictions):
    """
    Save the test dataset with predictions to a CSV file.
    """
    # Convert predictions from logits to class labels
    pred_labels = np.argmax(predictions, axis=1)

    # Convert test_dataset to pandas DataFrame
    test_df = pd.DataFrame(test_dataset)

    # Add the predictions to the test DataFrame
    test_df['predictions'] = pred_labels

    # Save the DataFrame to a CSV file in the results folder
    results_path = os.path.join(results_folder, "test_results.csv")
    test_df.to_csv(results_path, index=False)
    print(f"Test results saved to {results_path}")


def main(folder_path, model_name, results_folder):
    # Load the test dataset
    test_dataset = load_test_data(folder_path)

    # Load the tokenizer and model from Hugging Face model hub
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Adjust num_labels as needed

    # Tokenize the test dataset
    def tokenize_function(examples):
        return tokenizer(examples['question'], padding="max_length", truncation=True)

    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Define the training arguments (only for evaluation)
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=8,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
    )

    # Evaluate on the test set
    test_results = trainer.predict(test_dataset=tokenized_test_dataset)

    # Save test results with predictions
    save_test_results(results_folder, test_dataset, test_results.predictions)

if __name__ == "__main__":
    # Provide the folder path, model name, and results folder here
    folder_path = "../../Splits/webis_comparative_questions_2020_qi"
    model_name = "uhhlt/binary-compqa-classifier"
    results_folder = "./results"

    # Ensure the results folder exists
    os.makedirs(results_folder, exist_ok=True)

    main(folder_path, model_name, results_folder)