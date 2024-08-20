import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import numpy as np


def load_data(folder_path):
    """
    Loads data from the folder and prepares a DatasetDict for train, validation, and test.
    Expects folder structure:
    - folder_path/
      - train.csv
      - validate.csv
      - test.csv
    Each CSV should have at least two columns: 'text' and 'label'.
    """
    dataset_dict = {}
    for split in ['train', 'validate', 'test']:
        try:
            csv_path = os.path.join(folder_path, f"{split}.csv")
            data = pd.read_csv(csv_path)
            dataset = Dataset.from_pandas(data)
            dataset_dict[split] = dataset
        except FileNotFoundError:
            print(f"File not found: {csv_path}")

    if 'validate' not in dataset_dict:
        # If validation data is not available split the training data and convert it to a dataset object
        train_data = pd.read_csv(os.path.join(folder_path, "train.csv"))
        val_data = train_data.sample(frac=0.1, random_state=42)
        train_data = train_data.drop(val_data.index)
        dataset_dict['train'] = Dataset.from_pandas(train_data)
        dataset_dict['validate'] = Dataset.from_pandas(val_data)

    return DatasetDict({
        'train': dataset_dict['train'],
        'validate': dataset_dict['validate'],
        'test': dataset_dict['test'],
    })


def save_test_results(results_folder, test_dataset, predictions):
    """
    Save the test dataset with predictions to a CSV file.
    """
    # Convert predictions from logits to class labels
    pred_labels = np.argmax(predictions, axis=1)

    # Load the original test data
    test_df = pd.DataFrame(test_dataset)

    # Add the predictions to the test DataFrame
    test_df['predictions'] = pred_labels

    # Save the DataFrame to a CSV file in the results folder
    results_path = os.path.join(results_folder, "test_results.csv")
    test_df.to_csv(results_path, index=False)
    print(f"Test results saved to {results_path}")


def main(folder_path, model_name, results_folder):
    # Load the dataset
    datasets = load_data(folder_path)

    # Load the tokenizer and model from Hugging Face model hub
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Adjust num_labels as needed

    # Tokenize the datasets
    def tokenize_function(examples):
        return tokenizer(examples['question'], padding="max_length", truncation=True)

    tokenized_datasets = datasets.map(tokenize_function, batched=True)

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",  # This triggers evaluation at the end of every epoch
        save_strategy="epoch",  # Ensure the model is saved at the end of every epoch
        logging_dir='./logs',
        logging_steps=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True  # Load the best model at the end of training
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validate"],
    )

    # Train the model
    trainer.train()

    # Evaluate on the test set
    test_results = trainer.predict(test_dataset=tokenized_datasets["test"])

    # Save test results with predictions
    save_test_results(results_folder, datasets["test"], test_results.predictions)


if __name__ == "__main__":
    # Provide the folder path, model name, and results folder here
    folder_path = "../../Splits/webis_comparative_questions_2020_qi"
    model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    results_folder = "./testing_results"

    # Ensure the results folder exists
    os.makedirs(results_folder, exist_ok=True)

    main(folder_path, model_name, results_folder)
