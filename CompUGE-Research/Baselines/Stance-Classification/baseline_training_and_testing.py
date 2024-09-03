import json
import os

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


def load_data(train_folder, test_folder=None):
    dataset_dict = {}

    for split in ['train', 'val']:
        csv_path = os.path.join(train_folder, f"{split}.csv")
        if os.path.exists(csv_path):
            data = pd.read_csv(csv_path)
            dataset_dict[split] = Dataset.from_pandas(data)

    if 'train' not in dataset_dict:
        return None

    if 'val' not in dataset_dict:
        train_data = pd.read_csv(os.path.join(train_folder, "train.csv"))
        val_data = train_data.sample(frac=0.1, random_state=42)
        dataset_dict['train'] = Dataset.from_pandas(train_data.drop(val_data.index))
        dataset_dict['val'] = Dataset.from_pandas(val_data)

    if test_folder:
        test_data = pd.read_csv(os.path.join(test_folder, "test.csv"))
        dataset_dict['test'] = Dataset.from_pandas(test_data)

    return DatasetDict(dataset_dict)


def save_test_results(results_folder, test_dataset, predictions, train_folder_name, test_folder_name, model_name):
    pred_labels = np.argmax(predictions, axis=1)
    test_df = pd.DataFrame(test_dataset)
    test_df['predictions'] = pred_labels

    model_name = model_name.split('/')[0]
    results_file_name = f"{train_folder_name}_{test_folder_name}_{model_name}_test_results.csv"
    results_path = os.path.join(results_folder, results_file_name)

    os.makedirs(results_folder, exist_ok=True)
    test_df.to_csv(results_path, index=False)
    print(f"Test results saved to {results_path}")


def save_metrics(results_folder, train_folder_name, test_folder_name, model_name, metrics):
    metrics_data = {
        'training on': train_folder_name,
        'tested on': test_folder_name,
        'model': model_name.split('/')[0],
        'accuracy': metrics['test_accuracy'],
        'precision': metrics['test_precision'],
        'recall': metrics['test_recall'],
        'f1': metrics['test_f1']
    }
    metrics_file_path = os.path.join(results_folder, "metrics.csv")
    metrics_df = pd.DataFrame([metrics_data])

    metrics_df.to_csv(metrics_file_path, mode='a', header=not os.path.exists(metrics_file_path), index=False)
    print(f"Metrics saved to {metrics_file_path}")


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    accuracy = accuracy_score(p.label_ids, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='weighted', zero_division=0)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def save_checkpoint(checkpoint_file, current_index):
    with open(checkpoint_file, 'w') as f:
        json.dump({"current_index": current_index}, f)
    print(f"Checkpoint saved at index {current_index}")


def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            return checkpoint.get("current_index", 0)
    return 0


def remove_checkpoint_file(checkpoint_file):
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"Checkpoint file {checkpoint_file} removed.")


def main(train_folder, test_folders, model_name, results_folder):
    datasets = load_data(train_folder)
    if datasets is None:
        print(f"No training data found in folder {train_folder}. Exiting.")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    def tokenize_function(examples):
        return tokenizer(
            examples['sentence'],
            padding="max_length",  # Ensure all sequences are padded to the same length
            truncation=True,  # Truncate sequences that are longer than max_length
            max_length=342  # Set a fixed max length (you can adjust this value)
        )

    tokenized_datasets = datasets.map(tokenize_function, batched=True)

    # Define Training Arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",  # Evaluate at the end of every epoch
        save_strategy="epoch",  # Save checkpoint at the end of every epoch
        logging_dir='./logs',
        logging_steps=10,
        per_device_train_batch_size=8,  # Use batch size of 16 for training
        per_device_eval_batch_size=8,  # Use batch size of 16 for evaluation
        num_train_epochs=13,  # Train for 13 epochs
        weight_decay=0.1,  # Weight decay for AdamW optimizer
        learning_rate=3e-5,  # Learning rate for AdamW optimizer
        load_best_model_at_end=True,  # Load the best model when finished
        metric_for_best_model="f1",  # Use F1 score to select the best model
        warmup_steps=100,  # Warmup steps for learning rate scheduler
        lr_scheduler_type="cosine",  # Use cosine learning rate scheduler
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Evaluate on test datasets
    for test_folder in test_folders:
        test_dataset = load_data(train_folder, test_folder)['test']
        tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)
        test_results = trainer.predict(test_dataset=tokenized_test_dataset)

        print(f"Test results for {model_name.split('/')[0]} trained on {train_folder} and tested on {test_folder}:")
        print(f"Test Accuracy: {test_results.metrics['test_accuracy']:.4f}")
        print(f"Test Precision: {test_results.metrics['test_precision']:.4f}")
        print(f"Test Recall: {test_results.metrics['test_recall']:.4f}")
        print(f"Test F1: {test_results.metrics['test_f1']:.4f}")

        train_folder_name = os.path.basename(train_folder)
        test_folder_name = os.path.basename(test_folder)

        save_test_results(results_folder, test_dataset, test_results.predictions, train_folder_name, test_folder_name,
                          model_name)
        save_metrics(results_folder, train_folder_name, test_folder_name, model_name, test_results.metrics)


if __name__ == "__main__":
    checkpoint_file = "checkpoint.json"
    start_index = load_checkpoint(checkpoint_file)

    with open("../../datasets-metadata.json") as f:
        datasets_metadata = json.load(f)
        model_name = "FacebookAI/roberta-base"
        results_folder = f"./testing_results/{model_name.split('/')[0]}"
        os.makedirs(results_folder, exist_ok=True)

        print(
            "---------------------------------------------------------------------------------------------------------")
        print("total datasets: ")
        print(len(datasets_metadata["datasets"]))
        print("total datasets with task Stance Classification: ")
        print(len([dataset for dataset in datasets_metadata["datasets"] if dataset["task"] == "Stance Classification"]))
        print(
            "---------------------------------------------------------------------------------------------------------")

        for i, dataset1 in enumerate(datasets_metadata["datasets"]):
            if i < start_index:
                continue

            print("Training on: ", dataset1["folder"])
            if dataset1["task"] != "Stance Classification":
                continue

            test_folders = [f"../../Splits/{dataset2['folder']}" for dataset2 in datasets_metadata["datasets"]
                            if dataset2["task"] == "Stance Classification" and "merged" not in dataset2["folder"]]

            main(f"../../Splits/{dataset1['folder']}", test_folders, model_name, results_folder)
            save_checkpoint(checkpoint_file, i + 1)

            print(
                "---------------------------------------------------------------------------------------------------------")
            print(f"Finished training and testing {model_name} on {dataset1['folder']}.")
            print(
                "---------------------------------------------------------------------------------------------------------")

        # Remove checkpoint file after all loops are finished
        remove_checkpoint_file(checkpoint_file)
        print("Finished all training and testing.")
