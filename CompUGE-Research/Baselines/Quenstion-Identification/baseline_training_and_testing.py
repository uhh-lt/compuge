import json
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def load_data(train_folder, test_folder):
    dataset_dict = {}
    for split in ['train', 'validate']:
        try:
            csv_path = os.path.join(train_folder, f"{split}.csv")
            data = pd.read_csv(csv_path)
            dataset = Dataset.from_pandas(data)
            dataset_dict[split] = dataset
        except FileNotFoundError:
            print(f"File not found: {csv_path}")

    if 'train' not in dataset_dict:
        return None

    if 'validate' not in dataset_dict:
        train_data = pd.read_csv(os.path.join(train_folder, "train.csv"))
        val_data = train_data.sample(frac=0.1, random_state=42)
        train_data = train_data.drop(val_data.index)
        dataset_dict['train'] = Dataset.from_pandas(train_data)
        dataset_dict['validate'] = Dataset.from_pandas(val_data)

    test_csv_path = os.path.join(test_folder, "test.csv")
    test_data = pd.read_csv(test_csv_path)
    dataset_dict['test'] = Dataset.from_pandas(test_data)

    return DatasetDict({
        'train': dataset_dict['train'],
        'validate': dataset_dict['validate'],
        'test': dataset_dict['test'],
    })


def save_test_results(results_folder, test_dataset, predictions, train_folder_name, test_folder_name, model_name):
    # Generate prediction labels
    pred_labels = np.argmax(predictions, axis=1)
    test_df = pd.DataFrame(test_dataset)
    test_df['predictions'] = pred_labels

    # if model_name is a path, extract the model name from the path to use in the results file name
    if '/' in model_name:
        model_name = model_name.split('/')[-1]

    results_file_name = f"{train_folder_name}_SEP_{test_folder_name}_SEP_{model_name}_test_results.csv"
    results_path = os.path.join(results_folder, results_file_name)

    print(f"Saving test results to {results_path}")
    os.makedirs(results_folder, exist_ok=True)

    test_df.to_csv(results_path, index=False)
    print(f"Test results saved to {results_path}")


def save_metrics(results_folder, train_folder_name, test_folder_name, model_name, metrics):
    metrics_file_path = os.path.join(results_folder, "metrics.csv")
    metrics_data = {
        'training on': [train_folder_name],
        'tested on': [test_folder_name],
        'model': [model_name],
        'accuracy': [metrics['test_accuracy']],
        'precision': [metrics['test_precision']],
        'recall': [metrics['test_recall']],
        'f1': [metrics['test_f1']]
    }
    metrics_df = pd.DataFrame(metrics_data)

    if not os.path.exists(metrics_file_path):
        metrics_df.to_csv(metrics_file_path, index=False)
    else:
        metrics_df.to_csv(metrics_file_path, mode='a', header=False, index=False)

    print(f"Metrics saved to {metrics_file_path}")


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main(train_folder, test_folder, model_name, results_folder):
    os.makedirs(results_folder, exist_ok=True)
    datasets = load_data(train_folder, test_folder)
    if datasets is None:
        print(f"No training data found in folder {train_folder}. Exiting.")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def tokenize_function(examples):
        return tokenizer(examples['question'], padding="max_length", truncation=True)

    tokenized_datasets = datasets.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validate"],
        compute_metrics=compute_metrics
    )

    trainer.train()
    test_results = trainer.predict(test_dataset=tokenized_datasets["test"])

    print(f"Test Accuracy: {test_results.metrics['test_accuracy']:.4f}")
    print(f"Test Precision: {test_results.metrics['test_precision']:.4f}")
    print(f"Test Recall: {test_results.metrics['test_recall']:.4f}")
    print(f"Test F1: {test_results.metrics['test_f1']:.4f}")

    train_folder_name = os.path.basename(os.path.normpath(train_folder))
    test_folder_name = os.path.basename(os.path.normpath(test_folder))

    save_test_results(results_folder, datasets["test"], test_results.predictions, train_folder_name, test_folder_name,
                      model_name)
    save_metrics(results_folder, train_folder_name, test_folder_name, model_name, test_results.metrics)

    print("=========================================")
    print("=========================================")
    print(f"Finished testing {model_name} trained on {train_folder_name} and tested on {test_folder_name}.")
    print("=========================================")
    print("=========================================")


# Checkpoint file path
checkpoint_file = "checkpoint.json"

# Function to load checkpoint data
def load_checkpoint():
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            return json.load(f)
    else:
        return {"model_index": 0, "dataset1_index": 0, "dataset2_index": 0}

# Function to save checkpoint data
def save_checkpoint(model_index, dataset1_index, dataset2_index):
    checkpoint_data = {
        "model_index": model_index,
        "dataset1_index": dataset1_index,
        "dataset2_index": dataset2_index
    }
    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint_data, f)

if __name__ == "__main__":
    print("Starting training and testing...")

    with open("../../datasets-metadata.json") as f:
        print("Loading datasets metadata...")
        datasets_metadata = json.load(f)
        print("Datasets metadata loaded.")

        models = [
            "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        ]

        os.makedirs("./testing_results", exist_ok=True)

        print("Loading checkpoint data...")
        checkpoint = load_checkpoint()
        model_index = checkpoint["model_index"]
        dataset1_index = checkpoint["dataset1_index"]
        dataset2_index = checkpoint["dataset2_index"]
        print("Checkpoint data loaded.")

        print("Starting training and testing loop...")
        for m_idx, model in enumerate(models):
            if m_idx < model_index:
                continue  # Skip already processed models

            # Iterate over first dataset
            for d1_idx, dataset1 in enumerate(datasets_metadata["datasets"]):
                if m_idx == model_index and d1_idx < dataset1_index:
                    continue  # Skip already processed datasets
                if dataset1["task"] != "Question Identification":
                    continue

                # Iterate over second dataset
                for d2_idx, dataset2 in enumerate(datasets_metadata["datasets"]):
                    if m_idx == model_index and d1_idx == dataset1_index and d2_idx < dataset2_index:
                        continue  # Skip already processed comparisons
                    if dataset2["task"] != "Question Identification":
                        continue

                    # Run your main function here
                    main(
                        f"../../Splits/{dataset1['folder']}",
                        f"../../Splits/{dataset2['folder']}",
                        model,
                        f"./testing_results/{model}"
                    )

                    # Save the current progress to the checkpoint file
                    save_checkpoint(m_idx, d1_idx, d2_idx)

        # After all processing, remove the checkpoint file
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
