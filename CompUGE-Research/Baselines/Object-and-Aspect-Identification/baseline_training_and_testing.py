import json
import os
import pandas as pd
import transformers
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    AutoModelForTokenClassification,
)
import numpy as np
import evaluate


def compute_metrics(eval_preds):
    predictions, labels = eval_preds.predictions, eval_preds.label_ids

    # Convert predictions to label indices by taking the argmax across the last dimension
    predictions = np.argmax(predictions, axis=-1)

    # Filter out special tokens (-100) and flatten lists to compute metrics
    true_labels = [[l for l in label if l != -100] for label in labels]
    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Load the evaluation metric (seqeval is often used for sequence labeling tasks)
    metric = evaluate.load("seqeval")

    # Compute the metrics with true labels and predictions
    results = metric.compute(predictions=true_predictions, references=true_labels)

    # Flatten the dictionary to make the results easier to use
    results_unfolded = {}
    for key, value in results.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                results_unfolded[key + "_" + subkey] = subvalue
        else:
            results_unfolded[key] = value

    return results_unfolded


def model_init_helper(model_name):
    def model_init():
        # Directly load the model without label mappings
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            ignore_mismatched_sizes=True,
        )
        print(f"{model.config.num_labels = }")
        return model

    return model_init


def tokenize_and_align_labels(examples, one_label_per_word=True, **kwargs):
    tokenizer = kwargs["tokenizer"]

    tokenized_inputs = tokenizer(
        examples["words"], truncation=True, is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        current_word = None
        new_labels = []
        for word_id in word_ids:
            if word_id is None:
                new_labels.append(-100)
            elif word_id != current_word:
                current_word = word_id
                new_labels.append(label[word_id])
            else:
                lab = label[word_id]
                if lab % 2 == 1:
                    lab += 1
                new_labels.append(-100 if one_label_per_word else lab)
        labels.append(new_labels)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


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


def save_metrics(results_folder, train_folder_name, test_folder_name, model_name, metrics):
    # Prepare the metrics data in a structured format
    metrics_data = {
        'training on': train_folder_name,
        'tested on': test_folder_name,
        'model': model_name.split('/')[0],
        'accuracy': metrics['test_accuracy'],
        'precision': metrics['test_precision'],
        'recall': metrics['test_recall'],
        'f1': metrics['test_f1']
    }

    # Create or append the metrics data to a CSV file
    metrics_file_path = os.path.join(results_folder, "metrics.csv")
    metrics_df = pd.DataFrame([metrics_data])

    # Save the metrics DataFrame to a CSV file
    metrics_df.to_csv(metrics_file_path, mode='a', header=not os.path.exists(metrics_file_path), index=False)
    print(f"Metrics saved to {metrics_file_path}")


def save_test_results(results_folder, test_dataset, predictions, train_folder_name, test_folder_name, model_name):
    # Convert the model's output logits to predicted label IDs
    pred_labels = np.argmax(predictions.predictions, axis=-1)

    # Create a DataFrame with the test dataset
    test_df = pd.DataFrame(test_dataset)

    # Add the predicted labels as a new column
    test_df['predictions'] = pred_labels

    # Create a file name and path to save the test results
    model_name = model_name.split('/')[0]  # Get base model name
    results_file_name = f"{train_folder_name}_{test_folder_name}_{model_name}_test_results.csv"
    results_path = os.path.join(results_folder, results_file_name)

    # Ensure the results folder exists
    os.makedirs(results_folder, exist_ok=True)

    # Save the test DataFrame to a CSV file
    test_df.to_csv(results_path, index=False)
    print(f"Test results saved to {results_path}")


def train_and_test_on_datasets(train_folder, test_folders, results_folder, model_name):
    datasets = load_data(train_folder)
    if datasets is None:
        print(f"No training data found in folder {train_folder}. Exiting.")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=64, add_prefix_space=True)

    tokenized_datasets = datasets.map(
        tokenize_and_align_labels,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        learning_rate=2e-5,
        seed=42,
    )

    trainer = Trainer(
        model_init=model_init_helper(model_name),
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Training")
    trainer.train()

    for test_folder in test_folders:
        test_dataset = load_data(train_folder, test_folder)['test']
        tokenized_test_dataset = test_dataset.map(lambda examples: tokenizer(examples['words'], truncation=True, padding="max_length"), batched=True)
        test_results = trainer.predict(test_dataset=tokenized_test_dataset)

        print(f"Test results for model trained on {train_folder} and tested on {test_folder}:")
        print(f"Test Accuracy: {test_results.metrics['test_accuracy']:.4f}")
        print(f"Test Precision: {test_results.metrics['test_precision']:.4f}")
        print(f"Test Recall: {test_results.metrics['test_recall']:.4f}")
        print(f"Test F1: {test_results.metrics['test_f1']:.4f}")

        train_folder_name = os.path.basename(train_folder)
        test_folder_name = os.path.basename(test_folder)

        save_test_results(results_folder, test_dataset, test_results.predictions, train_folder_name, test_folder_name, model_name)
        save_metrics(results_folder, train_folder_name, test_folder_name, model_name, test_results.metrics)


def main():
    model_name = "google-bert/bert-base-uncased"  # Set the model name here
    with open("../../datasets-metadata.json") as f:
        datasets_metadata = json.load(f)
        results_folder = f"./testing_results/{model_name.replace('/', '-')}"
        os.makedirs(results_folder, exist_ok=True)

        for dataset_info in datasets_metadata["datasets"]:
            if dataset_info["task"] != "Object and Aspect Identification":
                continue

            train_folder = f"../../Splits/{dataset_info['folder']}"
            test_folders = [f"../../Splits/{other_dataset['folder']}" for other_dataset in datasets_metadata["datasets"] if other_dataset["task"] == "Named Entity Recognition"]

            train_and_test_on_datasets(train_folder, test_folders, results_folder, model_name)


if __name__ == "__main__":
    main()
