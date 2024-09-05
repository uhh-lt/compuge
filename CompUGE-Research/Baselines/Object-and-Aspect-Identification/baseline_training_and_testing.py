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

# Example label mapping; replace with your actual labels
id2label = {0: "O", 1: "B-OBJ", 2: "I-OBJ", 3: "B-ASPECT", 4: "I-ASPECT"}

def compute_metrics(eval_preds):
    print("Computing metrics...")
    predictions, labels = eval_preds.predictions, eval_preds.label_ids

    # Convert predictions to label indices by taking the argmax across the last dimension
    predictions = np.argmax(predictions, axis=-1)

    # Filter out special tokens (-100) and flatten lists to compute metrics
    true_labels = [[l for l in label if l != -100] for label in labels]
    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Convert numeric labels to string labels
    true_labels = [[id2label[l] for l in label] for label in true_labels]
    true_predictions = [[id2label[p] for p in prediction] for prediction in true_predictions]

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

    print(f"Metrics computed: {results_unfolded}")
    return results_unfolded


def model_init_helper(model_name, num_labels):
    def model_init():
        # Load the model with label mappings
        print(f"Loading model {model_name} with {num_labels} labels...")
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels,  # Set the number of labels here
            ignore_mismatched_sizes=True,
        )
        print(f"Model loaded with configuration: {model.config.num_labels}")
        return model

    return model_init


def tokenize_and_align_labels(examples, one_label_per_word=True, **kwargs):
    tokenizer = kwargs["tokenizer"]

    # Convert string representations of lists to actual lists
    words = [eval(word_list) for word_list in examples["words"]]
    labels = [eval(label_list) for label_list in examples["labels"]]

    # Tokenize inputs while keeping word boundaries
    print("Tokenizing and aligning labels...")
    tokenized_inputs = tokenizer(
        words, truncation=True, is_split_into_words=True
    )

    aligned_labels = []
    for i, label in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Get word IDs for the batch
        current_word = None
        new_labels = []
        for word_id in word_ids:
            if word_id is None:  # Token is a special token (like [CLS] or [SEP])
                new_labels.append(-100)  # Mask out special tokens for loss calculation
            elif word_id != current_word:  # New word
                current_word = word_id
                new_labels.append(label[word_id])  # Use the label for the new word
            else:
                lab = int(label[word_id])  # Ensure the label is an integer
                if lab % 2 == 1:  # Convert to "B" (Beginning) tag if "I" (Inside) tag
                    lab += 1
                # Mask out non-beginning tokens or include them depending on 'one_label_per_word'
                new_labels.append(-100 if one_label_per_word else lab)
        aligned_labels.append(new_labels)

    # Add the processed labels back to the tokenized inputs
    tokenized_inputs["labels"] = aligned_labels
    print("Labels aligned.")
    return tokenized_inputs


def load_data(train_folder, test_folder=None):
    print(f"Loading data from {train_folder}...")
    dataset_dict = {}

    for split in ['train', 'val']:
        csv_path = os.path.join(train_folder, f"{split}.csv")
        if os.path.exists(csv_path):
            print(f"Found {split} data at {csv_path}.")
            data = pd.read_csv(csv_path)
            dataset_dict[split] = Dataset.from_pandas(data)
        else:
            print(f"No {split} data found at {csv_path}.")

    if 'train' not in dataset_dict:
        print("No training data found. Exiting.")
        return None

    if 'val' not in dataset_dict:
        print("No validation data found. Creating validation set from training data.")
        train_data = pd.read_csv(os.path.join(train_folder, "train.csv"))
        val_data = train_data.sample(frac=0.1, random_state=42)
        dataset_dict['train'] = Dataset.from_pandas(train_data.drop(val_data.index))
        dataset_dict['val'] = Dataset.from_pandas(val_data)

    if test_folder:
        test_data = pd.read_csv(os.path.join(test_folder, "test.csv"))
        dataset_dict['test'] = Dataset.from_pandas(test_data)
        print(f"Loaded test data from {test_folder}.")

    return DatasetDict(dataset_dict)


def save_metrics(results_folder, train_folder_name, test_folder_name, model_name, metrics):
    print("Saving metrics...")
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
    print("Saving test results...")
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
    print(f"Training and testing with model {model_name}...")
    datasets = load_data(train_folder)
    if datasets is None:
        print(f"No training data found in folder {train_folder}. Exiting.")
        return

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=64, add_prefix_space=True)

    print("Tokenizing datasets...")
    tokenized_datasets = datasets.map(
        tokenize_and_align_labels,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
    )

    print("Setting up data collator...")
    data_collator = DataCollatorForTokenClassification(tokenizer)

    print("Setting up training arguments...")
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

    print("Initializing Trainer...")
    trainer = Trainer(
        model_init=model_init_helper(model_name, 5),
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Training...")
    trainer.train()

    for test_folder in test_folders:
        print(f"Testing on dataset from folder: {test_folder}...")
        test_dataset = load_data(train_folder, test_folder)
        if test_dataset is None or "test" not in test_dataset:
            print(f"No test data found in folder {test_folder}. Skipping.")
            continue

        test_results = trainer.predict(test_dataset["test"])
        save_test_results(
            results_folder,
            test_dataset["test"],
            test_results,
            train_folder.split('/')[-1],
            test_folder.split('/')[-1],
            model_name
        )
        metrics = compute_metrics(test_results)
        save_metrics(
            results_folder,
            train_folder.split('/')[-1],
            test_folder.split('/')[-1],
            model_name,
            metrics
        )
