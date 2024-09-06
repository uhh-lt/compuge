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

    print(f"Results: {results_unfolded}")
    print(results_unfolded.keys())
    return results_unfolded


def model_init_helper(model_name, num_labels):
    def model_init():
        # Load the model with label mappings
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels,  # Set the number of labels here
            ignore_mismatched_sizes=True,
        )
        print(f"{model.config.num_labels = }")
        return model

    return model_init



def tokenize_and_align_labels(examples, one_label_per_word=True, **kwargs):
    tokenizer = kwargs["tokenizer"]

    # Convert string representations of lists to actual lists
    words = [eval(word_list) for word_list in examples["words"]]
    labels = [eval(label_list) for label_list in examples["labels"]]

    # Tokenize inputs while keeping word boundaries
    tokenized_inputs = tokenizer(
        words, truncation=True, is_split_into_words=True
    )

    aligned_labels = []
    all_word_ids = []
    for i, label in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Get word IDs for the batch
        all_word_ids.append(word_ids)  # Store the word_ids for later use
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

    # Add the processed labels and word_ids back to the tokenized inputs
    tokenized_inputs["labels"] = aligned_labels
    tokenized_inputs["word_ids"] = all_word_ids
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
        'accuracy': metrics['overall_accuracy'],
        'precision': metrics['overall_precision'],
        'recall': metrics['overall_recall'],
        'f1': metrics['overall_f1']
    }

    # Create or append the metrics data to a CSV file
    metrics_file_path = os.path.join(results_folder, "metrics.csv")
    metrics_df = pd.DataFrame([metrics_data])

    # Save the metrics DataFrame to a CSV file
    metrics_df.to_csv(metrics_file_path, mode='a', header=not os.path.exists(metrics_file_path), index=False)
    print(f"Metrics saved to {metrics_file_path}")


def save_test_results(results_folder, test_dataset, predictions, train_folder_name, test_folder_name, model_name):
    # Convert the model's output logits to predicted label IDs
    pred_labels = np.argmax(predictions, axis=-1)

    # Convert predictions to a more usable format
    formatted_predictions = []
    for i, label_list in enumerate(pred_labels):
        word_ids = test_dataset[i]['word_ids']  # Now `word_ids` should be present

        # Ensure that we only consider valid word indices, excluding special tokens
        valid_predictions = []
        for j, word_id in enumerate(word_ids):
            if word_id is None or word_id == -100:
                continue
            # Append the label only if it matches a word_id
            valid_predictions.append(int(pred_labels[i][j]))

        formatted_predictions.append(valid_predictions)

    # Create a DataFrame with the test dataset
    test_df = pd.DataFrame(test_dataset)

    # Add the predicted labels as a new column
    test_df['predictions'] = formatted_predictions

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
    print("=========================================")
    print(f"Training on {train_folder}")
    print("=========================================")

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
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=8,
        weight_decay=0.1,
        load_best_model_at_end=True,
        learning_rate=2e-5,
        seed=0,
    )

    trainer = Trainer(
        model_init=model_init_helper(model_name, 5),
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Training =========================================")
    trainer.train()

    print("Testing =========================================")
    for test_folder in test_folders:
        test_dataset = load_data(train_folder, test_folder)['test']
        tokenized_test_dataset = test_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            fn_kwargs={"tokenizer": tokenizer},
        )
        test_results = trainer.predict(test_dataset=tokenized_test_dataset)
        print("=========================================")
        # dict_keys(['ASPECT_precision', 'ASPECT_recall', 'ASPECT_f1', 'ASPECT_number', 'OBJ_precision', 'OBJ_recall', 'OBJ_f1', 'OBJ_number', 'overall_precision', 'overall_recall', 'overall_f1', 'overall_accuracy'])
        print(f"Test results for model trained on {train_folder} and tested on {test_folder}:")
        print(test_results.metrics.keys())
        print(f"Test Accuracy: {test_results.metrics['test_overall_accuracy']:.4f}")
        print(f"Test Precision: {test_results.metrics['test_overall_precision']:.4f}")
        print(f"Test Recall: {test_results.metrics['test_overall_recall']:.4f}")
        print(f"Test F1: {test_results.metrics['test_overall_f1']:.4f}")
        train_folder_name = os.path.basename(train_folder)
        test_folder_name = os.path.basename(test_folder)

        save_test_results(results_folder, test_dataset, test_results.predictions, train_folder_name, test_folder_name, model_name)
        save_metrics(results_folder, train_folder_name, test_folder_name, model_name, test_results.metrics)


def main():
    model_name = "google-bert/bert-base-uncased"
    # Set the model name here
    train_and_test_on_datasets(
        "../../Splits/oai_beloucif",
        ["../../Splits/oai_beloucif"],
        "./testing_results",
        model_name
    )
    '''
    f
    with open("../../datasets-metadata.json") as f:
        datasets_metadata = json.load(f)
        results_folder = f"./testing_results/{model_name.replace('/', '-')}"
        os.makedirs(results_folder, exist_ok=True)

        for dataset_info in datasets_metadata["datasets"]:
            if dataset_info["task"] != "Object and Aspect Identification":
                continue

            train_folder = f"../../Splits/{dataset_info['folder']}"
            test_folders = [f"../../Splits/{other_dataset['folder']}" for other_dataset in datasets_metadata["datasets"] if other_dataset["task"] == "Object and Aspect Identification"]

            train_and_test_on_datasets(train_folder, test_folders, results_folder, model_name)
    '''

if __name__ == "__main__":
    main()
