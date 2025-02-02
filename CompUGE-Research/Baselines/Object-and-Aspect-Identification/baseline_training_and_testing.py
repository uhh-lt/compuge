import json
import os
from operator import contains

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

id2label = {0: "O", 1: "B-OBJ", 2: "I-OBJ", 3: "B-ASPECT", 4: "I-ASPECT"}

def compute_metrics(eval_preds):
    predictions, labels = eval_preds.predictions, eval_preds.label_ids
    predictions = np.argmax(predictions, axis=-1)
    true_labels = [[l for l in label if l != -100] for label in labels]
    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [[id2label[l] for l in label] for label in true_labels]
    true_predictions = [[id2label[p] for p in prediction] for prediction in true_predictions]
    metric = evaluate.load("seqeval")
    results = metric.compute(predictions=true_predictions, references=true_labels)
    results_unfolded = {}
    for key, value in results.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                results_unfolded[key + "_" + subkey] = subvalue
        else:
            results_unfolded[key] = value
    return results_unfolded

def model_init_helper(model_name, num_labels):
    def model_init():
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )
        return model
    return model_init

def tokenize_and_align_labels(examples, one_label_per_word=True, **kwargs):
    tokenizer = kwargs["tokenizer"]
    words = [eval(word_list) for word_list in examples["words"]]
    labels = [eval(label_list) for label_list in examples["labels"]]
    tokenized_inputs = tokenizer(
        words, truncation=True, is_split_into_words=True
    )
    aligned_labels = []
    all_word_ids = []
    for i, label in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        all_word_ids.append(word_ids)
        current_word = None
        new_labels = []
        for word_id in word_ids:
            if word_id is None:
                new_labels.append(-100)
            elif word_id != current_word:
                current_word = word_id
                new_labels.append(label[word_id])
            else:
                lab = int(label[word_id])
                if lab % 2 == 1:
                    lab += 1
                new_labels.append(-100 if one_label_per_word else lab)
        aligned_labels.append(new_labels)
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
    metrics_data = {
        'training on': train_folder_name,
        'tested on': test_folder_name,
        'model': model_name.split('/')[0],
        'accuracy': metrics['test_overall_accuracy'],
        'precision': metrics['test_overall_precision'],
        'recall': metrics['test_overall_recall'],
        'f1': metrics['test_overall_f1'],
        'object_precision': metrics['test_OBJ_precision'],
        'object_recall': metrics['test_OBJ_recall'],
        'object_f1': metrics['test_OBJ_f1'],
        'aspect_precision': metrics['test_ASPECT_precision'],
        'aspect_recall': metrics['test_ASPECT_recall'],
        'aspect_f1': metrics['test_ASPECT_f1'],
    }
    metrics_file_path = os.path.join(results_folder, "metrics.csv")
    metrics_df = pd.DataFrame([metrics_data])
    metrics_df.to_csv(metrics_file_path, mode='a', header=not os.path.exists(metrics_file_path), index=False)
    print(f"Metrics saved to {metrics_file_path}")

def save_test_results(results_folder, tokenized_test_dataset, predictions, train_folder_name, test_folder_name, model_name):
    pred_labels = np.argmax(predictions, axis=-1)
    formatted_predictions = []
    for i, label_list in enumerate(pred_labels):
        word_ids = tokenized_test_dataset[i]['word_ids']
        valid_predictions = []
        for j, word_id in enumerate(word_ids):
            if word_id is None or word_id == -100:
                continue
            valid_predictions.append(int(pred_labels[i][j]))
        formatted_predictions.append(valid_predictions)
    test_df = pd.DataFrame(tokenized_test_dataset)
    test_df['predictions'] = formatted_predictions
    model_name = model_name.split('/')[0]
    results_file_name = f"{train_folder_name}_{test_folder_name}_{model_name}_test_results.csv"
    results_path = os.path.join(results_folder, results_file_name)
    os.makedirs(results_folder, exist_ok=True)
    test_df.to_csv(results_path, index=False)
    print(f"Test results saved to {results_path}")

def train_and_test_on_datasets(train_folder, test_folders, results_folder, model_name, patch_size, learning_rate, weight_decay, num_train_epochs):
    datasets = load_data(train_folder)
    if datasets is None:
        print(f"No training data found in folder {train_folder}. Exiting.")
        return
    print("=========================================")
    print(f"Training on {train_folder}")
    print("=========================================")
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=patch_size, add_prefix_space=True)
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
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        learning_rate=learning_rate,
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
        if 'word_ids' not in tokenized_test_dataset[0]:
            raise ValueError("`word_ids` not found in tokenized test dataset. Please check the tokenization process.")
        test_results = trainer.predict(test_dataset=tokenized_test_dataset)
        print("=========================================")
        print(f"Test results for model trained on {train_folder} and tested on {test_folder}:")
        print(test_results.metrics.keys())
        print(f"Test Accuracy: {test_results.metrics['test_overall_accuracy']:.4f}")
        print(f"Test Precision: {test_results.metrics['test_overall_precision']:.4f}")
        print(f"Test Recall: {test_results.metrics['test_overall_recall']:.4f}")
        print(f"Test F1: {test_results.metrics['test_overall_f1']:.4f}")
        train_folder_name = os.path.basename(train_folder)
        test_folder_name = os.path.basename(test_folder)
        save_test_results(results_folder, tokenized_test_dataset, test_results.predictions, train_folder_name, test_folder_name, model_name)
        save_metrics(results_folder, train_folder_name, test_folder_name, model_name, test_results.metrics)

def main():
    models_data = [
        {
            "model": "google-bert/bert-base-uncased",
            "train_batch_size": 16,
            "num_train_epochs": 8,
            "weight_decay": 0.1,
            "learning_rate": 0.00005
        },
        {
            "model": "FacebookAI/roberta-base",
            "train_batch_size": 16,
            "num_train_epochs": 8,
            "weight_decay": 0.0001,
            "learning_rate": 0.0001
        },
        {
            "model": "distilbert/distilbert-base-uncased",
            "train_batch_size": 16,
            "num_train_epochs": 8,
            "weight_decay": 0.001,
            "learning_rate": 0.0001
        },
        {
            "model": "microsoft/deberta-v3-base",
            "train_batch_size": 16,
            "num_train_epochs": 8,
            "weight_decay": 0.01,
            "warmup_steps": 100,
            "learning_rate": 0.00007
        }
    ]

    for model_data in models_data:
        with open("../../datasets-metadata.json") as f:
            datasets_metadata = json.load(f)
            results_folder = f"./testing_results_predsless/{model_data['model'].replace('/', '-')}"
            os.makedirs(results_folder, exist_ok=True)
            for dataset_info in datasets_metadata["datasets"]:
                if dataset_info["task"] != "Object and Aspect Identification":
                    continue
                train_folder = f"../../Splits/{dataset_info['folder']}"
                # test on all datasets where task is Object and Aspect Identification
                test_folders = [f"../../Splits/{dataset['folder']}" for dataset in datasets_metadata["datasets"] if dataset["task"] == "Object and Aspect Identification"]
                train_and_test_on_datasets(
                    train_folder,
                    test_folders,
                    results_folder,
                    model_data["model"],
                    patch_size=512,
                    learning_rate=model_data["learning_rate"],
                    weight_decay=model_data["weight_decay"],
                    num_train_epochs=model_data["num_train_epochs"],
                )

if __name__ == "__main__":
    main()
