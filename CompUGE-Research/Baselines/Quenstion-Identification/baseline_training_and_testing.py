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

    results_file_name = f"{train_folder_name}_{test_folder_name}_{model_name}_test_results.csv"
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
        'accuracy': [metrics['accuracy']],
        'precision': [metrics['precision']],
        'recall': [metrics['recall']],
        'f1': [metrics['f1']]
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
    datasets = load_data(train_folder, test_folder)
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


if __name__ == "__main__":
    train_folder = "../../Splits/webis_comparative_questions_2020_qi"
    test_folder = "../../Splits/webis_comparative_questions_2020_qi"
    model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    results_folder = "./testing_results"
    os.makedirs(results_folder, exist_ok=True)
    main(train_folder, test_folder, model_name, results_folder)
