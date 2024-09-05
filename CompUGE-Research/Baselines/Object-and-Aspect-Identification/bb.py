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

def model_init_helper():
    def model_init():
        # Directly load the model without label mappings
        model = AutoModelForTokenClassification.from_pretrained(
            "microsoft/deberta-v3-large",
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

def train_bert(folder_path):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # uncomment to set a fixed seed
    transformers.set_seed(0)

    # Data loading

    train = pd.read_csv(os.path.join(folder_path, "train.csv"))
    val = pd.read_csv(os.path.join(folder_path, "val.csv"))
    test = pd.read_csv(os.path.join(folder_path, "test.csv"))

    ner_data = DatasetDict(
        {
            "train": Dataset.from_pandas(train),
            "valid": Dataset.from_pandas(val),
            "test": Dataset.from_pandas(test),
        }
    )

    """
    # Training
    """

    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/deberta-v3-large",
        model_max_length=64,
        add_prefix_space=True,
    )

    tokenized_datasets = ner_data.map(
        tokenize_and_align_labels,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    args = TrainingArguments(
        output_dir="./results",
        seed=0,  # uncomment for a fixed seed
        data_seed=0,  # uncomment for a fixed seed
        run_name="microsoft-deberta-v3-large",
        metric_for_best_model="eval_overall_f1",
        evaluation_strategy="steps",
        include_inputs_for_metrics=True,
    )
    args.set_dataloader(
        sampler_seed=0,  # uncomment for a fixed seed
        train_batch_size=16,
        eval_batch_size=8,
    )
    args.set_evaluate(
        strategy="steps",
        steps=100,
        delay=0.0,
        batch_size=8,
    )
    args.set_logging(
        strategy="steps",
        steps=100,
        report_to=["wandb"],
        first_step=True,
        level="info",
    )
    args.set_lr_scheduler(
        name="cosine",
        warmup_steps=100,
    )

    args.set_optimizer(
        name="adamw_torch",
        learning_rate=0.00005,
        weight_decay=0.01,
    )
    args.set_save(strategy="steps", steps=100)
    args.set_testing(batch_size=8)
    args.set_training(
        num_epochs=11,
        batch_size=16,
    )

    trainer = Trainer(
        model=model_init_helper(),
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Training")
    trainer.train()

if __name__ == "__main__":
    train_bert("./data/")



