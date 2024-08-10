import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# if directory model does not exist, create it
if not os.path.exists("model"):
    os.makedirs("model")

# if folder model is empty, download the model
if not os.listdir("model"):
    print("Downloading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "uhhlt/binary-compqa-classifier", num_labels=2
    )  # .to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("uhhlt/binary-compqa-classifier")
    model.save_pretrained("model")
    tokenizer.save_pretrained("model")
else:
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained("model", num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("model")


# ====================== ML ==========================


def analyse_sentence(sentence):
    print("This sentence will be analyzed: " + sentence)

    inputs = tokenizer(sentence, return_tensors="pt")  # .to("cuda")

    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_class_id = logits.argmax().item()

    return predicted_class_id


# ====================== TESTING ==========================
# Load test data from csv file
import pandas as pd

test_data = pd.read_csv("../../Splits/webis_comparative_questions_2020_qi/test.csv")

# Test the model on the test data and save the results in a csv file with question, label format
results = []
for index, row in test_data.iterrows():
    question = row["question"]
    label = row["label"]
    predicted_label = analyse_sentence(question)
    results.append([question, predicted_label])

results_df = pd.DataFrame(results, columns=["question", "label"])
results_df.to_csv("results.csv", index=False)
print("Results saved in results.csv")

