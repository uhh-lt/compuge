import os
from transformers import pipeline

# ====================== API ==========================

best_model_name = "uhhlt/comp-seqlab-deberta"
fast_model_name = "uhhlt/comp-seqlab-dslim-bert"
strategy = "simple"

if not os.path.exists("model"):
    os.makedirs("model")

if not os.path.exists("model/best"):
    os.makedirs("model/best")

if not os.path.exists("model/fast"):
    os.makedirs("model/fast")

if not os.listdir("model/best"):
    print("Downloading best model...")
    token_classifier = pipeline(
        "token-classification",
        model=best_model_name,
        aggregation_strategy=strategy,
    )
    token_classifier.save_pretrained("model/best")
else:
    print("Loading best model...")
    token_classifier = pipeline(
        "token-classification",
        model="model/best",
        aggregation_strategy=strategy,
    )

if not os.listdir("model/fast"):
    print("Downloading fast model...")
    token_classifier_fast = pipeline(
        "token-classification",
        model=fast_model_name,
        aggregation_strategy=strategy,
    )
    token_classifier_fast.save_pretrained("model/fast")
else:
    print("Loading fast model...")
    token_classifier_fast = pipeline(
        "token-classification",
        model="model/fast",
        aggregation_strategy=strategy,
    )


def predict(question: str, fast: bool):
    print(f"This question will be analyzed: {question}")

    if fast:
        tokens = token_classifier_fast(question)
    else:
        tokens = token_classifier(question)

    return {"tokens": tokens}


# ====================== TESTING ==========================
# Load test data from csv file
import pandas as pd

test_data = pd.read_csv("results.csv")
# remove column labels from the test data
test_data = test_data.drop(columns=["labels"])
# rename the column predictions to labels
test_data.columns = ["question", "labels"]
# save the test data without labels
test_data.to_csv("results.csv", index=False)