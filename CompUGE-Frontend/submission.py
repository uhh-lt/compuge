
# a method that recieves a list of lables and predictions and returns :
# 1. the accuracy of the model
# 2. the precision of the model
# 3. the recall of the model
# 4. the f1 score of the model
# 5. the overall score of the model
def evaluate_model(labels, predictions):
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    overall = (accuracy + precision + recall + f1) / 4
    return accuracy, precision, recall, f1, overall


# a method that calculates the accuracy of the model
def accuracy_score(labels, predictions):
    return sum([1 for i in range(len(labels)) if labels[i] == predictions[i]]) / len(labels)


def precision_score(labels, predictions):
    tp = sum([1 for i in range(len(labels)) if labels[i] == 1 and predictions[i] == 1])
    fp = sum([1 for i in range(len(labels)) if labels[i] == 0 and predictions[i] == 1])
    return tp / (tp + fp)


def recall_score(labels, predictions):
    tp = sum([1 for i in range(len(labels)) if labels[i] == 1 and predictions[i] == 1])
    fn = sum([1 for i in range(len(labels)) if labels[i] == 1 and predictions[i] == 0])
    return tp / (tp + fn)


def f1_score(labels, predictions):
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    return 2 * (precision * recall) / (precision + recall)


def overall_score(labels, predictions):
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    acc_weight = 0.25
    precision_weight = 0.25
    recall_weight = 0.25
    f1_weight = 0.25
    return (accuracy * acc_weight) + (precision * precision_weight) + (recall * recall_weight) + (f1 * f1_weight)

