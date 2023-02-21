from sklearn.metrics import classification_report
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn import metrics


""""
 False       1.00      1.00      1.00      1882
        True       0.97      0.99      0.98       240

    accuracy                           1.00      2122
   macro avg       0.98      0.99      0.99      2122
weighted avg       1.00      1.00      1.00      2122
"""

tokenizer = AutoTokenizer.from_pretrained(
    "./authenticheck")
model = AutoModelForSequenceClassification.from_pretrained(
    "./authenticheck")

y_true = []
y_pred = []


def predict_func(text: str):
    inputs = tokenizer(text, return_tensors='pt',
                       max_length=512, truncation=True)
    outputs = model(**inputs)
    scores = outputs.logits[0].softmax(0).detach().numpy()
    result = {"label": scores.argmax().item(), "score": scores.max().item()}
    return result


def predict(row):
    print("predicting for text: ", row['answer'])
    text = row['answer']
    y_true.append(row['chatgpt'])
    res = predict_func(text)
    pred = True if res['label'] == 1 else False
    y_pred.append(pred)


dataset = load_dataset("datasets",  data_files={
    "train": "train.jsonl", "test": "test.jsonl"})
dataset["test"].map(predict, batched=False)

print(classification_report(y_true, y_pred))
print("Precision ==", metrics.precision_score(y_true, y_pred))
print("Recall ==", metrics.recall_score(y_true, y_pred))
print("F1 ==", metrics.f1_score(y_true, y_pred))
print("Accuracy ==", metrics.accuracy_score(y_true, y_pred))
