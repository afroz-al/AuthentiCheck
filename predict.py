from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate

tokenizer = AutoTokenizer.from_pretrained(
    "./test_trainer")
model = AutoModelForSequenceClassification.from_pretrained(
    "./test_trainer")


def predict_func(text: str):
    inputs = tokenizer(text, return_tensors='pt',
                       max_length=512, truncation=True)
    outputs = model(**inputs)
    print(outputs.logits)
    scores = outputs.logits[0].softmax(0).detach().numpy()
    result = {"label": scores.argmax().item(), "score": scores.max().item()}
    return result


def predict(text):
    id2label = ['Human', 'ChatGPT']
    res = predict_func(text)
    return id2label[res['label']], res['score']


text = "Do not meddle in the affairs of wizards, for they are subtle and quick to anger."
print(predict(text))
