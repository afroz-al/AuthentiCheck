from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate

tokenizer = AutoTokenizer.from_pretrained(
    "Hello-SimpleAI/chatgpt-detector-roberta")
model = AutoModelForSequenceClassification.from_pretrained(
    "Hello-SimpleAI/chatgpt-detector-roberta").to("cuda")
id2label = ['Human', 'ChatGPT']


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy_metric = evaluate.load("accuracy")
    recall_metric = evaluate.load("recall")
    precision_metric = evaluate.load("precision")
    return {"acc":  accuracy_metric.compute(
        predictions=predictions, references=labels), "recall":  recall_metric.compute(predictions=predictions, references=labels), "precision": precision_metric.compute(
        predictions=predictions, references=labels)}


def tokenize_function(text):
    encoding = tokenizer(text['answer'] + "</s>" + text['question'], padding="max_length",
                         max_length=512, truncation=True)
    if (text["chatgpt"] == True):
        encoding["label"] = 1
    else:
        encoding["label"] = 0
    return encoding


training_args = TrainingArguments(
    output_dir="authenticheck", evaluation_strategy="epoch")


dataset = load_dataset("datasets",  data_files={
                       "train": "train-new.jsonl", "test": "test-new.jsonl"})
tokenized_datasets = dataset.map(tokenize_function, batched=False)

small_train_dataset = tokenized_datasets["train"].shuffle(
    seed=42)
small_eval_dataset = tokenized_datasets["test"].shuffle(
    seed=42)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

trainer.train()
trainer.save_model()
