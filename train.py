from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate

metric = evaluate.load("accuracy")
tokenizer = AutoTokenizer.from_pretrained(
    "Hello-SimpleAI/chatgpt-detector-roberta")
model = AutoModelForSequenceClassification.from_pretrained(
    "Hello-SimpleAI/chatgpt-detector-roberta")
id2label = ['Human', 'ChatGPT']


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def tokenize_function(text):
    encoding = tokenizer(text['answer'] + "</s>" + text['question'], padding="max_length",
                         max_length=512, truncation=True)
    if (text["chatgpt"] == True):
        encoding["labels"] = 1
    else:
        encoding["labels"] = 0
    return encoding


training_args = TrainingArguments(
    output_dir="test_trainer", evaluation_strategy="epoch")


dataset = load_dataset("data",  data_files={
                       "train": "train.jsonl", "test": "test.jsonl"})
tokenized_datasets = dataset.map(tokenize_function, batched=False)
print(tokenized_datasets)

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
