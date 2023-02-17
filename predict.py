from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate

tokenizer = AutoTokenizer.from_pretrained(
    "./authenticheck")
model = AutoModelForSequenceClassification.from_pretrained(
    "./authenticheck")


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


text = "Laughter is the best medicine'that's something that my family and I have known since I was little. About a year ago, my brother, my dad, and I went to the store to buy some new sneakers. My dad found the ones he wanted, and the cashier rang them up. When the cashier asked my dad if he needed a bag, my brother said, 'No, thank you. He's just gonna wear these all the way home.' At first, my dad just furrowed his brow and did that dad-eyeroll that dads do when their kids make jokes. But then, something funny happened. All of us started laughing and couldn't stop! I mean, your dad walking home in a fresh pair of kicks? It was way too funny and we were all just in stitches. The cashier and the people around us in the store started cracking up too. Eventually, my dad had to hold the shoes against his chest just so he wouldn't drop them! Who knew something as small as a joke could bring so much joy and laughter? Laughter can be a great thing to have when you're in a tough situation or just feeling down. It's the best way to get closer with the people around you, no matter the age. My dad, my brother, and I are still laughing about that time at the store to this day. It was a moment filled with joy and laughter that I won't soon forget"
print(predict(text))
