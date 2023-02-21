from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(
    "./authenticheck")
model = AutoModelForSequenceClassification.from_pretrained(
    "./authenticheck")


def predict_func(text: str):
    inputs = tokenizer(text, return_tensors='pt',
                       max_length=512, truncation=True)
    outputs = model(**inputs)
    scores = outputs.logits[0].softmax(0).detach().numpy()
    return scores


def predict(text):
    res = predict_func(text)
    return {'Human': f'{round(res[0] * 100, 2)}%', 'ChatGPT': f'{round(res[1] * 100, 2)}%'}


app = Flask(__name__)


@app.route("/")
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_text():
    """Grabs the input values and uses them to make prediction"""
    text = request.form["text"]
    output = predict(text)

    return render_template('index.html', prediction_text=f'{output}', input_text=text)


if __name__ == "__main__":
    app.run()
