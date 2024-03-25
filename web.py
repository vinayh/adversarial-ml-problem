import json

from flask import Flask, request, render_template
from PIL import Image

from adversarial import AdversarialGenerator


def read_imagenet_labels(path: str) -> dict[int, [str, str]]:
    with open(path) as json_data:
        data = json.load(json_data)
        return {int(k): v for k, v in data.items()}


app = Flask(__name__)
labels = read_imagenet_labels("imagenet_labels.json")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    orig_image = Image.open(request.files["image"])
    orig_preds = adversarial_generator.forward(orig_image)
    return str(orig_preds)
    

if __name__ == "__main__":
    adversarial_generator = AdversarialGenerator()
    app.run()