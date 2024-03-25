import json
import torch
from flask import Flask, request, render_template
from PIL import Image

from adversarial import AdversarialGenerator


def read_imagenet_labels(path: str) -> dict[int, [str, str]]:
    """
    Reads JSON of ImageNet labels into a Python dict
    {label_index_int: [label_id, label_name]}
    """
    with open(path) as json_data:
        data = json.load(json_data)
        return {int(k): v for k, v in data.items()}


def topk_to_tuples(preds: torch.return_types.topk) -> [tuple[str, float]]:
    """
    Reformats PyTorch topk return type (values and indices) to list of tuples
    [(label_name_str, confidence_float)]
    """
    return [(labels[int(preds.indices[i])], float(preds.values[i])) for i in range(len(preds.values))]


app = Flask(__name__)
labels = read_imagenet_labels("imagenet_labels.json")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    orig_image = Image.open(request.files["image"])
    orig_preds = adversarial_generator.forward(orig_image)
    return render_template("predict.html", orig_preds=topk_to_tuples(orig_preds))


@app.route("/adversarial", methods=["POST"])
def adversarial():
    orig_image = Image.open(request.files["image"])
    target_label = int(request.form["targetLabel"])
    orig_preds, noise, adv_image, adv_preds = adversarial_generator.forward_with_adversarial(orig_image, target_label)
    return render_template("adversarial.html",
                           orig_preds=topk_to_tuples(orig_preds),
                           adv_preds=topk_to_tuples(adv_preds),
                           target_label=target_label)


if __name__ == "__main__":
    adversarial_generator = AdversarialGenerator()
    app.run()
