import json

from flask import Flask, render_template

def read_imagenet_labels(path: str) -> dict[int, [str, str]]:
    with open(path) as json_data:
        data = json.load(json_data)
        return {int(k): v for k, v in data.items()}


app = Flask(__name__)
labels = read_imagenet_labels("imagenet_labels.json")


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run()