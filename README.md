# adversarial-ml-problem

## Basic use

To use the basic web UI to more easily play around with the adversarial sample generator, clone the repo and install the packages in the Conda environment `conda_environment.yml`, then run `python web.py` and access the web UI at `http://127.0.0.1:5000` or the URL shown in your terminal emulator

Otherwise, the adversarial example generator can be used through the `AdversarialGenerator` class, which allows for initialization with another Torchvision model besides the default `RegNet_X_800MF`:

- `AdversarialGenerator.forward(image)` returns the top ImageNet label predictions for the input image
- `AdversarialGenerator.forward_with_adversarial(image, target_label)` returns original/adversarial predictions, adversarial noise, and the adversarial output image

## Future work

- Improve scaling of noise tensor to better adversarially attack model to predict target label without affecting image quality (appears to work on the few images and several target labels that were tested)
- Explore training of noise tensor further to improve hyperparameters and architectural improvements
- Add more unit tests to make code more robust to changes
- Add more comprehensive end-to-end tests that better examine robustness across different images and target labels
- Evaluate effect of adjusting epsilon value (clipping of noise tensor) vs. target label confidence and other metrics
- Improve code documentation: docstrings, type annotations where missing, etc.