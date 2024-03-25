import torch
from PIL import Image
from torchvision.models import regnet_x_800mf, RegNet_X_800MF_Weights

from adversarial import AdversarialGenerator


def test_forward():
    adversarial_generator = AdversarialGenerator(model=regnet_x_800mf,
                                                 weights=RegNet_X_800MF_Weights.IMAGENET1K_V2,
                                                 epsilon=0.1)
    orig_image = Image.open("./static/n01677366_common_iguana.jpg")
    topk_preds = adversarial_generator.forward(orig_image)
    # TODO: Currently set rtol to 1e-2, may be helpful to lower if improving test cases later
    assert torch.all(torch.isclose(topk_preds.values, torch.tensor([0.3436, 0.0099, 0.0072, 0.0057, 0.0051]), rtol=1e-2))
    assert torch.all(topk_preds.indices == torch.tensor([39, 308, 79, 176, 46]))