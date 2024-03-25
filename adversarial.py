import torch
import torch.nn.functional as F
from torchvision.models import regnet_x_800mf, RegNet_X_800MF_Weights
from PIL import Image


class AdversarialGenerator:
    def __init__(self,
                 model=regnet_x_800mf,
                 weights=RegNet_X_800MF_Weights.IMAGENET1K_V2,
                 epsilon=0.1):
        self.epsilon = epsilon
        self.model = model(weights)
        self.model.eval()
        self.preprocess = weights.transforms()
        torch.manual_seed(1748)
    
    def forward(self, image: Image.Image) -> torch.return_types.topk:
        """
        Runs image (PIL.Image.Image) through model and returns
        top-k (k=5) predictions with softmax probabilities
        """
        preprocessed = self.preprocess(image)
        out = self.model(torch.unsqueeze(preprocessed, 0)).squeeze(0)
        probs = F.softmax(out)
        return torch.topk(probs.detach(), k=5, sorted=True)
    
    def adversarial_noise(self, image: torch.Tensor, orig_label: int, target_label: int) -> torch.Tensor:
        """
        Runs PGD attack (iterations of gradient descent similar to FGSM)
        - Init noise tensor and optimizer
        - Loop:
            - Run preprocessed image with noise through model
            - Optimizer step to reduce loss wrt target label and increase loss wrt original label
        """
        pass
    
    def forward_with_adversarial(self, image: Image.Image, target_label: int):
        """
        Run forward pass to find original labels, generate adversarial noise to classify
        as target label, return original predictions, noise tensor, adversarial image,
        and new adversarial predictions
        """
        pass
    