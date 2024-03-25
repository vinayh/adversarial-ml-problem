import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import regnet_x_800mf, RegNet_X_800MF_Weights
from PIL import Image
from typing import Union


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
    
    def forward(self, image: Union[Image.Image, torch.Tensor]) -> torch.return_types.topk:
        """
        Runs image (PIL.Image.Image) through model and returns
        top-k (k=5) predictions with softmax probabilities
        """
        preprocessed = self.preprocess(image)
        out = self.model(torch.unsqueeze(preprocessed, 0)).squeeze(0)
        probs = F.softmax(out)
        return torch.topk(probs.detach(), k=5, sorted=True)
    
    def adversarial_noise(self, image: torch.Tensor, orig_label: torch.Tensor, target_label: torch.Tensor) -> torch.Tensor:
        """
        Runs PGD attack (iterations of gradient descent similar to FGSM)
        - Init noise tensor (delta) and optimizer
        - Loop:
            - Run preprocessed image with noise through model
            - Optimizer step to reduce loss wrt target label and increase loss wrt original label
        """
        num_iterations = 50
        delta = torch.zeros_like(image, requires_grad=True)
        optimizer = torch.optim.SGD([delta], lr=1e-2)
        for i in range(num_iterations):
            optimizer.zero_grad()
            adv_image = self.preprocess(image + delta)
            out = self.model(torch.unsqueeze(adv_image, 0))
            loss = F.cross_entropy(out, torch.tensor([target_label])) - F.cross_entropy(out, torch.tensor([orig_label]))
            loss.backward()
            optimizer.step()
        return delta
    
    def forward_with_adversarial(self, image: Image.Image, target_label: int):
        """
        Runs forward pass to find original labels, generates adversarial noise to classify as target label
        Returns original predictions, noise tensor, adversarial image, and new adversarial predictions
        """
        orig_preds = self.forward(image)
        orig_label = orig_preds.indices[0]  # Top pred label of original image
        target_label = torch.tensor([target_label])
        image_t = T.ToTensor()(image) ##
        noise_t = self.adversarial_noise(image_t, orig_label, target_label)
        adversarial_image_t = image_t + noise_t
        adversarial_preds = self.forward(adversarial_image_t)
        return orig_preds, noise_t, adversarial_image_t, adversarial_preds
        
    