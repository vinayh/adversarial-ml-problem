import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import regnet_x_800mf, RegNet_X_800MF_Weights
from PIL import Image
from typing import Union


class AdversarialGenerator:
    """
    final_clip_epsilon: float or None, sets epsilon value for clipping of final noise tensor
    clip_grad_norm: float or None: sets delta grad norm clip value
    """
    def __init__(self,
                 model=regnet_x_800mf,
                 weights=RegNet_X_800MF_Weights.IMAGENET1K_V2,
                 final_clip_epsilon=0.2,
                 clip_grad_norm=None
                 ):
        self.final_clip_epsilon = final_clip_epsilon
        self.clip_grad_norm = clip_grad_norm
        self.model = model(weights=weights)
        self.model.eval()
        self.preprocess = weights.transforms()
        torch.manual_seed(1748)
    
    def forward(self, image: Union[Image.Image, torch.Tensor]) -> torch.return_types.topk:
        """
        Runs image (PIL.Image.Image) through model and returns
        top-k (k=5) predictions with softmax probabilities
        """
        preprocessed = self.preprocess(image)
        out = self.model(torch.unsqueeze(preprocessed, dim=0)).squeeze(0)
        probs = F.softmax(out, dim=0)
        return torch.topk(probs.detach(), k=5, sorted=True)
    
    def adversarial_noise(self, image: torch.Tensor, orig_label: torch.Tensor,
                          target_label: torch.Tensor) -> torch.Tensor:
        """
        Runs PGD attack (iterations of gradient descent similar to FGSM)
        - Init noise tensor (delta) and optimizer
        - Loop:
            - Run preprocessed image with noise through model
            - Optimizer step to reduce loss wrt target label and increase loss wrt original label
        """
        num_iterations = 100
        delta = torch.zeros_like(image, requires_grad=True)
        optimizer = torch.optim.SGD([delta], lr=1e-2)
        for i in range(num_iterations):
            optimizer.zero_grad()
            adv_image = self.preprocess(image + delta)
            out = self.model(torch.unsqueeze(adv_image, dim=0))
            loss = F.cross_entropy(out, torch.tensor([target_label])) - F.cross_entropy(out, torch.tensor([orig_label]))
            loss.backward()
            if self.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(delta, self.clip_grad_norm)
            optimizer.step()
        if self.final_clip_epsilon is not None:
            return torch.clamp(delta.nan_to_num(), -self.final_clip_epsilon, self.final_clip_epsilon)
        
    def forward_with_adversarial(self, image: Image.Image, target_label: int) -> tuple[
            torch.return_types.topk, torch.Tensor, torch.Tensor, torch.return_types.topk]:
        """
        Runs forward pass to find original labels, generates adversarial noise to classify as target label
        Returns original predictions, noise tensor, adversarial image, and new adversarial predictions
        """
        orig_preds = self.forward(image)
        orig_label = orig_preds.indices[0]  # Top pred label of original image
        target_label = torch.tensor([target_label])
        image_t = T.ToTensor()(image)
        noise_t = self.adversarial_noise(image_t, orig_label, target_label)
        adversarial_image_t = image_t + noise_t
        adversarial_preds = self.forward(adversarial_image_t)
        to_PIL_Image = T.ToPILImage()
        return orig_preds, to_PIL_Image(noise_t), to_PIL_Image(adversarial_image_t), adversarial_preds
