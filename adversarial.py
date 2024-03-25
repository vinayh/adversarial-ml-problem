from PIL.Image import Image
import torch
import torch.nn.functional as F
from torchvision.models import regnet_x_800mf, RegNet_X_800MF_Weights


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
        
    def forward(self, image: Image) -> torch.return_types.topk:
        preprocessed = self.preprocess(image)
        out = self.model(torch.unsqueeze(preprocessed, 0)).squeeze(0)
        probs = F.softmax(out)
        return torch.topk(probs.detach(), k=5, sorted=True)
