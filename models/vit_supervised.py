import torch
from torchvision import models
from models.base_model import BaseModel


class ViTSupervisedModel(BaseModel):
    def __init__(self, device: torch.device, pool: str = "cls"):
        self.device = device
        self.pool = pool
        self.model = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1)
        self.model.heads = torch.nn.Identity()
        self.model.eval().to(self.device)

    def get_embedding(self, x):
        x = x.to(self.device)
        # Replicate torchvision ViT forward to expose patch tokens
        x = self.model._process_input(x)
        n = x.shape[0]
        cls = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.model.encoder(x)
        if self.pool == "mean":
            return x[:, 1:].mean(dim=1)
        return x[:, 0]
