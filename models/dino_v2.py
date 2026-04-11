import torch
from models.base_model import BaseModel

class DINOv2Model(BaseModel):
    def __init__(self, device: torch.device, pool: str = "cls"):
        self.device = device
        self.pool = pool
        self.model = torch.hub.load('facebookresearch/dinov2', "dinov2_vitl14")
        self.model.eval().to(self.device)

    def get_embedding(self, x):
        x = x.to(self.device)
        if self.pool == "mean":
            feats = self.model.forward_features(x)
            return feats["x_norm_patchtokens"].mean(dim=1)
        return self.model(x)
