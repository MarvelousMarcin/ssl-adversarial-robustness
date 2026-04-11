import torch
from torchvision import models
from models.base_model import BaseModel

class ResNet50Model(BaseModel):
    def __init__(self, device: torch.device):
        self.device = device
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.model = torch.nn.Sequential(*list(model.children())[:-1])
        self.model.eval().to(self.device)

    def get_embedding(self, x):
        x = x.to(self.device)
        emb = self.model(x)
        return emb.squeeze(-1).squeeze(-1)
