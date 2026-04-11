import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ijepa'))

import ijepa.src.models.vision_transformer as vit
from models.base_model import BaseModel


class IJEPAModel(BaseModel):
    def __init__(self, device: torch.device, checkpoint_path: str):
        self.device = device
        self.encoder = vit.vit_huge(patch_size=14, img_size=[224])
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = {k.replace('module.', '', 1): v for k, v in checkpoint['encoder'].items()}
        self.encoder.load_state_dict(state_dict)
        self.encoder.eval().to(self.device)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        patch_tokens = self.encoder(x)  # [B, N, D]
        return patch_tokens.mean(dim=1)  # [B, D]
