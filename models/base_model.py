import torch
class BaseModel:
    def get_embedding(self, x) -> torch.Tensor:
        raise NotImplementedError