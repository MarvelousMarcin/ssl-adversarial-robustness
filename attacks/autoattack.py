import torch
import torch.nn as nn
import torch.nn.functional as F


class _CentroidClassifier(nn.Module):
    """Embedding model wrapped as a classifier via cosine similarity to class centroids.

    Accepts inputs in [0, 1] pixel space (AutoAttack convention) and normalizes
    internally, so centroids/embeddings stay consistent with the rest of the pipeline.
    """

    def __init__(self, encoder, centroids, device, logit_scale=10.0):
        super().__init__()
        self.encoder = encoder
        self.logit_scale = logit_scale
        self.register_buffer("centroids", centroids)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        )

    def forward(self, x_01):
        x_norm = (x_01 - self.mean) / self.std
        emb = self.encoder.get_embedding(x_norm)
        emb = F.normalize(emb, p=2, dim=1)
        return self.logit_scale * emb @ self.centroids.t()


def autoattack_centroid(model, images_normalized, labels, epsilon, centroids, device,
                        version="standard", seed=0):
    """Full AutoAttack (APGD-CE + APGD-DLR + FAB + Square) on the centroid classifier.

    Requires `torchattacks`. Operates in [0, 1] pixel space; images coming in are
    ImageNet-normalized, so we invert normalization, run AA, then re-normalize
    the adversarials for downstream embedding extraction.
    """
    try:
        import torchattacks
    except ImportError as e:
        raise ImportError(
            "AutoAttack attack requires `torchattacks`. Install with `pip install torchattacks`."
        ) from e

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    images_01 = (images_normalized * std + mean).clamp(0.0, 1.0)

    classifier = _CentroidClassifier(model, centroids, device).eval()
    for p in classifier.parameters():
        p.requires_grad_(False)

    n_classes = int(centroids.shape[0])
    atk = torchattacks.AutoAttack(
        classifier,
        norm="Linf",
        eps=float(epsilon),
        version=version,
        n_classes=n_classes,
        seed=seed,
        verbose=False,
    )
    adv_01 = atk(images_01, labels)

    adv_normalized = (adv_01 - mean) / std
    return adv_normalized.detach()
