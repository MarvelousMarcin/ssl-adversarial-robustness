import torch
from attacks.loss import centroid_cosine_loss, embedding_drift_loss


def _get_norm_bounds(device):
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    lower = (0 - mean) / std
    upper = (1 - mean) / std
    return std, lower, upper


def fgsm_attack(model, images, labels, epsilon=0.03, centroids=None, loss_fn=None):
    std, lower, upper = _get_norm_bounds(images.device)
    eps_normalized = epsilon / std

    images = images.clone().detach().requires_grad_(True)

    embeddings = model.get_embedding(images)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    if loss_fn is not None:
        loss = loss_fn(embeddings, labels)
    else:
        loss = centroid_cosine_loss(embeddings, labels, centroids)

    loss.backward()

    adv_images = images + eps_normalized * images.grad.sign()
    adv_images = torch.max(torch.min(adv_images, upper), lower)

    return adv_images.detach()


def fgsm_attack_label_free(model, images, epsilon=0.03):
    std, lower, upper = _get_norm_bounds(images.device)
    eps_normalized = epsilon / std

    with torch.no_grad():
        clean_emb = model.get_embedding(images)
        clean_emb = torch.nn.functional.normalize(clean_emb, p=2, dim=1)

    images = images.clone().detach().requires_grad_(True)

    embeddings = model.get_embedding(images)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    loss = embedding_drift_loss(embeddings, clean_emb.detach())
    loss.backward()

    adv_images = images + eps_normalized * images.grad.sign()
    adv_images = torch.max(torch.min(adv_images, upper), lower)

    return adv_images.detach()
