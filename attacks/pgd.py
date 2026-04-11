import torch
import torch.nn.functional as F
from attacks.loss import centroid_cosine_loss, embedding_drift_loss


def _get_norm_bounds(device):
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    lower = (0 - mean) / std
    upper = (1 - mean) / std
    return std, lower, upper


def _pgd_project(adv_images, images, eps_normalized, lower, upper):
    perturbation = torch.max(torch.min(adv_images - images, eps_normalized), -eps_normalized)
    return torch.max(torch.min(images + perturbation, upper), lower).detach()


def _pgd_init(images, eps_normalized, lower, upper):
    adv_images = images.clone().detach()
    delta = torch.empty_like(images).uniform_(-1, 1) * eps_normalized
    return torch.max(torch.min(adv_images + delta, upper), lower).detach()


def _per_sample_centroid_loss(emb, labels, centroids):
    # Higher = more successful attack (we want to MAX this).
    return 1.0 - F.cosine_similarity(emb, centroids[labels])


def _per_sample_drift_loss(emb, clean_ref):
    return 1.0 - F.cosine_similarity(emb, clean_ref)


def _run_pgd(model, images, eps_normalized, alpha_normalized, steps,
             lower, upper, per_sample_loss_fn):
    adv = _pgd_init(images, eps_normalized, lower, upper)
    for _ in range(steps):
        adv.requires_grad_(True)
        emb = model.get_embedding(adv)
        emb = F.normalize(emb, p=2, dim=1)
        loss = per_sample_loss_fn(emb).mean()
        loss.backward()
        with torch.no_grad():
            adv = adv + alpha_normalized * adv.grad.sign()
            adv = _pgd_project(adv, images, eps_normalized, lower, upper)
    return adv


def _eval_per_sample(model, adv, per_sample_loss_fn):
    with torch.no_grad():
        emb = model.get_embedding(adv)
        emb = F.normalize(emb, p=2, dim=1)
        return per_sample_loss_fn(emb)


def _restart_loop(model, images, steps, restarts, eps_normalized, alpha_normalized,
                  lower, upper, per_sample_loss_fn):
    best_adv = images.clone().detach()
    best_loss = torch.full((images.size(0),), -float("inf"), device=images.device)
    for _ in range(restarts):
        adv = _run_pgd(model, images, eps_normalized, alpha_normalized, steps,
                       lower, upper, per_sample_loss_fn)
        losses = _eval_per_sample(model, adv, per_sample_loss_fn)
        mask = losses > best_loss
        best_loss = torch.where(mask, losses, best_loss)
        mask_e = mask.view(-1, 1, 1, 1)
        best_adv = torch.where(mask_e, adv, best_adv)
    return best_adv


def pgd_attack(model, images, labels, epsilon=0.03, alpha=0.007, steps=10,
               restarts=1, centroids=None, loss_fn=None):
    std, lower, upper = _get_norm_bounds(images.device)
    eps_n = epsilon / std
    alpha_n = alpha / std

    def per_sample(emb):
        return _per_sample_centroid_loss(emb, labels, centroids)

    return _restart_loop(model, images, steps, restarts, eps_n, alpha_n,
                         lower, upper, per_sample)


def pgd_attack_label_free(model, images, epsilon=0.03, alpha=0.007, steps=10,
                          restarts=1):
    std, lower, upper = _get_norm_bounds(images.device)
    eps_n = epsilon / std
    alpha_n = alpha / std

    with torch.no_grad():
        clean_emb = model.get_embedding(images)
        clean_emb = F.normalize(clean_emb, p=2, dim=1)

    def per_sample(emb):
        return _per_sample_drift_loss(emb, clean_emb)

    return _restart_loop(model, images, steps, restarts, eps_n, alpha_n,
                         lower, upper, per_sample)
