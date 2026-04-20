import torch
import torch.nn.functional as F


def _get_norm_bounds(device):
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    lower = (0 - mean) / std
    upper = (1 - mean) / std
    return std, lower, upper


def _project(adv, orig, eps_n, lower, upper):
    delta = torch.max(torch.min(adv - orig, eps_n), -eps_n)
    return torch.max(torch.min(orig + delta, upper), lower)


def _init_random(orig, eps_n, lower, upper):
    delta = torch.empty_like(orig).uniform_(-1, 1) * eps_n
    return torch.max(torch.min(orig + delta, upper), lower).detach()


def _per_sample_centroid_loss(emb, labels, centroids):
    return 1.0 - F.cosine_similarity(emb, centroids[labels])


def _per_sample_drift_loss(emb, clean_ref):
    return 1.0 - F.cosine_similarity(emb, clean_ref)


def _apgd_checkpoints(n_iter):
    # APGD checkpoint schedule from Croce & Hein 2020, eq. 3.
    p = [0.0, 0.22]
    while True:
        nxt = p[-1] + max(p[-1] - p[-2] - 0.03, 0.06)
        if nxt >= 1.0:
            break
        p.append(nxt)
    return sorted({int(round(f * n_iter)) for f in p if 0 < int(round(f * n_iter)) < n_iter})


def _run_apgd(model, images, eps_n, steps, lower, upper, per_sample_loss_fn,
              rho=0.75, alpha_mom=0.75):
    device = images.device
    batch = images.size(0)

    x_adv = _init_random(images, eps_n, lower, upper)
    x_prev = x_adv.clone()
    x_best = x_adv.clone()

    step_mult = torch.ones(batch, 1, 1, 1, device=device)
    base_step = 2.0 * eps_n

    loss_best = torch.full((batch,), -float("inf"), device=device)
    loss_last_ckpt = loss_best.clone()
    counter = torch.zeros(batch, device=device)
    last_ckpt = 0

    ckpts = set(_apgd_checkpoints(steps))

    for k in range(steps):
        x_adv.requires_grad_(True)
        emb = F.normalize(model.get_embedding(x_adv), p=2, dim=1)
        loss = per_sample_loss_fn(emb)
        grad = torch.autograd.grad(loss.sum(), x_adv)[0]

        with torch.no_grad():
            mask = loss > loss_best
            loss_best = torch.where(mask, loss, loss_best)
            x_best = torch.where(mask.view(-1, 1, 1, 1), x_adv.detach(), x_best)
            counter += mask.float()

            z = x_adv + step_mult * base_step * grad.sign()
            z = _project(z, images, eps_n, lower, upper)

            if k == 0:
                x_new = z
            else:
                x_new = x_adv + alpha_mom * (z - x_adv) + (1.0 - alpha_mom) * (x_adv - x_prev)
                x_new = _project(x_new, images, eps_n, lower, upper)

            x_prev = x_adv.detach().clone()
            x_adv = x_new.detach()

        if (k + 1) in ckpts:
            interval = (k + 1) - last_ckpt
            cond1 = counter < rho * interval
            cond2 = (loss_best - loss_last_ckpt) < 1e-6
            halve = (cond1 | cond2).view(-1, 1, 1, 1)

            step_mult = torch.where(halve, step_mult * 0.5, step_mult)
            x_adv = torch.where(halve, x_best, x_adv)

            counter.zero_()
            loss_last_ckpt = loss_best.clone()
            last_ckpt = k + 1

    return x_best


def _eval_per_sample(model, adv, per_sample_loss_fn):
    with torch.no_grad():
        emb = model.get_embedding(adv)
        emb = F.normalize(emb, p=2, dim=1)
        return per_sample_loss_fn(emb)


def _restart_loop(model, images, steps, restarts, eps_n, lower, upper, per_sample_loss_fn):
    best_adv = images.clone().detach()
    best_loss = torch.full((images.size(0),), -float("inf"), device=images.device)
    for _ in range(restarts):
        adv = _run_apgd(model, images, eps_n, steps, lower, upper, per_sample_loss_fn)
        losses = _eval_per_sample(model, adv, per_sample_loss_fn)
        mask = losses > best_loss
        best_loss = torch.where(mask, losses, best_loss)
        best_adv = torch.where(mask.view(-1, 1, 1, 1), adv, best_adv)
    return best_adv


def apgd_attack(model, images, labels, epsilon=0.03, steps=100, restarts=1, centroids=None):
    std, lower, upper = _get_norm_bounds(images.device)
    eps_n = epsilon / std

    def per_sample(emb):
        return _per_sample_centroid_loss(emb, labels, centroids)

    return _restart_loop(model, images, steps, restarts, eps_n, lower, upper, per_sample)


def apgd_attack_label_free(model, images, epsilon=0.03, steps=100, restarts=1):
    std, lower, upper = _get_norm_bounds(images.device)
    eps_n = epsilon / std

    with torch.no_grad():
        clean_emb = F.normalize(model.get_embedding(images), p=2, dim=1)

    def per_sample(emb):
        return _per_sample_drift_loss(emb, clean_emb)

    return _restart_loop(model, images, steps, restarts, eps_n, lower, upper, per_sample)
