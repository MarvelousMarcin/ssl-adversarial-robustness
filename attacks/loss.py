import torch


def compute_centroids(reference_embeddings, reference_labels, num_classes, embed_dim, device, dtype):
    ref_emb = torch.tensor(reference_embeddings, dtype=dtype, device=device)
    ref_lab = torch.tensor(reference_labels, device=device)

    centroids = torch.zeros(num_classes, embed_dim, device=device, dtype=dtype)
    for c in torch.unique(ref_lab):
        mask = ref_lab == c
        centroids[c] = torch.nn.functional.normalize(ref_emb[mask].mean(dim=0, keepdim=True), p=2, dim=1)

    return centroids


def centroid_cosine_loss(embeddings, labels, centroids):
    target_centroids = centroids[labels]
    loss = -torch.nn.functional.cosine_similarity(embeddings, target_centroids).mean()
    return loss


def embedding_drift_loss(embeddings, clean_ref):
    loss = -torch.nn.functional.cosine_similarity(embeddings, clean_ref).mean()
    return loss
