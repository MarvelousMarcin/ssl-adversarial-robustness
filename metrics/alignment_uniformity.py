import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def alignment(embeddings, labels, alpha=2):
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    total = 0.0
    count = 0
    for label in np.unique(labels):
        mask = labels == label
        class_embs = embeddings[mask]
        n = len(class_embs)
        if n < 2:
            continue
        dists = euclidean_distances(class_embs)
        triu = np.triu_indices(n, k=1)
        total += np.sum(dists[triu] ** alpha)
        count += len(triu[0])

    return total / count if count > 0 else 0.0

def uniformity(embeddings, t=2):
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    dists_sq = euclidean_distances(embeddings) ** 2

    n = len(embeddings)
    triu = np.triu_indices(n, k=1)
    pairwise = np.exp(-t * dists_sq[triu])

    return np.log(pairwise.mean())