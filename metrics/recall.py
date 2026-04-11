import numpy as np


def recall(embeddings, labels, k_list: list[int], gallery_embeddings=None, gallery_labels=None,
           exclude_self=False):
    from sklearn.metrics.pairwise import cosine_similarity

    if gallery_embeddings is not None and gallery_labels is not None:
        # Cross-set recall: query embeddings vs gallery embeddings
        sims = cosine_similarity(embeddings, gallery_embeddings)
        if exclude_self:
            assert len(embeddings) == len(gallery_embeddings), \
                "exclude_self requires query and gallery to be index-aligned (same length)."
            np.fill_diagonal(sims, -1)
        query_labels = labels
        ref_labels = gallery_labels
    else:
        # Same-set recall: exclude self-match
        sims = cosine_similarity(embeddings)
        np.fill_diagonal(sims, -1)
        query_labels = labels
        ref_labels = labels

    recalls = []
    for k in k_list:
        hits = 0
        for i in range(len(query_labels)):
            top_k = np.argsort(sims[i])[-k:]
            top_k_labels = ref_labels[top_k]
            hits += int(np.any(top_k_labels == query_labels[i]))
        recalls.append(hits / len(query_labels))
    return recalls
