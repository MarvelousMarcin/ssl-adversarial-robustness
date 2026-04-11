import numpy as np


def embedding_shift(clean_embeddings, adversarial_embeddings):
    diffs = clean_embeddings - adversarial_embeddings
    distances = np.linalg.norm(diffs, axis=1)
    return float(distances.mean())


def cosine_shift(clean_embeddings, adversarial_embeddings):

    clean_norm = clean_embeddings / (np.linalg.norm(clean_embeddings, axis=1, keepdims=True) + 1e-10)
    adv_norm = adversarial_embeddings / (np.linalg.norm(adversarial_embeddings, axis=1, keepdims=True) + 1e-10)
    cos_sim = np.sum(clean_norm * adv_norm, axis=1)
    return float((1 - cos_sim).mean())
