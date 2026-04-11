import numpy as np
import json
import os
import argparse
import matplotlib.pyplot as plt
from metrics.cka import linear_CKA
from utils.cache import Cache

os.makedirs('./results/figures', exist_ok=True)
os.makedirs('./results/tables', exist_ok=True)

MODEL_NAMES = ["dino", "ijepa", "vit_sup", "resnet50"]
MODEL_LABELS = {"dino": "DINOv2", "ijepa": "I-JEPA", "vit_sup": "ViT-Sup", "resnet50": "ResNet50"}


def load_embeddings(model, dataset, attack="none", epsilon=None, max_samples=5000):
    if attack == "none":
        path = f"./results/{model}_{dataset}.pkl"
    else:
        path = f"./results/{model}_{dataset}_{attack}_eps{epsilon}.pkl"

    cache = Cache(path)
    if not cache.exists():
        return None, None
    data = cache.load()
    emb, lab = data["embeddings"], data["labels"]

    if len(emb) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(emb), max_samples, replace=False)
        emb, lab = emb[idx], lab[idx]
    return emb, lab


def plot_cka_matrix(matrix, labels, title, save_path):
    n = len(labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap='YlOrRd', vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=11, color="black" if val < 0.7 else "white")

    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def clean_vs_clean_cka(dataset):
    print(f"\n{'='*50}")
    print(f"CKA: Clean vs Clean — {dataset.upper()}")
    print(f"{'='*50}")

    embeddings = {}
    for model in MODEL_NAMES:
        emb, _ = load_embeddings(model, dataset)
        if emb is not None:
            embeddings[model] = emb
            print(f"  Loaded {model}: {emb.shape}")
        else:
            print(f"  Missing: {model}")

    models_present = [m for m in MODEL_NAMES if m in embeddings]
    n = len(models_present)
    matrix = np.full((n, n), np.nan)

    min_n = min(len(embeddings[m]) for m in models_present)
    for m in models_present:
        embeddings[m] = embeddings[m][:min_n]

    for i, m1 in enumerate(models_present):
        for j, m2 in enumerate(models_present):
            matrix[i, j] = linear_CKA(embeddings[m1], embeddings[m2])
            print(f"  CKA({MODEL_LABELS[m1]}, {MODEL_LABELS[m2]}) = {matrix[i, j]:.4f}")

    labels = [MODEL_LABELS[m] for m in models_present]
    plot_cka_matrix(matrix, labels,
                    f"CKA — Clean Representations\n{dataset.upper()}",
                    f"./results/figures/cka_clean_{dataset}.png")

    result = {"dataset": dataset, "models": models_present,
              "cka_matrix": matrix.tolist()}
    with open(f"./results/tables/cka_clean_{dataset}.json", "w") as f:
        json.dump(result, f, indent=2)


def clean_vs_adversarial_cka(dataset, attack, epsilon):
    print(f"\n{'='*50}")
    print(f"CKA: Clean vs {attack.upper()} (ε={epsilon}) — {dataset.upper()}")
    print(f"{'='*50}")

    results = {}
    for model in MODEL_NAMES:
        clean_emb, _ = load_embeddings(model, dataset, "none")
        adv_emb, _ = load_embeddings(model, dataset, attack, epsilon)

        if clean_emb is None or adv_emb is None:
            print(f"  {MODEL_LABELS[model]}: SKIPPED (missing data)")
            continue

        n = min(len(clean_emb), len(adv_emb))
        cka_score = linear_CKA(clean_emb[:n], adv_emb[:n])
        results[model] = cka_score
        print(f"  {MODEL_LABELS[model]}: CKA(clean, adv) = {cka_score:.4f}")

    if not results:
        return

    models_present = [m for m in MODEL_NAMES if m in results]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        [MODEL_LABELS[m] for m in models_present],
        [results[m] for m in models_present],
        color=[colors[MODEL_NAMES.index(m)] for m in models_present]
    )
    ax.set_ylabel("CKA Score")
    ax.set_title(f"CKA(Clean, Adversarial) — {attack.upper()} ε={epsilon}\n"
                 f"Higher = more robust (representation preserved)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, m in zip(bars, models_present):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{results[m]:.3f}", ha='center', fontsize=11)

    plt.tight_layout()
    path = f"./results/figures/cka_clean_vs_{attack}_{dataset}_eps{epsilon}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    with open(f"./results/tables/cka_{attack}_{dataset}_eps{epsilon}.json", "w") as f:
        json.dump({"dataset": dataset, "attack": attack, "epsilon": epsilon,
                   "cka_scores": {MODEL_LABELS[m]: round(results[m], 4) for m in models_present}}, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument("--attack", default=None, help="If set, also compute clean-vs-adversarial CKA")
    parser.add_argument("--epsilon", default=0.03, type=float)
    args = parser.parse_args()

    clean_vs_clean_cka(args.dataset)

    if args.attack:
        clean_vs_adversarial_cka(args.dataset, args.attack, args.epsilon)
    else:
        for attack in ["fgsm", "pgd"]:
            for eps in [0.01, 0.02, 0.03, 0.04, 0.08]:
                clean_vs_adversarial_cka(args.dataset, attack, eps)
