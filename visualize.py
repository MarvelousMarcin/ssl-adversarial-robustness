import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils.cache import Cache

os.makedirs('./results/figures', exist_ok=True)

MODELS = ["dino", "ijepa", "vit_sup", "resnet50"]
MODEL_LABELS = {"dino": "DINOv2", "ijepa": "I-JEPA", "vit_sup": "ViT-Sup", "resnet50": "ResNet50"}
COLORS = {"dino": "#1f77b4", "ijepa": "#ff7f0e", "vit_sup": "#2ca02c", "resnet50": "#d62728"}

DEFAULT_EPSILONS = [0.008, 0.016, 0.031, 0.063]
DEFAULT_DATASET = "imagenet"
DEFAULT_POOL = "mean"


def _cache_path(model, dataset, pool, attack=None, epsilon=None):
    tag = f"_{attack}_eps{epsilon}" if attack and attack != "none" else ""
    return f"./results/{model}_{dataset}_{pool}{tag}.pkl"


def load_results(path="./results/tables/all_results.json"):
    with open(path, "r") as f:
        return json.load(f)


def _filter(results, dataset, pool):
    out = []
    for r in results:
        if r.get("dataset") != dataset:
            continue
        if "pool" in r and r["pool"] != pool:
            continue
        out.append(r)
    return out

def plot_robustness_curves(results, dataset, pool, attack):
    metrics_to_plot = ["knn_acc", "linear_probe_acc", "recall@1", "alignment", "uniformity"]
    metric_titles = {
        "knn_acc": "KNN Accuracy", "linear_probe_acc": "Linear Probing Accuracy",
        "recall@1": "Recall@1", "alignment": "Alignment ↓", "uniformity": "Uniformity ↑"
    }

    rows = _filter(results, dataset, pool)
    data = {}
    for r in rows:
        model = r["model"]
        eps = r["epsilon"] if r["epsilon"] is not None else 0.0
        atk = r["attack"]
        if atk not in ("none", attack):
            continue
        data.setdefault(model, {})[eps] = r

    if not data:
        print(f"No results for dataset={dataset}, pool={pool}, attack={attack}")
        return

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(4 * len(metrics_to_plot), 4))
    if len(metrics_to_plot) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics_to_plot):
        for model in MODELS:
            if model not in data:
                continue
            epsilons = sorted(data[model].keys())
            values = [data[model][e].get(metric, None) for e in epsilons]
            if any(v is None for v in values):
                continue
            ax.plot(epsilons, values, 'o-', label=MODEL_LABELS[model],
                    color=COLORS[model], linewidth=2, markersize=5)
        ax.set_xlabel("Epsilon (ε)")
        ax.set_ylabel(metric_titles.get(metric, metric))
        ax.set_title(metric_titles.get(metric, metric))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Robustness — {attack.upper()} on {dataset.upper()} (pool={pool})", fontsize=14, y=1.02)
    plt.tight_layout()
    path = f"./results/figures/robustness_{attack}_{dataset}_{pool}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_clean_comparison(results, dataset, pool):
    metrics = ["knn_acc", "linear_probe_acc", "recall@1", "recall@5"]
    metric_labels = ["KNN Acc", "Linear Probe", "Recall@1", "Recall@5"]

    rows = _filter(results, dataset, pool)
    clean = {r["model"]: r for r in rows if r["attack"] == "none"}
    if not clean:
        print(f"No clean results for dataset={dataset}, pool={pool}")
        return

    models_present = [m for m in MODELS if m in clean]
    x = np.arange(len(metrics))
    width = 0.8 / len(models_present)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(models_present):
        values = [clean[model].get(m, 0) for m in metrics]
        ax.bar(x + i * width, values, width, label=MODEL_LABELS[model], color=COLORS[model])

    ax.set_ylabel("Score")
    ax.set_title(f"Clean Representation Quality — {dataset.upper()} (pool={pool})")
    ax.set_xticks(x + width * (len(models_present) - 1) / 2)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

    path = f"./results/figures/clean_comparison_{dataset}_{pool}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_tsne(model_name, dataset, pool, attack, epsilon, max_samples=3000):
    clean_cache = Cache(_cache_path(model_name, dataset, pool))
    adv_cache = Cache(_cache_path(model_name, dataset, pool, attack, epsilon))

    if not clean_cache.exists():
        print(f"Missing clean cache: {clean_cache.path}")
        return
    if not adv_cache.exists():
        print(f"Missing adversarial cache: {adv_cache.path}")
        return

    clean_data = clean_cache.load()
    adv_data = adv_cache.load()

    n = min(max_samples, len(clean_data["labels"]), len(adv_data["labels"]))
    idx = np.random.RandomState(42).choice(len(clean_data["labels"]), n, replace=False)

    clean_emb, clean_lab = clean_data["embeddings"][idx], clean_data["labels"][idx]
    adv_emb, adv_lab = adv_data["embeddings"][idx], adv_data["labels"][idx]

    all_emb = np.concatenate([clean_emb, adv_emb], axis=0)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    all_2d = tsne.fit_transform(all_emb)

    clean_2d = all_2d[:n]
    adv_2d = all_2d[n:]

    unique_labels = np.unique(clean_lab)
    num_classes = len(unique_labels)
    cmap = plt.cm.get_cmap('tab20', min(num_classes, 20))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for i, label in enumerate(unique_labels[:20]):
        mask = clean_lab == label
        ax1.scatter(clean_2d[mask, 0], clean_2d[mask, 1], c=[cmap(i % 20)],
                    s=5, alpha=0.6, label=str(label) if num_classes <= 20 else None)
        mask_adv = adv_lab == label
        ax2.scatter(adv_2d[mask_adv, 0], adv_2d[mask_adv, 1], c=[cmap(i % 20)],
                    s=5, alpha=0.6)

    ax1.set_title(f"{MODEL_LABELS.get(model_name, model_name)} — Clean")
    ax2.set_title(f"{MODEL_LABELS.get(model_name, model_name)} — {attack.upper()} (ε={epsilon})")

    for ax in (ax1, ax2):
        ax.set_xticks([])
        ax.set_yticks([])

    if num_classes <= 20:
        ax1.legend(fontsize=6, markerscale=3, ncol=2)

    fig.suptitle(f"t-SNE — {MODEL_LABELS.get(model_name, model_name)} on {dataset.upper()} (pool={pool})", fontsize=14)
    plt.tight_layout()
    path = f"./results/figures/tsne_{model_name}_{dataset}_{pool}_{attack}_eps{epsilon}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

def plot_embedding_shift(dataset, pool, attack, epsilons):
    from metrics.embedding_shift import embedding_shift, cosine_shift

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(epsilons))
    width = 0.8 / len(MODELS)

    for i, model in enumerate(MODELS):
        clean_cache = Cache(_cache_path(model, dataset, pool))
        if not clean_cache.exists():
            continue
        clean_data = clean_cache.load()

        l2_shifts = []
        cos_shifts = []

        for eps in epsilons:
            adv_cache = Cache(_cache_path(model, dataset, pool, attack, eps))
            if not adv_cache.exists():
                l2_shifts.append(0)
                cos_shifts.append(0)
                continue
            adv_data = adv_cache.load()
            n = min(len(clean_data["embeddings"]), len(adv_data["embeddings"]))
            l2_shifts.append(embedding_shift(clean_data["embeddings"][:n], adv_data["embeddings"][:n]))
            cos_shifts.append(cosine_shift(clean_data["embeddings"][:n], adv_data["embeddings"][:n]))

        ax1.bar(x + i * width, l2_shifts, width, label=MODEL_LABELS[model], color=COLORS[model])
        ax2.bar(x + i * width, cos_shifts, width, label=MODEL_LABELS[model], color=COLORS[model])

    for ax, title in [(ax1, "L2 Embedding Shift"), (ax2, "Cosine Embedding Shift")]:
        ax.set_xlabel("Epsilon (ε)")
        ax.set_ylabel("Shift")
        ax.set_title(f"{title} — {attack.upper()} on {dataset.upper()} (pool={pool})")
        ax.set_xticks(x + width * (len(MODELS) - 1) / 2)
        ax.set_xticklabels([str(e) for e in epsilons])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = f"./results/figures/embedding_shift_{attack}_{dataset}_{pool}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--pool", default=DEFAULT_POOL, choices=["cls", "mean"])
    parser.add_argument("--epsilons", nargs="+", type=float, default=DEFAULT_EPSILONS)
    parser.add_argument("--tsne_model", default=None, help="Generate only t-SNE for this model")
    parser.add_argument("--tsne_attack", default="pgd_lf")
    parser.add_argument("--tsne_epsilon", default=0.063, type=float)
    args = parser.parse_args()

    results_path = "./results/tables/all_results.json"
    attacks = ["fgsm", "pgd", "fgsm_lf", "pgd_lf"]

    if args.tsne_model:
        plot_tsne(args.tsne_model, args.dataset, args.pool, args.tsne_attack, args.tsne_epsilon)
    else:
        if os.path.exists(results_path):
            results = load_results(results_path)
            plot_clean_comparison(results, args.dataset, args.pool)
            for atk in attacks:
                plot_robustness_curves(results, args.dataset, args.pool, atk)
        else:
            print(f"No results file at {results_path}")

        for atk in attacks:
            plot_embedding_shift(args.dataset, args.pool, atk, args.epsilons)

        for model in MODELS:
            plot_tsne(model, args.dataset, args.pool, args.tsne_attack, args.tsne_epsilon)
