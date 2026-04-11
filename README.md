# Are Self-Supervised Representations Robust? A Comparative Study

**A Comparative Study of Representation Quality and Adversarial Robustness in DINOv2, I-JEPA, and Vision Transformers**

Self-supervised learning (SSL) methods like DINOv2 and I-JEPA learn powerful visual representations without labels -- but how well do those representations hold up under adversarial attack? This project provides a systematic benchmark comparing the embedding quality and adversarial robustness of four vision models across multiple datasets, attacks, and evaluation metrics.

---

## Models Under Evaluation

| Model | Type | Architecture | Params |
|-------|------|-------------|--------|
| **DINOv2** | Self-supervised (distillation) | ViT-L/14 | ~300M |
| **I-JEPA** | Self-supervised (predictive) | ViT-H/14 | ~630M |
| **ViT-Supervised** | Supervised (ImageNet-1K) | ViT-L/16 | ~300M |
| **ResNet50** | Supervised (ImageNet-1K v2) | ResNet-50 | ~25M |

All models output L2-normalized embeddings. ViT-based models support configurable pooling (`cls` token vs `mean` patch tokens); I-JEPA always uses mean pooling.

## Attack Methods

The benchmark includes both **label-dependent** and **label-free** adversarial attacks that operate directly in the embedding space:

- **FGSM** -- Single-step gradient attack (fast, diagnostic baseline)
- **PGD** -- Multi-step iterative attack with random restarts (stronger, headline results)

Each attack has two variants:
- **Centroid-based** (requires labels): pushes embeddings *away* from their class centroid
- **Label-free**: maximizes embedding drift from the clean reference -- no labels needed, enabling fair cross-model comparison

Perturbation budgets: `epsilon in {0.008, 0.016, 0.031, 0.063}`, normalized per-channel by ImageNet statistics.

## Evaluation Metrics

### Representation Quality
| Metric | What it measures |
|--------|-----------------|
| **Recall@k** | Retrieval accuracy -- do nearest neighbors share the same class? |
| **KNN Accuracy** | k=5 nearest-neighbor classification (no training needed) |
| **Linear Probing** | Logistic regression on frozen embeddings (measures linear separability) |
| **Alignment** | Intra-class compactness -- are same-class embeddings close together? |
| **Uniformity** | Are embeddings spread uniformly on the hypersphere? |

### Adversarial Robustness
| Metric | What it measures |
|--------|-----------------|
| **Embedding Shift (L2)** | Mean L2 distance between clean and adversarial embeddings |
| **Embedding Shift (Cosine)** | Mean cosine dissimilarity (1 - cos_sim) under attack |
| **CKA** | Centered Kernel Alignment between clean and perturbed representation spaces |
| **Degradation curves** | How each quality metric decays as epsilon increases |

## Datasets

| Dataset | Classes | Split used |
|---------|---------|-----------|
| ImageNet (Imagenette) | 10 | test |
| CIFAR-10 | 10 | test |
---

## Project Structure

```
ssl_research/
├── main.py                 # End-to-end pipeline: embed -> attack -> evaluate
├── run_all.sh              # Orchestrates full experiment grid
├── visualize.py            # Publication-quality plots (robustness curves, t-SNE, bar charts)
├── cka_analysis.py         # CKA similarity analysis (cross-model & clean-vs-adversarial)
│
├── models/
│   ├── base_model.py       # Abstract base class
│   ├── dino_v2.py          # DINOv2 ViT-L/14 (Facebook Research)
│   ├── jepa.py             # I-JEPA ViT-H/14 (from checkpoint)
│   ├── vit_supervised.py   # Supervised ViT-L/16 (torchvision)
│   └── resnet50.py         # ResNet50 (torchvision, ImageNet-1K v2)
│
├── attacks/
│   ├── loss.py             # Centroid computation & loss functions
│   ├── fgsm.py             # FGSM (centroid-based + label-free)
│   └── pgd.py              # PGD with random restarts (centroid-based + label-free)
│
├── metrics/
│   ├── recall.py           # Recall@k retrieval metric
│   ├── linear_probing.py   # Linear probe (sklearn LogisticRegression)
│   ├── knn_accuracy.py     # KNN accuracy (k=5, cosine distance)
│   ├── alignment_uniformity.py  # SSL-specific alignment & uniformity
│   ├── embedding_shift.py  # L2 and cosine embedding shift
│   └── cka.py              # Linear CKA (representation similarity)
│
├── dataset/
│   └── dataset_manager.py  # Unified loader for all datasets (224x224, ImageNet norm)
│
├── utils/
│   └── cache.py            # Pickle-based embedding cache
│
├── results/                # Generated outputs (gitignored)
│   ├── *.pkl               # Cached embeddings & centroids
│   ├── tables/             # all_results.json, all_results.csv
│   ├── logs/               # Timestamped experiment logs
│   └── figures/            # Generated plots
│
└── ijepa/                  # I-JEPA source (ViT implementation + checkpoint)
```

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

Requires: `torch`, `torchvision`, `transformers`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `tqdm`, `pyyaml`

### Running a Single Experiment

```bash
# Clean evaluation
python main.py --model dino --dataset imagenet --attack none --pool mean

# PGD label-free attack (eps=0.031)
python main.py --model dino --dataset imagenet --attack pgd_lf --epsilon 0.031 \
    --pgd_steps 20 --pgd_restarts 5 --pool mean
```

### Running the Full Benchmark

```bash
bash run_all.sh
```

This runs all 5 phases across 4 models x 2 datasets x 4 epsilon values:

1. **Clean baselines** -- extract and evaluate unperturbed embeddings
2. **FGSM (centroid)** -- single-step centroid-based attack
3. **PGD (centroid)** -- multi-step centroid-based attack
4. **FGSM (label-free)** -- fair single-step comparison
5. **PGD (label-free)** -- headline attack for the paper

### Generating Visualizations

```bash
python visualize.py     # Robustness curves, bar charts, t-SNE
python cka_analysis.py  # CKA heatmaps and clean-vs-adversarial analysis
```

## Pipeline Overview

```
                    +-----------+
                    |  Dataset  |
                    |  Loader   |
                    +-----+-----+
                          |
                    +-----v-----+
                    |   Model   |  DINOv2 / I-JEPA / ViT / ResNet50
                    |  Encoder  |
                    +-----+-----+
                          |
              +-----------+-----------+
              |                       |
        +-----v-----+          +-----v-----+
        |   Clean   |          | Adversarial|
        | Embedding |          |   Attack   |  FGSM / PGD (centroid or label-free)
        +-----+-----+          +-----+------+
              |                       |
              +-----------+-----------+
                          |
                    +-----v-----+
                    |  Metrics  |  Recall@k, KNN, Linear Probe,
                    | Evaluation|  Alignment, Uniformity, Shift, CKA
                    +-----+-----+
                          |
                    +-----v-----+
                    |  Results  |  JSON / CSV / Plots / Logs
                    +-----------+
```

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `dino` | Model to evaluate (`dino`, `ijepa`, `vit_sup`, `resnet50`) |
| `--dataset` | `car` | Dataset (`car`, `imagenet`, `cifar100`, `cifar10`) |
| `--attack` | `none` | Attack type (`none`, `fgsm`, `pgd`, `fgsm_lf`, `pgd_lf`) |
| `--epsilon` | `0.03` | Perturbation budget |
| `--pgd_steps` | `20` | PGD iteration count |
| `--pgd_restarts` | `5` | Random restarts (keeps per-sample worst case) |
| `--pool` | `mean` | Token pooling strategy (`cls`, `mean`) |
| `--batch_size` | `128` | Inference batch size |
| `--attack_batch_size` | `16` | Attack batch size (smaller to fit gradients in VRAM) |
| `--subset_size` | `None` | Limit dataset to first N samples (for debugging) |

## Design Decisions

- **Label-free attacks as headline**: centroid-based attacks implicitly give the attacker access to the label distribution, biasing results toward supervised models. The label-free variant (maximize embedding drift) provides a fairer comparison.
- **Mean pooling by default**: aligns DINOv2 and supervised ViT with I-JEPA's native pooling, ensuring representations are structurally comparable.
- **Caching**: embeddings and centroids are pickled to disk, making it fast to rerun analysis or add new metrics without re-extracting.
- **Reproducibility**: all sampling uses `random_state=42`; attack loops are deterministic given the same cache state.
