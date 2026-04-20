"""Cross-model transfer attack eval.

Generate adversarials with a source model, embed them with a target model, then
evaluate target-side metrics against the target's clean reference. Only label-free
attacks are supported — centroid-based attacks bind perturbations to the source's
own centroid geometry and wouldn't be a meaningful transfer test.

Usage:
    python transfer_attack.py --source_model dino --source_pool mean \\
        --target_model ijepa --target_pool mean \\
        --dataset cifar100 --attack apgd_lf --epsilon 0.031
"""

import argparse
import csv
import json
import logging
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset.dataset_manager import DatasetManager
from models.dino_v2 import DINOv2Model
from models.jepa import IJEPAModel
from models.vit_supervised import ViTSupervisedModel
from models.resnet50 import ResNet50Model

from attacks.pgd import pgd_attack_label_free
from attacks.apgd import apgd_attack_label_free
from attacks.fgsm import fgsm_attack_label_free

from metrics.recall import recall
from metrics.linear_probing import linear_probe
from metrics.knn_accuracy import knn_accuracy
from metrics.alignment_uniformity import alignment, uniformity
from metrics.embedding_shift import embedding_shift, cosine_shift

from utils.cache import Cache


os.makedirs("./results/logs", exist_ok=True)
os.makedirs("./results/tables", exist_ok=True)
log_file = f"./results/logs/transfer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def build_model(name, pool, device, checkpoint):
    if name == "dino":
        return DINOv2Model(device, pool=pool)
    if name == "ijepa":
        return IJEPAModel(device, checkpoint_path=checkpoint)
    if name == "vit_sup":
        return ViTSupervisedModel(device, pool=pool)
    if name == "resnet50":
        return ResNet50Model(device)
    raise ValueError(f"Unknown model: {name}")


parser = argparse.ArgumentParser()
parser.add_argument("--source_model", required=True,
                    choices=["dino", "ijepa", "vit_sup", "resnet50"])
parser.add_argument("--source_pool", default="mean", choices=["cls", "mean"])
parser.add_argument("--target_model", required=True,
                    choices=["dino", "ijepa", "vit_sup", "resnet50"])
parser.add_argument("--target_pool", default="mean", choices=["cls", "mean"])
parser.add_argument("--dataset", required=True,
                    choices=["car", "imagenet", "cifar100", "cifar10"])
parser.add_argument("--attack", default="apgd_lf",
                    choices=["fgsm_lf", "pgd_lf", "apgd_lf"],
                    help="Only label-free attacks make sense for transfer.")
parser.add_argument("--epsilon", type=float, default=0.031)
parser.add_argument("--pgd_steps", type=int, default=20)
parser.add_argument("--pgd_restarts", type=int, default=2)
parser.add_argument("--pgd_alpha", type=float, default=None)
parser.add_argument("--apgd_steps", type=int, default=100)
parser.add_argument("--apgd_restarts", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--attack_batch_size", type=int, default=16)
parser.add_argument("--subset_size", type=int, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--checkpoint", default="./ijepa/checkpoints/IN1K-vit.h.14-300e.pth.tar")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
logger.info(f"DEVICE = {device}")

if args.source_model == args.target_model and args.source_pool == args.target_pool:
    logger.info("Source == target (same model+pool). Skipping self-transfer; use main.py.")
    raise SystemExit(0)

if args.pgd_alpha is None:
    args.pgd_alpha = 2.5 * args.epsilon / args.pgd_steps

logger.info(
    f"TRANSFER  src={args.source_model}:{args.source_pool}  "
    f"tgt={args.target_model}:{args.target_pool}  "
    f"dataset={args.dataset}  attack={args.attack}  eps={args.epsilon}"
)

seed_tag = f"_seed{args.seed}" if args.seed != 0 else ""
transfer_tag = f"_from_{args.source_model}-{args.source_pool}"
cache_name = (
    f"./results/{args.target_model}_{args.dataset}_{args.target_pool}"
    f"_{args.attack}_eps{args.epsilon}{transfer_tag}{seed_tag}.pkl"
)
cache = Cache(cache_name)

clean_cache_path = f"./results/{args.target_model}_{args.dataset}_{args.target_pool}.pkl"
if not os.path.exists(clean_cache_path):
    logger.error(
        f"Target clean cache missing: {clean_cache_path}. "
        f"Run `python main.py --model {args.target_model} "
        f"--dataset {args.dataset} --pool {args.target_pool} --attack none` first."
    )
    raise SystemExit(1)


if not cache.exists():
    # Phase 1: attack on source, collect adv images on CPU.
    logger.info("Loading source model and generating adversarials...")
    source = build_model(args.source_model, args.source_pool, device, args.checkpoint)

    dm_atk = DatasetManager(batch_size=args.attack_batch_size, num_workers=4)
    loader_fn = {
        "car": dm_atk.get_cars, "imagenet": dm_atk.get_imagenet,
        "cifar100": dm_atk.get_cifar100, "cifar10": dm_atk.get_cifar10,
    }
    loader = loader_fn[args.dataset](subset_size=args.subset_size, split="test")

    adv_batches = []
    label_batches = []
    for image, label in tqdm(loader, f"Attack on source ({args.source_model})"):
        image = image.to(device)
        if args.attack == "apgd_lf":
            adv = apgd_attack_label_free(
                source, image, epsilon=args.epsilon,
                steps=args.apgd_steps, restarts=args.apgd_restarts,
            )
        elif args.attack == "pgd_lf":
            adv = pgd_attack_label_free(
                source, image, epsilon=args.epsilon,
                alpha=args.pgd_alpha, steps=args.pgd_steps,
                restarts=args.pgd_restarts,
            )
        elif args.attack == "fgsm_lf":
            adv = fgsm_attack_label_free(source, image, epsilon=args.epsilon)
        adv_batches.append(adv.detach().cpu())
        label_batches.append(label.detach().cpu())

    del source
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Phase 2: embed adv images with target model.
    logger.info("Loading target model and embedding adversarials...")
    target = build_model(args.target_model, args.target_pool, device, args.checkpoint)

    emb_batches = []
    for adv in tqdm(adv_batches, f"Embed on target ({args.target_model})"):
        adv = adv.to(device)
        with torch.no_grad():
            emb = target.get_embedding(adv)
            emb = F.normalize(emb, p=2, dim=1)
        emb_batches.append(emb.detach().cpu().to(torch.float32))

    del target
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    embeddings = torch.cat(emb_batches, dim=0).numpy()
    labels = torch.cat(label_batches, dim=0).numpy()
    cache.save(embeddings=embeddings, labels=labels)
else:
    logger.info(f"Loading cached transfer embeddings from {cache_name}")
    data = cache.load()
    embeddings, labels = data["embeddings"], data["labels"]


logger.info("=" * 60)
logger.info("METRICS (target-side)")
logger.info("=" * 60)

clean_data = Cache(clean_cache_path).load()
clean_emb = clean_data["embeddings"]
clean_labels = clean_data["labels"]
n = min(len(clean_emb), len(embeddings))

recall_ks = [1, 5, 10, 20]
align_score = alignment(embeddings, labels)
uniform_score = uniformity(embeddings)
l2_shift_val = embedding_shift(clean_emb[:n], embeddings[:n])
cos_shift_val = cosine_shift(clean_emb[:n], embeddings[:n])
recall_scores = recall(
    embeddings[:n], labels[:n], recall_ks,
    gallery_embeddings=clean_emb[:n], gallery_labels=clean_labels[:n],
    exclude_self=True,
)
knn_acc = knn_accuracy(
    clean_emb[:n], clean_labels[:n],
    test_embeddings=embeddings[:n], test_labels=labels[:n],
)
lp_acc = linear_probe(clean_emb[:n], clean_labels[:n], embeddings[:n], labels[:n])

logger.info(f"RECALL@{recall_ks} = {recall_scores}")
logger.info(f"LINEAR PROBING ACC = {lp_acc:.4f}")
logger.info(f"KNN ACC = {knn_acc:.4f}")
logger.info(f"EMBEDDING SHIFT (L2) = {l2_shift_val:.6f}")
logger.info(f"EMBEDDING SHIFT (Cosine) = {cos_shift_val:.6f}")
logger.info(f"ALIGNMENT = {align_score:.6f}")
logger.info(f"UNIFORMITY = {uniform_score:.6f}")

result_row = {
    "timestamp": datetime.now().isoformat(),
    "model": args.target_model,
    "dataset": args.dataset,
    "pool": args.target_pool,
    "subset_size": args.subset_size,
    "attack": args.attack,
    "epsilon": args.epsilon,
    "seed": args.seed,
    "num_samples": int(len(labels)),
    "num_classes": int(len(np.unique(labels))),
    "transfer_source_model": args.source_model,
    "transfer_source_pool": args.source_pool,
}
for i, k in enumerate(recall_ks):
    result_row[f"recall@{k}"] = round(float(recall_scores[i]), 4)
result_row["linear_probe_acc"] = round(float(lp_acc), 4)
result_row["knn_acc"] = round(float(knn_acc), 4)
result_row["alignment"] = round(float(align_score), 6)
result_row["uniformity"] = round(float(uniform_score), 6)
result_row["embedding_shift_l2"] = round(float(l2_shift_val), 6)
result_row["embedding_shift_cosine"] = round(float(cos_shift_val), 6)

json_path = "./results/tables/all_results.json"
if os.path.exists(json_path):
    with open(json_path, "r") as f:
        all_results = json.load(f)
else:
    all_results = []
all_results.append(result_row)
with open(json_path, "w") as f:
    json.dump(all_results, f, indent=2)

# Separate CSV for transfer results (schema differs from main runs).
csv_path = "./results/tables/transfer_results.csv"
file_exists = os.path.exists(csv_path)
with open(csv_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=result_row.keys())
    if not file_exists:
        writer.writeheader()
    writer.writerow(result_row)

logger.info(f"Results saved to {json_path} and {csv_path}")
logger.info("=" * 60)
