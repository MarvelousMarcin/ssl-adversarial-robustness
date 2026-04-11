import torch
import logging
import os
import json
import csv
from datetime import datetime
from dataset.dataset_manager import DatasetManager
from models.dino_v2 import DINOv2Model
from models.jepa import IJEPAModel
from models.vit_supervised import ViTSupervisedModel
from models.resnet50 import ResNet50Model
from tqdm import tqdm
import argparse
from utils.cache import Cache
from metrics.recall import recall
from metrics.linear_probing import linear_probe
from metrics.knn_accuracy import knn_accuracy
from metrics.alignment_uniformity import alignment, uniformity
from attacks.fgsm import fgsm_attack, fgsm_attack_label_free
from attacks.pgd import pgd_attack, pgd_attack_label_free
from attacks.loss import compute_centroids
from metrics.embedding_shift import embedding_shift, cosine_shift
import numpy as np

os.makedirs('./results/logs', exist_ok=True)
os.makedirs('./results/tables', exist_ok=True)
log_file = f'./results/logs/run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
logger.info(f"DEVICE = {device}")

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--dataset", default="car", choices=["car", "imagenet", "cifar100", "cifar10"])
parser.add_argument("--model", default="dino", choices=["dino", "ijepa", "vit_sup", "resnet50"])
parser.add_argument("--checkpoint", default="./ijepa/checkpoints/IN1K-vit.h.14-300e.pth.tar",
                    help="Path to model checkpoint (required for ijepa)")
parser.add_argument("--subset_size", default=None, type=int, help="Limit dataset to first N samples")
parser.add_argument("--attack", default="none", choices=["none", "fgsm", "pgd", "fgsm_lf", "pgd_lf"],
                    help="Adversarial attack: fgsm/pgd (centroid-based) or fgsm_lf/pgd_lf (label-free)")
parser.add_argument("--epsilon", default=0.03, type=float, help="Perturbation budget for adversarial attacks")
parser.add_argument("--pgd_steps", default=20, type=int, help="Number of PGD iterations")
parser.add_argument("--pgd_restarts", default=5, type=int,
                    help="Number of PGD random restarts (per-sample worst case kept)")
parser.add_argument("--pgd_alpha", default=None, type=float, help="PGD step size")
parser.add_argument("--pool", default="mean", choices=["cls", "mean"],
                    help="Token pooling for ViT backbones (I-JEPA is always mean — this matches DINOv2/ViT-sup)")
parser.add_argument("--attack_batch_size", default=16, type=int,
                    help="Batch size during attack (smaller to avoid OOM due to gradient computation)")

args = parser.parse_args()

dm = DatasetManager(batch_size=args.batch_size, num_workers=4)

logger.info(f"Model = {args.model}  Dataset = {args.dataset}  Attack = {args.attack}  Epsilon = {args.epsilon}")
loader = None
model = None

loader_fn = {"car": dm.get_cars, "imagenet": dm.get_imagenet,
             "cifar100": dm.get_cifar100, "cifar10": dm.get_cifar10}

loader = loader_fn[args.dataset](subset_size=args.subset_size, split="test")

if args.model == "dino":
    model = DINOv2Model(device, pool=args.pool)
elif args.model == "ijepa":
    model = IJEPAModel(device, checkpoint_path=args.checkpoint)
elif args.model == "vit_sup":
    model = ViTSupervisedModel(device, pool=args.pool)
elif args.model == "resnet50":
    model = ResNet50Model(device)

if args.pgd_alpha is None:
    args.pgd_alpha = 2.5 * args.epsilon / args.pgd_steps

attack_tag = f"_{args.attack}_eps{args.epsilon}" if args.attack != "none" else ""
cache_name = f"./results/{args.model}_{args.dataset}_{args.pool}{attack_tag}.pkl"
cache = Cache(cache_name)

if not cache.exists():
    embeddings = []
    labels_list = []

    if args.attack == "none":
        with torch.no_grad():
            for image, label in tqdm(loader, "Extracting clean embeddings"):
                image = image.to(device)
                emb_img = model.get_embedding(image)
                emb_img = torch.nn.functional.normalize(emb_img, p=2, dim=1)
                embeddings.append(emb_img.detach().cpu().to(torch.float32))
                labels_list.append(label.detach().cpu())
    else:
        is_label_free = args.attack.endswith("_lf")
        centroids = None

        if not is_label_free:
            centroid_cache_path = f"./results/centroids_{args.model}_{args.dataset}_{args.pool}.pkl"
            centroid_cache = Cache(centroid_cache_path)

            if centroid_cache.exists():
                logger.info(f"Loading cached centroids from {centroid_cache_path}")
                centroids_np = centroid_cache.load()["centroids"]
                centroids = torch.tensor(centroids_np, device=device, dtype=torch.float32)
            else:
                logger.info("Computing reference embeddings for attack centroids from train split...")
                dm_train = DatasetManager(batch_size=args.batch_size, num_workers=4)
                train_loader_fn = {"car": dm_train.get_cars, "imagenet": dm_train.get_imagenet,
                                   "cifar100": dm_train.get_cifar100, "cifar10": dm_train.get_cifar10}
                train_loader = train_loader_fn[args.dataset](split="train")
                ref_embeddings = []
                ref_labels = []
                with torch.no_grad():
                    for image, label in tqdm(train_loader, "Building reference centroids"):
                        image = image.to(device)
                        emb = model.get_embedding(image)
                        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                        ref_embeddings.append(emb.detach().cpu().to(torch.float32))
                        ref_labels.append(label.detach().cpu())
                ref_embeddings_np = torch.cat(ref_embeddings, dim=0).numpy()
                ref_labels_np = torch.cat(ref_labels, dim=0).numpy()

                embed_dim = ref_embeddings_np.shape[1]
                num_classes = int(ref_labels_np.max()) + 1
                centroids = compute_centroids(
                    ref_embeddings_np, ref_labels_np,
                    num_classes, embed_dim, device, torch.float32
                )
                centroid_cache.save(centroids=centroids.detach().cpu().numpy())
                del ref_embeddings, ref_labels, ref_embeddings_np, ref_labels_np, train_loader
                torch.cuda.empty_cache()

        attack_name = args.attack.upper()
        dm_attack = DatasetManager(batch_size=args.attack_batch_size, num_workers=4)
        attack_loader_fn = {"car": dm_attack.get_cars, "imagenet": dm_attack.get_imagenet,
                            "cifar100": dm_attack.get_cifar100, "cifar10": dm_attack.get_cifar10}
        attack_loader = attack_loader_fn[args.dataset](subset_size=args.subset_size, split="test")

        for image, label in tqdm(attack_loader, f"Running {attack_name} attack (eps={args.epsilon})"):
            image, label = image.to(device), label.to(device)

            if args.attack == "fgsm":
                adv_images = fgsm_attack(
                    model, image, label, epsilon=args.epsilon,
                    centroids=centroids,
                )
            elif args.attack == "pgd":
                adv_images = pgd_attack(
                    model, image, label, epsilon=args.epsilon,
                    alpha=args.pgd_alpha, steps=args.pgd_steps,
                    restarts=args.pgd_restarts,
                    centroids=centroids,
                )
            elif args.attack == "fgsm_lf":
                adv_images = fgsm_attack_label_free(
                    model, image, epsilon=args.epsilon,
                )
            elif args.attack == "pgd_lf":
                adv_images = pgd_attack_label_free(
                    model, image, epsilon=args.epsilon,
                    alpha=args.pgd_alpha, steps=args.pgd_steps,
                    restarts=args.pgd_restarts,
                )

            with torch.no_grad():
                emb_adv = model.get_embedding(adv_images)
                emb_adv = torch.nn.functional.normalize(emb_adv, p=2, dim=1)

            embeddings.append(emb_adv.detach().cpu().to(torch.float32))
            labels_list.append(label.detach().cpu())

    embeddings = torch.cat(embeddings, dim=0).numpy()
    labels = torch.cat(labels_list, dim=0).numpy()
    cache.save(embeddings=embeddings, labels=labels)
else:
    logger.info(f"Loading cached embeddings from {cache_name}")
    data = cache.load()
    embeddings, labels = data["embeddings"], data["labels"]


logger.info("=" * 60)
logger.info("METRICS")
logger.info("=" * 60)

recall_ks = [1, 5, 10, 20]
align_score = alignment(embeddings, labels)
uniform_score = uniformity(embeddings)

l2_shift_val = None
cos_shift_val = None
clean_cache_path = f"./results/{args.model}_{args.dataset}_{args.pool}.pkl"

from sklearn.model_selection import train_test_split

if args.attack == "none":
    recall_scores = recall(embeddings, labels, recall_ks)
    knn_acc = knn_accuracy(embeddings, labels)
    lp_train_idx, lp_test_idx = train_test_split(
        np.arange(len(labels)), test_size=0.2, random_state=42, stratify=labels
    )
    lp_acc = linear_probe(embeddings[lp_train_idx], labels[lp_train_idx],
                          embeddings[lp_test_idx], labels[lp_test_idx])
else:
    if os.path.exists(clean_cache_path):
        clean_data = Cache(clean_cache_path).load()
        clean_emb = clean_data["embeddings"]
        clean_labels = clean_data["labels"]
        n = min(len(clean_emb), len(embeddings))

        l2_shift_val = embedding_shift(clean_emb[:n], embeddings[:n])
        cos_shift_val = cosine_shift(clean_emb[:n], embeddings[:n])
        logger.info(f"EMBEDDING SHIFT (L2) = {l2_shift_val:.6f}")
        logger.info(f"EMBEDDING SHIFT (Cosine) = {cos_shift_val:.6f}")

        recall_scores = recall(embeddings[:n], labels[:n], recall_ks,
                               gallery_embeddings=clean_emb[:n], gallery_labels=clean_labels[:n],
                               exclude_self=True)

        knn_acc = knn_accuracy(clean_emb[:n], clean_labels[:n],
                               test_embeddings=embeddings[:n], test_labels=labels[:n])

        lp_acc = linear_probe(clean_emb[:n], clean_labels[:n],
                              embeddings[:n], labels[:n])
    else:
        logger.warning(f"Clean cache not found at {clean_cache_path} — skipping robustness metrics")
        logger.warning("Run clean evaluation first (--attack none) to generate the reference cache.")
        recall_scores = recall(embeddings, labels, recall_ks)
        knn_acc = knn_accuracy(embeddings, labels)
        lp_acc = 0.0

logger.info(f"RECALL@{recall_ks} = {recall_scores}")
logger.info(f"LINEAR PROBING ACC = {lp_acc:.4f}")
logger.info(f"KNN ACC = {knn_acc:.4f}")
logger.info(f"ALIGNMENT = {align_score:.6f}")
logger.info(f"UNIFORMITY = {uniform_score:.6f}")

result_row = {
    "timestamp": datetime.now().isoformat(),
    "model": args.model,
    "dataset": args.dataset,
    "pool": args.pool,
    "subset_size": args.subset_size,
    "attack": args.attack,
    "epsilon": args.epsilon if args.attack != "none" else None,
    "num_samples": int(len(labels)),
    "num_classes": int(len(np.unique(labels))),
}
for i, k in enumerate(recall_ks):
    result_row[f"recall@{k}"] = round(float(recall_scores[i]), 4)
result_row["linear_probe_acc"] = round(float(lp_acc), 4)
result_row["knn_acc"] = round(float(knn_acc), 4)
result_row["alignment"] = round(float(align_score), 6)
result_row["uniformity"] = round(float(uniform_score), 6)
if l2_shift_val is not None:
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

csv_path = "./results/tables/all_results.csv"
file_exists = os.path.exists(csv_path)
with open(csv_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=result_row.keys())
    if not file_exists:
        writer.writeheader()
    writer.writerow(result_row)

logger.info(f"Results saved to {json_path} and {csv_path}")
logger.info("=" * 60)
