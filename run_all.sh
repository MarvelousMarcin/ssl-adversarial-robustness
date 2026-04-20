#!/bin/bash
# Run all experiments with matched pooling (fix #2) and strong PGD (fix #3).
# Usage: bash run_all.sh
#
# NOTE: cache filenames don't encode pool/steps/restarts, so we wipe stale
# adversarial caches up front to avoid mixing weak-attack and strong-attack runs.

set -e

MODELS=("dino" "ijepa" "vit_sup" "resnet50")
DATASETS=("cifar100")
EPSILONS=(0.008 0.016 0.031 0.063)
POOL="mean"
PGD_STEPS=20
PGD_RESTARTS=2

echo "=============================="
echo "  SSL Robustness Experiments"
echo "  pool=$POOL  pgd=${PGD_STEPS}x${PGD_RESTARTS}  eps=${EPSILONS[*]}"
echo "=============================="

# Wipe stale caches (clean + adversarial) since cache name doesn't encode
# pool/steps/restarts. Comment out if you want to reuse clean embeddings.
# echo "Clearing stale embedding caches..."
# rm -f ./results/*.pkl

# Phase 1: Clean evaluation (pool-matched)
echo ""
echo "--- Phase 1: Clean Evaluation ---"
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        echo "[CLEAN] model=$model  dataset=$dataset"
        python main.py --model "$model" --dataset "$dataset" \
            --attack none --pool "$POOL"
    done
done

# Phase 2: FGSM (centroid-based) — secondary diagnostic
echo ""
echo "--- Phase 2: FGSM (centroid-based, diagnostic only) ---"
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for eps in "${EPSILONS[@]}"; do
            echo "[FGSM] model=$model  dataset=$dataset  eps=$eps"
            python main.py --model "$model" --dataset "$dataset" \
                --attack fgsm --epsilon "$eps" \
                --pool "$POOL"
        done
    done
done

# Phase 3: PGD (centroid-based) — secondary diagnostic
echo ""
echo "--- Phase 3: PGD (centroid-based, diagnostic only) ---"
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for eps in "${EPSILONS[@]}"; do
            echo "[PGD] model=$model  dataset=$dataset  eps=$eps"
            python main.py --model "$model" --dataset "$dataset" \
                --attack pgd --epsilon "$eps" \
                --pgd_steps "$PGD_STEPS" --pgd_restarts "$PGD_RESTARTS" \
                --pool "$POOL"
        done
    done
done

# Phase 4: FGSM label-free — fair cross-model single-step baseline
echo ""
echo "--- Phase 4: FGSM label-free ---"
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for eps in "${EPSILONS[@]}"; do
            echo "[FGSM-LF] model=$model  dataset=$dataset  eps=$eps"
            python main.py --model "$model" --dataset "$dataset" \
                --attack fgsm_lf --epsilon "$eps" \
                --pool "$POOL"
        done
    done
done

# Phase 5: PGD label-free — HEADLINE attack for the paper
echo ""
echo "--- Phase 5: PGD label-free (HEADLINE) ---"
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for eps in "${EPSILONS[@]}"; do
            echo "[PGD-LF] model=$model  dataset=$dataset  eps=$eps"
            python main.py --model "$model" --dataset "$dataset" \
                --attack pgd_lf --epsilon "$eps" \
                --pgd_steps "$PGD_STEPS" --pgd_restarts "$PGD_RESTARTS" \
                --pool "$POOL"
        done
    done
done

echo ""
echo "=============================="
echo "  All experiments complete!"
echo "  Results: ./results/tables/"
echo "=============================="
