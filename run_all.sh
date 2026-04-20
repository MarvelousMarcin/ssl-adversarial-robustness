#!/bin/bash
# Full benchmark grid.
# Usage: bash run_all.sh

set -e

MODEL_POOLS=("dino:mean" "dino:cls" "ijepa:mean" "vit_sup:mean" "resnet50:mean")
DATASETS=("cifar100" "imagenet")
EPSILONS=(0.008 0.016 0.063)
PGD_STEPS=20
PGD_RESTARTS=2
APGD_STEPS=50
APGD_RESTARTS=1
SEED=0
RUN_AA=${RUN_AA:-0}
RUN_TRANSFER=1
# Epsilons used for transfer phase (narrower by default — transfer grid grows quickly).
TRANSFER_EPSILONS=(0.031)

echo "=============================="
echo "  SSL Robustness Experiments"
echo "  model-pools=${MODEL_POOLS[*]}"
echo "  pgd=${PGD_STEPS}x${PGD_RESTARTS}  apgd=${APGD_STEPS}x${APGD_RESTARTS}"
echo "  eps=${EPSILONS[*]}  seed=${SEED}"
echo "  RUN_AA=${RUN_AA}  RUN_TRANSFER=${RUN_TRANSFER}  transfer_eps=${TRANSFER_EPSILONS[*]}"
echo "=============================="

run_for_mp() {
    local model="$1"
    local pool="$2"
    shift 2
    python main.py --model "$model" --pool "$pool" --seed "$SEED" "$@"
}

# Phase 1: Clean evaluation (pool-matched)
echo ""
echo "--- Phase 1: Clean Evaluation ---"
for mp in "${MODEL_POOLS[@]}"; do
    model="${mp%%:*}"; pool="${mp##*:}"
    for dataset in "${DATASETS[@]}"; do
        echo "[CLEAN] model=$model  pool=$pool  dataset=$dataset"
        run_for_mp "$model" "$pool" --dataset "$dataset" --attack none
    done
done

# Phase 2: FGSM (centroid-based) — diagnostic baseline
echo ""
echo "--- Phase 2: FGSM (centroid, diagnostic) ---"
for mp in "${MODEL_POOLS[@]}"; do
    model="${mp%%:*}"; pool="${mp##*:}"
    for dataset in "${DATASETS[@]}"; do
        for eps in "${EPSILONS[@]}"; do
            echo "[FGSM] model=$model  pool=$pool  dataset=$dataset  eps=$eps"
            run_for_mp "$model" "$pool" --dataset "$dataset" \
                --attack fgsm --epsilon "$eps"
        done
    done
done

# Phase 3: PGD (centroid-based) — diagnostic
echo ""
echo "--- Phase 3: PGD (centroid, diagnostic) ---"
for mp in "${MODEL_POOLS[@]}"; do
    model="${mp%%:*}"; pool="${mp##*:}"
    for dataset in "${DATASETS[@]}"; do
        for eps in "${EPSILONS[@]}"; do
            echo "[PGD] model=$model  pool=$pool  dataset=$dataset  eps=$eps"
            run_for_mp "$model" "$pool" --dataset "$dataset" \
                --attack pgd --epsilon "$eps" \
                --pgd_steps "$PGD_STEPS" --pgd_restarts "$PGD_RESTARTS"
        done
    done
done

# Phase 4: FGSM label-free — fair single-step baseline
echo ""
echo "--- Phase 4: FGSM label-free ---"
for mp in "${MODEL_POOLS[@]}"; do
    model="${mp%%:*}"; pool="${mp##*:}"
    for dataset in "${DATASETS[@]}"; do
        for eps in "${EPSILONS[@]}"; do
            echo "[FGSM-LF] model=$model  pool=$pool  dataset=$dataset  eps=$eps"
            run_for_mp "$model" "$pool" --dataset "$dataset" \
                --attack fgsm_lf --epsilon "$eps"
        done
    done
done

# Phase 5: PGD label-free — previous headline attack
echo ""
echo "--- Phase 5: PGD label-free ---"
for mp in "${MODEL_POOLS[@]}"; do
    model="${mp%%:*}"; pool="${mp##*:}"
    for dataset in "${DATASETS[@]}"; do
        for eps in "${EPSILONS[@]}"; do
            echo "[PGD-LF] model=$model  pool=$pool  dataset=$dataset  eps=$eps"
            run_for_mp "$model" "$pool" --dataset "$dataset" \
                --attack pgd_lf --epsilon "$eps" \
                --pgd_steps "$PGD_STEPS" --pgd_restarts "$PGD_RESTARTS"
        done
    done
done

# Phase 6: APGD (centroid-based) — core AutoAttack component
echo ""
echo "--- Phase 6: APGD (centroid) ---"
for mp in "${MODEL_POOLS[@]}"; do
    model="${mp%%:*}"; pool="${mp##*:}"
    for dataset in "${DATASETS[@]}"; do
        for eps in "${EPSILONS[@]}"; do
            echo "[APGD] model=$model  pool=$pool  dataset=$dataset  eps=$eps"
            run_for_mp "$model" "$pool" --dataset "$dataset" \
                --attack apgd --epsilon "$eps" \
                --apgd_steps "$APGD_STEPS" --apgd_restarts "$APGD_RESTARTS"
        done
    done
done

# Phase 7: APGD label-free — NEW HEADLINE attack for the paper
echo ""
echo "--- Phase 7: APGD label-free (HEADLINE) ---"
for mp in "${MODEL_POOLS[@]}"; do
    model="${mp%%:*}"; pool="${mp##*:}"
    for dataset in "${DATASETS[@]}"; do
        for eps in "${EPSILONS[@]}"; do
            echo "[APGD-LF] model=$model  pool=$pool  dataset=$dataset  eps=$eps"
            run_for_mp "$model" "$pool" --dataset "$dataset" \
                --attack apgd_lf --epsilon "$eps" \
                --apgd_steps "$APGD_STEPS" --apgd_restarts "$APGD_RESTARTS"
        done
    done
done

if [ "$RUN_TRANSFER" = "1" ]; then
    echo ""
    echo "--- Phase 9: Transfer attack (APGD-LF) ---"
    for src in "${MODEL_POOLS[@]}"; do
        src_model="${src%%:*}"; src_pool="${src##*:}"
        for tgt in "${MODEL_POOLS[@]}"; do
            tgt_model="${tgt%%:*}"; tgt_pool="${tgt##*:}"
            if [ "$src_model" = "$tgt_model" ] && [ "$src_pool" = "$tgt_pool" ]; then
                continue
            fi
            for dataset in "${DATASETS[@]}"; do
                for eps in "${TRANSFER_EPSILONS[@]}"; do
                    echo "[TRANSFER] ${src} -> ${tgt}  dataset=$dataset  eps=$eps"
                    python transfer_attack.py \
                        --source_model "$src_model" --source_pool "$src_pool" \
                        --target_model "$tgt_model" --target_pool "$tgt_pool" \
                        --dataset "$dataset" --attack apgd_lf \
                        --epsilon "$eps" \
                        --apgd_steps "$APGD_STEPS" --apgd_restarts "$APGD_RESTARTS" \
                        --seed "$SEED"
                done
            done
        done
    done
fi

echo ""
echo "--- Phase 10: Aggregate results (LaTeX tables + figures) ---"
python aggregate_results.py

echo ""
echo "=============================="
echo "  All experiments complete!"
echo "  Raw results:   ./results/tables/"
echo "  Paper output:  ./results/paper/"
echo "=============================="
