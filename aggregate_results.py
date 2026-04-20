"""Aggregate all experimental results into publication artefacts.

Reads `results/tables/all_results.json` (populated by main.py + transfer_attack.py)
and produces:
    results/paper/
        tables/       — *.tex (booktabs) + *.md previews
        figures/      — *.pdf + *.png
        summary.md    — headline findings, sanity checks, generation log

Handles missing fields from older runs (e.g. entries with no `pool` or `seed`
column) by filling defaults. Multi-seed entries are aggregated as mean ± std;
single-seed entries show mean only. Duplicate runs (same config re-executed)
are deduped keeping the latest timestamp.
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_JSON = Path("./results/tables/all_results.json")
OUT_DIR = Path("./results/paper")
TABLES_DIR = OUT_DIR / "tables"
FIGS_DIR = OUT_DIR / "figures"

MODEL_LABEL = {
    "dino": "DINOv2",
    "ijepa": "I-JEPA",
    "vit_sup": "ViT-sup",
    "resnet50": "ResNet50",
}
MODEL_COLOR = {
    "dino": "#1f77b4",
    "ijepa": "#d62728",
    "vit_sup": "#2ca02c",
    "resnet50": "#ff7f0e",
}
MODEL_ORDER = ["dino", "ijepa", "vit_sup", "resnet50"]
DATASET_LABEL = {
    "imagenet": "Imagenette",
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "car": "CARS-196",
}
ATTACK_LABEL = {
    "none": "Clean",
    "fgsm": "FGSM",
    "pgd": "PGD",
    "apgd": "APGD",
    "fgsm_lf": "FGSM-LF",
    "pgd_lf": "PGD-LF",
    "apgd_lf": "APGD-LF",
    "aa": "AutoAttack",
}

HEADLINE_ATTACK = "apgd_lf"
FIXED_EPS = 0.031


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataframe():
    if not RESULTS_JSON.exists():
        raise SystemExit(f"Missing {RESULTS_JSON}. Run main.py first.")
    with open(RESULTS_JSON) as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)

    defaults = {
        "pool": "mean",
        "seed": 0,
        "transfer_source_model": pd.NA,
        "transfer_source_pool": pd.NA,
        "epsilon": 0.0,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    dedup_cols = [
        "model", "dataset", "attack", "epsilon", "pool", "seed",
        "transfer_source_model", "transfer_source_pool",
    ]
    df = df.sort_values("timestamp").drop_duplicates(subset=dedup_cols, keep="last")
    df["is_transfer"] = df["transfer_source_model"].notna()
    return df


def aggregate(df, metric, group_cols):
    """Group and compute mean, std across seeds."""
    g = df.groupby(group_cols, dropna=False)[metric].agg(["mean", "std", "count"]).reset_index()
    g["std"] = g["std"].fillna(0.0)
    return g


def fmt_cell(mean, std, bold=False, ndigits=3):
    if pd.isna(mean):
        s = "---"
    elif std > 1e-4:
        s = f"{mean:.{ndigits}f}$\\pm${std:.{ndigits}f}"
    else:
        s = f"{mean:.{ndigits}f}"
    return f"\\textbf{{{s}}}" if bold else s


def write_latex(path, caption, label, colspec, header_rows, body_rows):
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
    ]
    lines.extend(header_rows)
    lines.append("\\midrule")
    lines.extend(body_rows)
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def table_clean_baselines(df, out_dir):
    clean = df[(df["attack"] == "none") & (~df["is_transfer"])]
    if clean.empty:
        print("[skip] clean baselines: no rows")
        return
    metrics = ["recall@1", "knn_acc", "linear_probe_acc", "alignment", "uniformity"]
    metric_label = {
        "recall@1": "R@1", "knn_acc": "kNN", "linear_probe_acc": "LP",
        "alignment": "Align", "uniformity": "Unif",
    }

    datasets = sorted(clean["dataset"].unique())
    for dataset in datasets:
        sub = clean[clean["dataset"] == dataset]
        # Pool shown for DINO (mean, cls) otherwise single row per model
        rows = []
        for model in MODEL_ORDER:
            sub_m = sub[sub["model"] == model]
            if sub_m.empty:
                continue
            for pool in sorted(sub_m["pool"].unique()):
                sub_mp = sub_m[sub_m["pool"] == pool]
                agg = sub_mp[metrics].agg(["mean", "std"])
                tag = f"{MODEL_LABEL[model]}"
                if sub_m["pool"].nunique() > 1:
                    tag += f" ({pool})"
                cells = [fmt_cell(agg.loc["mean", m], agg.loc["std", m]) for m in metrics]
                rows.append(f"{tag} & " + " & ".join(cells) + " \\\\")

        header = "Model & " + " & ".join(metric_label[m] for m in metrics) + " \\\\"
        write_latex(
            out_dir / f"table_clean_{dataset}.tex",
            caption=f"Clean representation quality on {DATASET_LABEL.get(dataset, dataset)}.",
            label=f"tab:clean-{dataset}",
            colspec="l" + "c" * len(metrics),
            header_rows=[header],
            body_rows=rows,
        )
        print(f"[ok] table_clean_{dataset}.tex")


def table_headline(df, out_dir):
    """APGD-LF R@1 across epsilons — one table per dataset, model × eps."""
    sub = df[(df["attack"] == HEADLINE_ATTACK) & (~df["is_transfer"])]
    if sub.empty:
        print(f"[skip] headline ({HEADLINE_ATTACK}): no rows")
        return

    metric = "recall@1"
    for dataset in sorted(sub["dataset"].unique()):
        s = sub[sub["dataset"] == dataset]
        epsilons = sorted(s["epsilon"].unique())
        if not epsilons:
            continue
        rows = []
        # Find best per epsilon for bolding
        best = {}
        for eps in epsilons:
            s_eps = s[s["epsilon"] == eps]
            grouped = s_eps.groupby(["model", "pool"])[metric].mean()
            if len(grouped):
                best[eps] = grouped.max()

        for model in MODEL_ORDER:
            s_m = s[s["model"] == model]
            if s_m.empty:
                continue
            for pool in sorted(s_m["pool"].unique()):
                s_mp = s_m[s_m["pool"] == pool]
                tag = MODEL_LABEL[model]
                if s_m["pool"].nunique() > 1:
                    tag += f" ({pool})"
                cells = []
                for eps in epsilons:
                    s_eps = s_mp[s_mp["epsilon"] == eps]
                    if s_eps.empty:
                        cells.append("---")
                    else:
                        mean = s_eps[metric].mean()
                        std = s_eps[metric].std() if len(s_eps) > 1 else 0.0
                        is_best = (mean >= best.get(eps, -1) - 1e-6)
                        cells.append(fmt_cell(mean, std, bold=is_best))
                rows.append(f"{tag} & " + " & ".join(cells) + " \\\\")

        header = "Model & " + " & ".join(f"$\\epsilon$={e}" for e in epsilons) + " \\\\"
        attack_name = ATTACK_LABEL[HEADLINE_ATTACK]
        write_latex(
            out_dir / f"table_headline_{dataset}.tex",
            caption=(f"R@1 under {attack_name} attack on {DATASET_LABEL.get(dataset, dataset)}. "
                     f"Higher = more robust. Best per column in bold."),
            label=f"tab:headline-{dataset}",
            colspec="l" + "c" * len(epsilons),
            header_rows=[header],
            body_rows=rows,
        )
        print(f"[ok] table_headline_{dataset}.tex")


def table_attack_progression(df, out_dir):
    """Show FGSM vs PGD vs APGD at fixed eps to expose attack-strength gap."""
    sub = df[
        (df["attack"].isin(["fgsm_lf", "pgd_lf", "apgd_lf"]))
        & (df["epsilon"].round(4) == FIXED_EPS)
        & (~df["is_transfer"])
    ]
    if sub.empty:
        print("[skip] attack progression: no rows")
        return

    attacks = ["fgsm_lf", "pgd_lf", "apgd_lf"]
    metric = "recall@1"
    for dataset in sorted(sub["dataset"].unique()):
        s = sub[sub["dataset"] == dataset]
        rows = []
        for model in MODEL_ORDER:
            s_m = s[s["model"] == model]
            if s_m.empty:
                continue
            for pool in sorted(s_m["pool"].unique()):
                s_mp = s_m[s_m["pool"] == pool]
                tag = MODEL_LABEL[model]
                if s_m["pool"].nunique() > 1:
                    tag += f" ({pool})"
                cells = []
                for atk in attacks:
                    s_atk = s_mp[s_mp["attack"] == atk]
                    if s_atk.empty:
                        cells.append("---")
                    else:
                        mean = s_atk[metric].mean()
                        std = s_atk[metric].std() if len(s_atk) > 1 else 0.0
                        cells.append(fmt_cell(mean, std))
                rows.append(f"{tag} & " + " & ".join(cells) + " \\\\")

        header = "Model & " + " & ".join(ATTACK_LABEL[a] for a in attacks) + " \\\\"
        write_latex(
            out_dir / f"table_attack_progression_{dataset}.tex",
            caption=(f"Attack-strength progression at $\\epsilon={FIXED_EPS}$ on "
                     f"{DATASET_LABEL.get(dataset, dataset)}: R@1 under label-free FGSM / PGD / APGD. "
                     f"Large FGSM$\\to$APGD gap per model is a gradient-masking warning sign."),
            label=f"tab:attack-progression-{dataset}",
            colspec="l" + "c" * len(attacks),
            header_rows=[header],
            body_rows=rows,
        )
        print(f"[ok] table_attack_progression_{dataset}.tex")


def table_pool_ablation(df, out_dir):
    """DINO pool=mean vs pool=cls under APGD-LF."""
    sub = df[(df["model"] == "dino") & (df["attack"] == HEADLINE_ATTACK) & (~df["is_transfer"])]
    if sub.empty or sub["pool"].nunique() < 2:
        print("[skip] pool ablation: need both pool=mean and pool=cls")
        return
    metric = "recall@1"
    for dataset in sorted(sub["dataset"].unique()):
        s = sub[sub["dataset"] == dataset]
        epsilons = sorted(s["epsilon"].unique())
        rows = []
        for pool in sorted(s["pool"].unique()):
            s_p = s[s["pool"] == pool]
            cells = []
            for eps in epsilons:
                s_eps = s_p[s_p["epsilon"] == eps]
                if s_eps.empty:
                    cells.append("---")
                else:
                    mean = s_eps[metric].mean()
                    std = s_eps[metric].std() if len(s_eps) > 1 else 0.0
                    cells.append(fmt_cell(mean, std))
            rows.append(f"DINOv2 ({pool}) & " + " & ".join(cells) + " \\\\")
        header = "Config & " + " & ".join(f"$\\epsilon$={e}" for e in epsilons) + " \\\\"
        write_latex(
            out_dir / f"table_pool_ablation_{dataset}.tex",
            caption=(f"DINOv2 pooling ablation on {DATASET_LABEL.get(dataset, dataset)} under "
                     f"{ATTACK_LABEL[HEADLINE_ATTACK]} — R@1 by pool."),
            label=f"tab:pool-{dataset}",
            colspec="l" + "c" * len(epsilons),
            header_rows=[header],
            body_rows=rows,
        )
        print(f"[ok] table_pool_ablation_{dataset}.tex")


def table_transfer(df, out_dir):
    """Source × target matrix of R@1 under transfer APGD-LF."""
    sub = df[df["is_transfer"] & (df["attack"] == HEADLINE_ATTACK)]
    if sub.empty:
        print("[skip] transfer matrix: no transfer rows")
        return
    metric = "recall@1"
    for dataset in sorted(sub["dataset"].unique()):
        for eps in sorted(sub["epsilon"].unique()):
            s = sub[(sub["dataset"] == dataset) & (sub["epsilon"].round(4) == round(eps, 4))]
            if s.empty:
                continue
            # Also include self-attack (from main results) as diagonal reference.
            self_ref = df[
                (df["attack"] == HEADLINE_ATTACK) & (~df["is_transfer"])
                & (df["dataset"] == dataset) & (df["epsilon"].round(4) == round(eps, 4))
            ]

            def tag(model, pool):
                t = MODEL_LABEL[model]
                if df[df["model"] == model]["pool"].nunique() > 1:
                    t += f":{pool}"
                return t

            sources = sorted(
                set(zip(s["transfer_source_model"], s["transfer_source_pool"]))
                | set(zip(self_ref["model"], self_ref["pool"]))
            )
            targets = sorted(
                set(zip(s["model"], s["pool"])) | set(zip(self_ref["model"], self_ref["pool"]))
            )
            if not sources or not targets:
                continue

            rows = []
            for src in sources:
                cells = []
                for tgt in targets:
                    if src == tgt:
                        ref = self_ref[(self_ref["model"] == tgt[0]) & (self_ref["pool"] == tgt[1])]
                        if ref.empty:
                            cells.append("---")
                        else:
                            val = ref[metric].mean()
                            cells.append(f"\\underline{{{val:.3f}}}")  # diagonal reference
                    else:
                        row = s[
                            (s["transfer_source_model"] == src[0])
                            & (s["transfer_source_pool"] == src[1])
                            & (s["model"] == tgt[0])
                            & (s["pool"] == tgt[1])
                        ]
                        if row.empty:
                            cells.append("---")
                        else:
                            val = row[metric].mean()
                            cells.append(f"{val:.3f}")
                rows.append(f"{tag(*src)} & " + " & ".join(cells) + " \\\\")

            header = "Source $\\downarrow$ / Target $\\rightarrow$ & " + \
                     " & ".join(tag(*t) for t in targets) + " \\\\"
            write_latex(
                out_dir / f"table_transfer_{dataset}_eps{eps}.tex",
                caption=(f"Transfer attack matrix on {DATASET_LABEL.get(dataset, dataset)} at "
                         f"$\\epsilon={eps}$ under {ATTACK_LABEL[HEADLINE_ATTACK]}. "
                         f"Rows = source model, columns = target model. "
                         f"Underlined = white-box self-attack (diagonal). "
                         f"Low off-diagonal $\\Rightarrow$ small transferability $\\Rightarrow$ "
                         f"source-side robustness was not gradient masking."),
                label=f"tab:transfer-{dataset}-eps{eps}",
                colspec="l" + "c" * len(targets),
                header_rows=[header],
                body_rows=rows,
            )
            print(f"[ok] table_transfer_{dataset}_eps{eps}.tex")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _model_pool_key(row):
    m, p = row["model"], row["pool"]
    return f"{MODEL_LABEL[m]} ({p})" if m == "dino" else MODEL_LABEL[m]


def fig_robustness_curves(df, out_dir):
    sub = df[(df["attack"] == HEADLINE_ATTACK) & (~df["is_transfer"])]
    if sub.empty:
        print("[skip] robustness curves")
        return

    datasets = sorted(sub["dataset"].unique())
    metric = "recall@1"
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        s = sub[sub["dataset"] == dataset]
        for model in MODEL_ORDER:
            s_m = s[s["model"] == model]
            if s_m.empty:
                continue
            for pool in sorted(s_m["pool"].unique()):
                s_mp = s_m[s_m["pool"] == pool]
                agg = s_mp.groupby("epsilon")[metric].agg(["mean", "std", "count"]).reset_index()
                agg = agg.sort_values("epsilon")
                label = MODEL_LABEL[model] + (f" ({pool})" if s_m["pool"].nunique() > 1 else "")
                linestyle = "-" if pool == "mean" else "--"
                ax.plot(
                    agg["epsilon"], agg["mean"],
                    marker="o", color=MODEL_COLOR[model], linestyle=linestyle, label=label,
                )
                if (agg["std"] > 1e-4).any():
                    ax.fill_between(
                        agg["epsilon"], agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                        color=MODEL_COLOR[model], alpha=0.15,
                    )
        ax.set_xlabel(r"Perturbation budget $\epsilon$")
        ax.set_ylabel("R@1" if ax is axes[0] else "")
        ax.set_title(DATASET_LABEL.get(dataset, dataset))
        ax.grid(alpha=0.3)
    axes[-1].legend(loc="upper right", fontsize=9)
    fig.suptitle(f"Robustness under {ATTACK_LABEL[HEADLINE_ATTACK]} (higher = more robust)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_robustness_curves.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[ok] fig_robustness_curves.{pdf,png}")


def fig_embedding_shift(df, out_dir):
    sub = df[(df["attack"] == HEADLINE_ATTACK) & (~df["is_transfer"])]
    if sub.empty or "embedding_shift_cosine" not in sub.columns:
        print("[skip] embedding shift")
        return

    datasets = sorted(sub["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        s = sub[sub["dataset"] == dataset]
        for model in MODEL_ORDER:
            s_m = s[s["model"] == model]
            if s_m.empty:
                continue
            for pool in sorted(s_m["pool"].unique()):
                s_mp = s_m[s_m["pool"] == pool]
                agg = s_mp.groupby("epsilon")["embedding_shift_cosine"].mean().reset_index()
                agg = agg.sort_values("epsilon")
                label = MODEL_LABEL[model] + (f" ({pool})" if s_m["pool"].nunique() > 1 else "")
                linestyle = "-" if pool == "mean" else "--"
                ax.plot(
                    agg["epsilon"], agg["embedding_shift_cosine"],
                    marker="o", color=MODEL_COLOR[model], linestyle=linestyle, label=label,
                )
        ax.set_xlabel(r"$\epsilon$")
        ax.set_ylabel(r"$1-\cos(\mathrm{adv},\mathrm{clean})$" if ax is axes[0] else "")
        ax.set_title(DATASET_LABEL.get(dataset, dataset))
        ax.grid(alpha=0.3)
    axes[-1].legend(loc="lower right", fontsize=9)
    fig.suptitle("Embedding drift under APGD-LF")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_embedding_shift.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[ok] fig_embedding_shift.{pdf,png}")


def fig_attack_progression(df, out_dir):
    attacks = ["fgsm_lf", "pgd_lf", "apgd_lf"]
    sub = df[
        df["attack"].isin(attacks)
        & (df["epsilon"].round(4) == FIXED_EPS)
        & (~df["is_transfer"])
    ]
    if sub.empty:
        print("[skip] attack progression")
        return

    datasets = sorted(sub["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        s = sub[sub["dataset"] == dataset]
        model_pools = []
        for model in MODEL_ORDER:
            s_m = s[s["model"] == model]
            for pool in sorted(s_m["pool"].unique()):
                model_pools.append((model, pool))
        if not model_pools:
            continue
        x = np.arange(len(model_pools))
        width = 0.25
        for i, atk in enumerate(attacks):
            vals = []
            for model, pool in model_pools:
                row = s[(s["model"] == model) & (s["pool"] == pool) & (s["attack"] == atk)]
                vals.append(row["recall@1"].mean() if not row.empty else np.nan)
            ax.bar(x + (i - 1) * width, vals, width, label=ATTACK_LABEL[atk])
        ax.set_xticks(x)
        ax.set_xticklabels(
            [MODEL_LABEL[m] + (f"\n({p})" if m == "dino" else "") for m, p in model_pools],
            rotation=0, fontsize=8,
        )
        ax.set_ylabel("R@1" if ax is axes[0] else "")
        ax.set_title(DATASET_LABEL.get(dataset, dataset))
        ax.grid(axis="y", alpha=0.3)
    axes[-1].legend(loc="upper right", fontsize=9)
    fig.suptitle(f"Attack-strength progression at $\\epsilon={FIXED_EPS}$")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_attack_progression.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[ok] fig_attack_progression.{pdf,png}")


def fig_transfer_heatmap(df, out_dir):
    sub = df[df["is_transfer"] & (df["attack"] == HEADLINE_ATTACK)]
    if sub.empty:
        print("[skip] transfer heatmap")
        return

    self_ref = df[(df["attack"] == HEADLINE_ATTACK) & (~df["is_transfer"])]

    for dataset in sorted(sub["dataset"].unique()):
        for eps in sorted(sub["epsilon"].unique()):
            s = sub[(sub["dataset"] == dataset) & (sub["epsilon"].round(4) == round(eps, 4))]
            ref = self_ref[(self_ref["dataset"] == dataset) & (self_ref["epsilon"].round(4) == round(eps, 4))]
            if s.empty:
                continue
            sources = sorted(
                set(zip(s["transfer_source_model"], s["transfer_source_pool"]))
                | set(zip(ref["model"], ref["pool"]))
            )
            targets = sorted(
                set(zip(s["model"], s["pool"]))
                | set(zip(ref["model"], ref["pool"]))
            )
            if not sources or not targets:
                continue
            mat = np.full((len(sources), len(targets)), np.nan)
            for i, src in enumerate(sources):
                for j, tgt in enumerate(targets):
                    if src == tgt:
                        r = ref[(ref["model"] == tgt[0]) & (ref["pool"] == tgt[1])]
                    else:
                        r = s[
                            (s["transfer_source_model"] == src[0])
                            & (s["transfer_source_pool"] == src[1])
                            & (s["model"] == tgt[0])
                            & (s["pool"] == tgt[1])
                        ]
                    if not r.empty:
                        mat[i, j] = r["recall@1"].mean()

            fig, ax = plt.subplots(figsize=(1.2 * len(targets) + 2, 1.0 * len(sources) + 1.5))
            im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(np.arange(len(targets)))
            ax.set_yticks(np.arange(len(sources)))
            ax.set_xticklabels(
                [MODEL_LABEL[m] + (f":{p}" if m == "dino" else "") for m, p in targets],
                rotation=25, ha="right",
            )
            ax.set_yticklabels(
                [MODEL_LABEL[m] + (f":{p}" if m == "dino" else "") for m, p in sources]
            )
            ax.set_xlabel("Target (evaluates adv)")
            ax.set_ylabel("Source (generates adv)")
            for i in range(len(sources)):
                for j in range(len(targets)):
                    if not np.isnan(mat[i, j]):
                        txt_color = "black" if 0.3 < mat[i, j] < 0.8 else "white"
                        ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                                color=txt_color, fontsize=9)
            fig.colorbar(im, ax=ax, label="R@1")
            ax.set_title(f"Transfer matrix — {DATASET_LABEL.get(dataset, dataset)}, "
                         f"$\\epsilon={eps}$, {ATTACK_LABEL[HEADLINE_ATTACK]}\n"
                         f"(diagonal = white-box self-attack)")
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(out_dir / f"fig_transfer_{dataset}_eps{eps}.{ext}",
                            dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[ok] fig_transfer_{dataset}_eps{eps}.{{pdf,png}}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def build_summary(df, out_dir):
    clean = df[(df["attack"] == "none") & (~df["is_transfer"])]
    headline = df[(df["attack"] == HEADLINE_ATTACK) & (~df["is_transfer"])]
    fgsm_lf = df[(df["attack"] == "fgsm_lf") & (~df["is_transfer"])]

    counts = df.groupby(["attack", "is_transfer"]).size().reset_index(name="rows")

    lines = [
        "# Aggregated results",
        "",
        f"Source: `{RESULTS_JSON}` — {len(df)} deduplicated rows.",
        "",
        "## Row counts by attack",
        "",
        "| Attack | Transfer | Rows |",
        "|---|---|---|",
    ]
    for _, r in counts.iterrows():
        lines.append(f"| {ATTACK_LABEL.get(r['attack'], r['attack'])} | "
                     f"{'yes' if r['is_transfer'] else 'no'} | {r['rows']} |")

    seed_cols = df[~df["is_transfer"]].groupby(
        ["model", "dataset", "attack", "epsilon", "pool"]
    )["seed"].nunique()
    n_multi_seed = int((seed_cols > 1).sum())
    lines += [
        "",
        f"Multi-seed configs: **{n_multi_seed}** of {len(seed_cols)}. "
        f"Run with `--seed 1`, `--seed 2` etc. to add more; std bars appear in plots once ≥2 seeds exist.",
        "",
    ]

    # Headline findings — degradation at fixed eps
    if not headline.empty and not clean.empty:
        lines += ["## Headline: R@1 drop under APGD-LF ($\\epsilon={}$)".format(FIXED_EPS), ""]
        lines += ["| Model | Dataset | Clean R@1 | APGD-LF R@1 | Drop |",
                  "|---|---|---|---|---|"]
        for dataset in sorted(headline["dataset"].unique()):
            clean_d = clean[clean["dataset"] == dataset]
            head_d = headline[(headline["dataset"] == dataset) & (headline["epsilon"].round(4) == FIXED_EPS)]
            for model in MODEL_ORDER:
                for pool in sorted(set(clean_d[clean_d["model"] == model]["pool"])):
                    c = clean_d[(clean_d["model"] == model) & (clean_d["pool"] == pool)]["recall@1"]
                    h = head_d[(head_d["model"] == model) & (head_d["pool"] == pool)]["recall@1"]
                    if c.empty or h.empty:
                        continue
                    c_val, h_val = c.mean(), h.mean()
                    tag = MODEL_LABEL[model] + (f" ({pool})" if model == "dino" else "")
                    lines.append(f"| {tag} | {DATASET_LABEL.get(dataset, dataset)} | "
                                 f"{c_val:.3f} | {h_val:.3f} | {c_val - h_val:+.3f} |")
        lines.append("")

    # Sanity: FGSM-LF vs APGD-LF gap (warning if > 0.5)
    warnings = []
    for model in MODEL_ORDER:
        for dataset in sorted(df["dataset"].unique()):
            for pool in sorted(df[df["model"] == model]["pool"].unique()):
                fgsm = fgsm_lf[
                    (fgsm_lf["model"] == model) & (fgsm_lf["dataset"] == dataset)
                    & (fgsm_lf["pool"] == pool) & (fgsm_lf["epsilon"].round(4) == FIXED_EPS)
                ]
                apgd = headline[
                    (headline["model"] == model) & (headline["dataset"] == dataset)
                    & (headline["pool"] == pool) & (headline["epsilon"].round(4) == FIXED_EPS)
                ]
                if fgsm.empty or apgd.empty:
                    continue
                gap = fgsm["recall@1"].mean() - apgd["recall@1"].mean()
                if gap > 0.5:
                    tag = MODEL_LABEL[model] + (f" ({pool})" if model == "dino" else "")
                    warnings.append(
                        f"- **{tag} on {DATASET_LABEL.get(dataset, dataset)}**: "
                        f"FGSM-LF R@1={fgsm['recall@1'].mean():.3f} vs "
                        f"APGD-LF R@1={apgd['recall@1'].mean():.3f} "
                        f"(gap={gap:.3f}). Large gap is a gradient-masking warning."
                    )
    if warnings:
        lines += ["## Sanity flags (FGSM → APGD gap > 0.5)", "", *warnings, ""]

    # Artefact listing
    lines += ["## Generated artefacts", ""]
    for p in sorted((out_dir / "tables").glob("*.tex")):
        lines.append(f"- `{p.relative_to(out_dir)}`")
    for p in sorted((out_dir / "figures").glob("*.pdf")):
        lines.append(f"- `{p.relative_to(out_dir)}`")

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("[ok] summary.md")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataframe()
    print(f"Loaded {len(df)} rows after dedup.")

    table_clean_baselines(df, TABLES_DIR)
    table_headline(df, TABLES_DIR)
    table_attack_progression(df, TABLES_DIR)
    table_pool_ablation(df, TABLES_DIR)
    table_transfer(df, TABLES_DIR)

    fig_robustness_curves(df, FIGS_DIR)
    fig_embedding_shift(df, FIGS_DIR)
    fig_attack_progression(df, FIGS_DIR)
    fig_transfer_heatmap(df, FIGS_DIR)

    build_summary(df, OUT_DIR)
    print(f"\nAll artefacts in: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
