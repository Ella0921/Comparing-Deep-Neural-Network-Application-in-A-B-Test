"""
experiments/run_all.py
======================
Reproduces all figures from the paper AND adds the missing head-to-head
comparison between MLP and partial factorial design.

Run from the project root:
    python experiments/run_all.py

Outputs are saved to results/figures/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from mlp.model import MLP
from mlp.evaluate import (
    make_ground_truth, all_factor_combinations,
    compute_pcs, compute_pcs_iterative,
)
from decentralized.factorial import compute_pcs_factorial

# ── output directory ──────────────────────────────────────────────────────────
RESULTS_DIR = Path("results/figures")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── shared experimental settings ─────────────────────────────────────────────
N_FACTORS   = 5          # keep small so 2^5=32 combinations are tractable
NOISE_STD   = 0.5
BETA        = make_ground_truth(N_FACTORS, seed=0)
COMBINATIONS = all_factor_combinations(N_FACTORS)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — PCS vs Sample Size for different MLP configurations
#            Reproduces Figure 2 in the paper
# ─────────────────────────────────────────────────────────────────────────────
def fig_pcs_vs_samplesize():
    print("\n[Fig 1] PCS vs Sample Size — MLP configurations")
    sample_sizes = [50, 100, 200, 400, 600, 800, 1000, 1500, 2000]

    configs = [
        {"label": "L=2, H=200 (rep=100)", "depth": 2, "width": 200, "n_reps": 100},
        {"label": "L=2, H=100 (rep=1000)", "depth": 2, "width": 100, "n_reps": 200},
        {"label": "L=3, H=100 (rep=1000)", "depth": 3, "width": 100, "n_reps": 200},
    ]
    markers = ["o", "s", "^"]
    colors  = ["#2196F3", "#FF9800", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for cfg, marker, color in zip(configs, markers, colors):
        pcs_vals = []
        for ss in sample_sizes:
            pcs = compute_pcs(
                MLP,
                {"input_dim": N_FACTORS, "depth": cfg["depth"],
                 "width": cfg["width"], "learning_rate": 0.01},
                N_FACTORS, BETA, ss,
                n_reps=cfg["n_reps"],
                noise_std=NOISE_STD,
                fit_kwargs={"epochs": 300},
            )
            pcs_vals.append(pcs)
            print(f"  {cfg['label']}  n={ss:5d}  PCS={pcs:.3f}")
        ax.plot(sample_sizes, pcs_vals, marker=marker, color=color,
                label=cfg["label"], linewidth=2, markersize=6)

    ax.set_xlabel("Sample Size", fontsize=12)
    ax.set_ylabel("PCS", fontsize=12)
    ax.set_title("PCS vs Sample Size for different MLP configurations", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    path = RESULTS_DIR / "fig1_pcs_vs_samplesize.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Iterative data addition (bug vs fix)
#            Reproduces Figure 3 and explains the plateau
# ─────────────────────────────────────────────────────────────────────────────
def fig_iterative():
    print("\n[Fig 2] Iterative data addition — bug vs fix")
    MAX_SAMPLES = 2000
    BATCH_SIZE  = 100
    N_REPS      = 200
    mlp_kwargs  = {"input_dim": N_FACTORS, "depth": 2, "width": 100,
                   "learning_rate": 0.01}
    fit_kwargs  = {"epochs": 300}

    sizes_bug, pcs_bug = compute_pcs_iterative(
        MLP, mlp_kwargs, N_FACTORS, BETA,
        MAX_SAMPLES, BATCH_SIZE, N_REPS, NOISE_STD,
        fit_kwargs=fit_kwargs, retrain_from_scratch=False)

    sizes_fix, pcs_fix = compute_pcs_iterative(
        MLP, mlp_kwargs, N_FACTORS, BETA,
        MAX_SAMPLES, BATCH_SIZE, N_REPS, NOISE_STD,
        fit_kwargs=fit_kwargs, retrain_from_scratch=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sizes_bug, pcs_bug, "o-", color="#E53935", label="Bug: train only on last batch",
            linewidth=2, markersize=5)
    ax.plot(sizes_fix, pcs_fix, "s-", color="#43A047", label="Fix: retrain on full dataset",
            linewidth=2, markersize=5)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.6, label="PCS = 0.5 (random)")
    ax.set_xlabel("Cumulative Sample Size", fontsize=12)
    ax.set_ylabel("PCS", fontsize=12)
    ax.set_title("Iterative Data Addition — Bug Diagnosis & Fix", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    path = RESULTS_DIR / "fig2_iterative_fix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — HEAD-TO-HEAD COMPARISON (missing from the paper!)
#            MLP vs Partial / Half / Full factorial at equal budgets
# ─────────────────────────────────────────────────────────────────────────────
def fig_comparison():
    print("\n[Fig 3] Head-to-head: MLP vs Factorial designs")

    # --- Factorial baselines (budget = n_runs determined by OA) ---
    results = {}
    for frac in ["partial", "half", "full"]:
        pcs, n_runs = compute_pcs_factorial(
            frac, N_FACTORS, BETA, n_reps=500, noise_std=NOISE_STD)
        results[frac] = (n_runs, pcs)
        print(f"  Factorial [{frac:7s}]  n_runs={n_runs:4d}  PCS={pcs:.3f}")

    # --- MLP evaluated at the same budgets ---
    budgets = sorted({v[0] for v in results.values()})
    # add a few extra points for a smooth MLP curve
    extra = [8, 16, 32, 64, 128, 256]
    all_budgets = sorted(set(budgets + extra))

    mlp_pcs = {}
    for b in all_budgets:
        pcs = compute_pcs(
            MLP,
            {"input_dim": N_FACTORS, "depth": 2, "width": 100,
             "learning_rate": 0.01},
            N_FACTORS, BETA, b,
            n_reps=300, noise_std=NOISE_STD,
            fit_kwargs={"epochs": 300},
        )
        mlp_pcs[b] = pcs
        print(f"  MLP                  n_samples={b:4d}  PCS={pcs:.3f}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 5))

    # MLP curve
    xs = list(mlp_pcs.keys())
    ys = list(mlp_pcs.values())
    ax.plot(xs, ys, "o-", color="#1565C0", linewidth=2.5,
            markersize=5, label="MLP (L=2, H=100)", zorder=3)

    # Factorial markers
    frac_styles = {
        "partial": ("^", "#E53935", "Partial Factorial"),
        "half":    ("s", "#FF8F00", "Half-Fraction Factorial"),
        "full":    ("D", "#2E7D32", "Full Factorial"),
    }
    for frac, (marker, color, label) in frac_styles.items():
        n_runs, pcs = results[frac]
        ax.scatter([n_runs], [pcs], marker=marker, color=color,
                   s=150, zorder=5, label=f"{label} (n={n_runs})")
        ax.axvline(n_runs, color=color, linestyle=":", alpha=0.4)

    ax.set_xlabel("Experimental Budget (# observations / runs)", fontsize=12)
    ax.set_ylabel("PCS", fontsize=12)
    ax.set_title("MLP vs Factorial Designs — Probability of Correct Selection",
                 fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    path = RESULTS_DIR / "fig3_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Summary dashboard (all results at a glance)
# ─────────────────────────────────────────────────────────────────────────────
def fig_dashboard():
    """Combine the three figures into one poster-style dashboard."""
    print("\n[Fig 4] Summary dashboard")

    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # Load saved figures as images
    import matplotlib.image as mpimg
    panels = [
        ("fig1_pcs_vs_samplesize.png", "MLP: PCS vs Sample Size"),
        ("fig2_iterative_fix.png",     "Iterative Bug & Fix"),
        ("fig3_comparison.png",        "MLP vs Factorial Designs"),
    ]
    for i, (fname, title) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        img_path = RESULTS_DIR / fname
        if img_path.exists():
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(title, fontsize=11, fontweight="bold", pad=6)

    fig.suptitle(
        f"A/B Test: MLP vs Decentralised Factorial Designs  "
        f"({N_FACTORS} factors, noise σ={NOISE_STD})",
        fontsize=13, fontweight="bold", y=1.02,
    )
    path = RESULTS_DIR / "fig4_dashboard.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print(" A/B Test Experiment Runner")
    print(f" n_factors={N_FACTORS}, noise_std={NOISE_STD}")
    print("=" * 60)

    fig_pcs_vs_samplesize()
    fig_iterative()
    fig_comparison()
    fig_dashboard()

    print("\n✓ All figures saved to results/figures/")
