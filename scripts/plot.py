#!/usr/bin/env python3
"""Two-panel MAE + Jaccard figure from a sweep CSV.

Mirrors slide 26 of Nikolija's deck: x-axis is sample size (log2), y-axes are
mean ± SE of |p_hat - p| and Jaccard, both vs n.

Usage:
    python scripts/plot.py results/p5/p_recovery_sweep.csv
    python scripts/plot.py results/p5/p_recovery_sweep.csv --out figs/p5.png
    python scripts/plot.py results/p{5,20,40}/p_recovery_sweep.csv  # multiple files
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def load_rows(csv_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "ok":
                rows.append(row)
    return rows


METRIC_KEYS = [
    ("abs_error", "abs_err"),
    ("p_count_error", "p_cnt_err"),
    ("jaccard", "jacc_idx"),                      # legacy index-level
    ("jaccard_true_p", "jacc_idx_true_p"),
    ("jaccard_aligned", "jacc_aln"),
    ("jaccard_aligned_true_p", "jacc_aln_true_p"),
    ("gap_at_true_p", "gap_true_p"),
]


def aggregate(rows: List[Dict[str, str]]) -> Tuple[List[int], Dict[str, List[float]], Dict[str, List[float]], int]:
    """Return (ns_sorted, mean_dict_keyed_by_short_name, se_dict, total_seeds)."""
    by_n: Dict[int, Dict[str, List[float]]] = {}
    for r in rows:
        n = int(r["n"])
        if n not in by_n:
            by_n[n] = {short: [] for _, short in METRIC_KEYS}
        for col, short in METRIC_KEYS:
            v = r.get(col, "")
            if v != "":
                try:
                    by_n[n][short].append(float(v))
                except ValueError:
                    pass

    ns = sorted(by_n.keys())
    means: Dict[str, List[float]] = {short: [] for _, short in METRIC_KEYS}
    ses: Dict[str, List[float]] = {short: [] for _, short in METRIC_KEYS}

    total = 0
    for n in ns:
        for col, short in METRIC_KEYS:
            xs = by_n[n][short]
            total = max(total, len(xs))
            means[short].append(float(np.mean(xs)) if xs else float("nan"))
            ses[short].append(float(np.std(xs, ddof=1) / math.sqrt(len(xs))) if len(xs) > 1 else 0.0)

    return ns, means, ses, total


def plot_one(csv_path: Path, out_path: Path, title_suffix: str = "") -> None:
    rows = load_rows(csv_path)
    if not rows:
        print(f"  [{csv_path.name}] no successful rows; skipping.")
        return

    ns, means, ses, n_seeds = aggregate(rows)

    p_true_set = sorted({int(r["p_true"]) for r in rows})
    p_label = ",".join(str(p) for p in p_true_set)
    title_p = f"p={p_label}{title_suffix}"

    # Two-panel headline figure: |p - p_hat| (decreasing) and aligned Jaccard
    # at the true p (the meaningful Theorem 1 metric — permutation-invariant
    # via greedy correlation alignment of recovered Ẑ to ground-truth Z).
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.errorbar(ns, means["abs_err"], yerr=ses["abs_err"], marker="o", capsize=3, color="#1f77b4")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sample size n")
    ax.set_ylabel(r"Mean $|p - \hat{p}|$")
    ax.set_title(f"Latent-dim recovery error  ({title_p})")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    ax = axes[1]
    ax.errorbar(ns, means["jacc_aln_true_p"], yerr=ses["jacc_aln_true_p"],
                marker="o", capsize=3, color="#2ca02c")
    ax.axhline(1.0, linestyle="--", color="grey", alpha=0.5, label="perfect recovery")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sample size n")
    ax.set_ylabel("Aligned Jaccard at true p")
    ax.set_title(f"Shared-coord recovery (Theorem 1)  ({title_p})")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    fig.suptitle(f"{csv_path.name}  —  {n_seeds} seeds per n", fontsize=10, alpha=0.7)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  [{csv_path.name}] saved {out_path}  (ns={ns}, seeds={n_seeds})")

    # Also save a "diagnostics" 6-panel grid so we can compare metrics.
    diag_path = out_path.with_name(out_path.stem + "_diagnostics.png")
    plot_diagnostics(ns, means, ses, n_seeds, csv_path.name, title_p, diag_path)
    print(f"  [{csv_path.name}] saved {diag_path}")


def plot_diagnostics(ns, means, ses, n_seeds, csv_name, title_p, out_path):
    """Six-panel grid: every metric we track, with consistent x-axis."""
    plot_specs = [
        ("abs_err",          "Mean $|p - \\hat{p}|$",                "tab:blue",   None),
        ("p_cnt_err",        "Count error $|\\hat{p}_{elbow} - p|$", "tab:orange", None),
        ("jacc_aln_true_p",  "Aligned Jaccard at true p",            "tab:green",  (-0.05, 1.05)),
        ("jacc_aln",         "Aligned Jaccard at $\\hat{p}_{elbow}$","tab:olive",  (-0.05, 1.05)),
        ("jacc_idx_true_p",  "Index Jaccard at true p (no alignment)", "tab:gray",   (-0.05, 1.05)),
        ("gap_true_p",       "Spectral log-gap at true p",           "tab:purple", None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()
    for ax, (key, ylabel, color, ylim) in zip(axes, plot_specs):
        ax.errorbar(ns, means[key], yerr=ses[key], marker="o", capsize=3, color=color)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("n")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if ylim:
            ax.set_ylim(*ylim)
            if ylim == (-0.05, 1.05):
                ax.axhline(1.0, linestyle="--", color="grey", alpha=0.4)
    fig.suptitle(f"{title_p} — {csv_name} — {n_seeds} seeds per n", fontsize=10, alpha=0.7)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("csv", nargs="+", help="One or more sweep CSV paths.")
    parser.add_argument("--out", type=str, default=None,
                        help="Output PNG path. Required iff exactly one CSV is given. "
                             "When multiple CSVs are given, the output names are derived "
                             "from each CSV's parent directory under results/figs/.")
    parser.add_argument("--out-dir", type=str, default="results/figs",
                        help="Output directory for derived names (default: results/figs).")
    args = parser.parse_args()

    csvs = [Path(p) for p in args.csv]
    for c in csvs:
        if not c.exists():
            print(f"ERROR: {c} does not exist", file=sys.stderr)
            return 2

    out_dir = Path(args.out_dir)

    for c in csvs:
        if args.out and len(csvs) == 1:
            out = Path(args.out)
        else:
            # Derive: results/p5/p_recovery_sweep.csv → results/figs/p5.png
            tag = c.parent.name
            out = out_dir / f"{tag}.png"
        plot_one(c, out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
