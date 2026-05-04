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


def aggregate(rows: List[Dict[str, str]]) -> Tuple[List[int], List[float], List[float], List[float], List[float], int]:
    """Return (ns_sorted, mae_mean, mae_se, jacc_mean, jacc_se, total_seeds)."""
    by_n: Dict[int, Dict[str, List[float]]] = {}
    for r in rows:
        n = int(r["n"])
        by_n.setdefault(n, {"mae": [], "jacc": []})
        if r.get("abs_error", "") != "":
            by_n[n]["mae"].append(float(r["abs_error"]))
        if r.get("jaccard", "") != "":
            by_n[n]["jacc"].append(float(r["jaccard"]))

    ns = sorted(by_n.keys())
    mae_mean, mae_se = [], []
    jacc_mean, jacc_se = [], []
    total = 0

    for n in ns:
        mae_vals = by_n[n]["mae"]
        jacc_vals = by_n[n]["jacc"]
        total = max(total, len(mae_vals))

        mae_mean.append(float(np.mean(mae_vals)) if mae_vals else float("nan"))
        mae_se.append(float(np.std(mae_vals, ddof=1) / math.sqrt(len(mae_vals)))
                      if len(mae_vals) > 1 else 0.0)

        jacc_mean.append(float(np.mean(jacc_vals)) if jacc_vals else float("nan"))
        jacc_se.append(float(np.std(jacc_vals, ddof=1) / math.sqrt(len(jacc_vals)))
                       if len(jacc_vals) > 1 else 0.0)

    return ns, mae_mean, mae_se, jacc_mean, jacc_se, total


def plot_one(csv_path: Path, out_path: Path, title_suffix: str = "") -> None:
    rows = load_rows(csv_path)
    if not rows:
        print(f"  [{csv_path.name}] no successful rows; skipping.")
        return

    ns, mae_mean, mae_se, jacc_mean, jacc_se, n_seeds = aggregate(rows)

    p_true_set = sorted({int(r["p_true"]) for r in rows})
    p_label = ",".join(str(p) for p in p_true_set)
    title_p = f"p={p_label}{title_suffix}"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # MAE panel
    ax = axes[0]
    ax.errorbar(ns, mae_mean, yerr=mae_se, marker="o", capsize=3, color="#1f77b4")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sample size n")
    ax.set_ylabel(r"Mean $|p - \hat{p}|$")
    ax.set_title(f"Latent-dim recovery error  ({title_p})")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Jaccard panel
    ax = axes[1]
    ax.errorbar(ns, jacc_mean, yerr=jacc_se, marker="o", capsize=3, color="#2ca02c")
    ax.axhline(1.0, linestyle="--", color="grey", alpha=0.5, label="perfect recovery")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sample size n")
    ax.set_ylabel("Jaccard similarity")
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
