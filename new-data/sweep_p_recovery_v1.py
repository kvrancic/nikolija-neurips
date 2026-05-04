#!/usr/bin/env python3
"""
Sweep experiment for p-recovery.

This script runs only the full pipeline once per (sample size n, seed),
saves a CSV, and optionally saves a plot of mean |p_hat - p| versus n.

It expects sanity_checks_param_v6.py to be in the same directory.
"""

import argparse
import csv
import os
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from typing import List

import numpy as np

from sanity_checks_param_v6 import (
    parse_gpu_ids,
    num_poly_features,
    run_full_pipeline_once,
)


def parse_powers(s: str) -> List[int]:
    """
    Accepts:
      "5-19"
      "5,7,9,11"
      "5,7,10-13"
    Returns sorted unique integer powers.
    """
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            step = 1 if b >= a else -1
            out.extend(list(range(a, b + step, step)))
        else:
            out.append(int(part))
    return sorted(set(out))


def auto_d_obs(p_true: int, q1: int, q2: int, degree: int, buffer: int) -> int:
    d1 = p_true + q1
    d2 = p_true + q2
    dj = p_true + q1 + q2

    # Per-view observed dimension should be at least enough for view feature rank.
    view_need = max(num_poly_features(d1, degree), num_poly_features(d2, degree))

    # Joint observed dimension is 2*d_obs, so per-view d_obs should satisfy:
    # 2*d_obs >= joint feature count.
    joint_need_per_view = int(np.ceil(num_poly_features(dj, degree) / 2))

    return max(view_need, joint_need_per_view) + buffer


def build_v6_args(args, n: int) -> SimpleNamespace:
    d_obs = args.d_obs
    if d_obs is None:
        d_obs = auto_d_obs(args.p_true, args.q1, args.q2, args.degree, args.d_obs_buffer)

    gpu_ids = parse_gpu_ids(args.gpus)

    return SimpleNamespace(
        # dimensions / data
        p_true=args.p_true,
        q1=args.q1,
        q2=args.q2,
        d_obs=d_obs,
        degree=args.degree,
        n=n,

        # candidate dimension scan
        max_dim_view=args.max_dim_view,
        max_dim_joint=args.max_dim_joint,
        extra_dim_buffer=args.extra_dim_buffer,

        # training
        gpu_ids=gpu_ids,
        n_restarts=args.n_restarts,
        n_iter=args.n_iter,
        n_samples=args.n_samples,
        n_samples_joint=args.n_samples_joint,
        lr=args.lr,
        max_val=args.max_val,

        # selection
        threshold=args.threshold,
        selection_abs_tol=args.selection_abs_tol,
        selection_rel_tol=args.selection_rel_tol,
        view_floor_slack=args.view_floor_slack,
        joint_floor_slack=args.joint_floor_slack,
        next_improvement_abs=args.next_improvement_abs,
        next_improvement_rel=args.next_improvement_rel,
    )


def maybe_make_plot(csv_path: Path, out_dir: Path, plot_name: str) -> None:
    import matplotlib.pyplot as plt

    rows = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "ok":
                rows.append(row)

    if not rows:
        print("No successful rows to plot.")
        return

    ns = sorted(set(int(r["n"]) for r in rows))
    xs = []
    means = []
    ses = []
    accs = []

    for n in ns:
        vals = [float(r["abs_error"]) for r in rows if int(r["n"]) == n]
        succ = [int(r["success"]) for r in rows if int(r["n"]) == n]
        if not vals:
            continue
        xs.append(n)
        means.append(float(np.mean(vals)))
        if len(vals) > 1:
            ses.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals))))
        else:
            ses.append(0.0)
        accs.append(float(np.mean(succ)))

    fig_path = out_dir / plot_name

    plt.figure(figsize=(7, 4.5))
    plt.errorbar(xs, means, yerr=ses, marker="o", capsize=3)
    plt.xscale("log", base=2)
    plt.xlabel("Sample size n")
    plt.ylabel("Mean absolute error |p_hat - p|")
    plt.title("Shared dimension recovery")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    print(f"Saved plot: {fig_path}")

    acc_path = out_dir / plot_name.replace(".png", "_accuracy.png")
    plt.figure(figsize=(7, 4.5))
    plt.plot(xs, accs, marker="o")
    plt.xscale("log", base=2)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Sample size n")
    plt.ylabel("Exact recovery accuracy")
    plt.title("Exact p recovery")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(acc_path, dpi=200)
    print(f"Saved plot: {acc_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    # Sweep setup
    parser.add_argument("--powers", type=str, default="5-19",
                        help="Powers of two for n. Examples: '5-19', '5,7,9,11', '5,7,10-13'.")
    parser.add_argument("--seeds-per-n", type=int, default=20)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="sweep_results")
    parser.add_argument("--csv-name", type=str, default="p_recovery_sweep.csv")
    parser.add_argument("--make-plot", action="store_true")

    # True dimensions
    parser.add_argument("--p-true", type=int, default=2)
    parser.add_argument("--q1", type=int, default=2)
    parser.add_argument("--q2", type=int, default=2)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--d-obs", type=int, default=None,
                        help="Observed dimension per view. If omitted, chosen automatically.")
    parser.add_argument("--d-obs-buffer", type=int, default=10)

    # Candidate scan
    parser.add_argument("--max-dim-view", type=int, default=None)
    parser.add_argument("--max-dim-joint", type=int, default=None)
    parser.add_argument("--extra-dim-buffer", type=int, default=3)

    # Compute/training
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--n-restarts", type=int, default=4)
    parser.add_argument("--n-iter", type=int, default=800)
    parser.add_argument("--n-samples", type=int, default=768)
    parser.add_argument("--n-samples-joint", type=int, default=1024)
    parser.add_argument("--max-val", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-2)

    # Dimension selection
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--selection-abs-tol", type=float, default=2e-3)
    parser.add_argument("--selection-rel-tol", type=float, default=0.05)
    parser.add_argument("--view-floor-slack", type=float, default=5e-3)
    parser.add_argument("--joint-floor-slack", type=float, default=5e-4)
    parser.add_argument("--next-improvement-abs", type=float, default=2e-3)
    parser.add_argument("--next-improvement-rel", type=float, default=0.10)

    args = parser.parse_args()

    powers = parse_powers(args.powers)
    ns = [2 ** k for k in powers]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / args.csv_name
    detail_dir = out_dir / "logs"
    detail_dir.mkdir(parents=True, exist_ok=True)

    d1_true = args.p_true + args.q1
    d2_true = args.p_true + args.q2
    dj_true = args.p_true + args.q1 + args.q2

    d_obs = args.d_obs
    if d_obs is None:
        d_obs = auto_d_obs(args.p_true, args.q1, args.q2, args.degree, args.d_obs_buffer)

    print("=" * 80)
    print("P-RECOVERY SWEEP")
    print("=" * 80)
    print(f"p_true={args.p_true}, q1={args.q1}, q2={args.q2}")
    print(f"targets: d1={d1_true}, d2={d2_true}, dj={dj_true}, p={args.p_true}")
    print(f"degree={args.degree}, d_obs={d_obs}, powers={powers}")
    print(f"seeds_per_n={args.seeds_per_n}, gpus={args.gpus}, restarts={args.n_restarts}")
    print(f"CSV: {csv_path}")
    print("=" * 80)

    fieldnames = [
        "status", "n", "power", "seed_index", "seed",
        "p_true", "q1", "q2", "d1_true", "d2_true", "dj_true",
        "d_obs", "degree",
        "d1_hat", "d2_hat", "dj_hat", "p_hat",
        "abs_error", "success", "runtime_sec", "detail_log", "error",
    ]

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            f_csv.flush()

        total = len(ns) * args.seeds_per_n
        counter = 0

        for power, n in zip(powers, ns):
            for seed_index in range(args.seeds_per_n):
                counter += 1
                seed = args.seed_offset + 100000 * power + seed_index
                label = f"n{n}_seed{seed_index}"
                detail_log = detail_dir / f"{label}.log"

                print(f"\n[{counter}/{total}] n=2^{power}={n}, seed_index={seed_index}, seed={seed}")
                t0 = time.time()

                v6_args = build_v6_args(args, n=n)
                # force auto d_obs value into v6 args for reproducibility
                v6_args.d_obs = d_obs

                try:
                    with detail_log.open("w") as f_detail:
                        with redirect_stdout(f_detail), redirect_stderr(f_detail):
                            passed, d1_hat, d2_hat, dj_hat, p_hat = run_full_pipeline_once(
                                v6_args,
                                seed=seed,
                                label=label,
                            )

                    runtime = time.time() - t0
                    abs_error = abs(int(p_hat) - int(args.p_true))

                    row = {
                        "status": "ok",
                        "n": n,
                        "power": power,
                        "seed_index": seed_index,
                        "seed": seed,
                        "p_true": args.p_true,
                        "q1": args.q1,
                        "q2": args.q2,
                        "d1_true": d1_true,
                        "d2_true": d2_true,
                        "dj_true": dj_true,
                        "d_obs": d_obs,
                        "degree": args.degree,
                        "d1_hat": d1_hat,
                        "d2_hat": d2_hat,
                        "dj_hat": dj_hat,
                        "p_hat": p_hat,
                        "abs_error": abs_error,
                        "success": int(bool(passed)),
                        "runtime_sec": f"{runtime:.2f}",
                        "detail_log": str(detail_log),
                        "error": "",
                    }
                    writer.writerow(row)
                    f_csv.flush()

                    print(
                        f"    result: d1={d1_hat}, d2={d2_hat}, dj={dj_hat}, "
                        f"p_hat={p_hat}, |err|={abs_error}, success={bool(passed)}, "
                        f"time={runtime/60:.1f} min"
                    )

                except Exception as e:
                    runtime = time.time() - t0
                    err = repr(e)
                    tb = traceback.format_exc()

                    with detail_log.open("a") as f_detail:
                        f_detail.write("\n\nERROR\n")
                        f_detail.write(tb)

                    row = {
                        "status": "error",
                        "n": n,
                        "power": power,
                        "seed_index": seed_index,
                        "seed": seed,
                        "p_true": args.p_true,
                        "q1": args.q1,
                        "q2": args.q2,
                        "d1_true": d1_true,
                        "d2_true": d2_true,
                        "dj_true": dj_true,
                        "d_obs": d_obs,
                        "degree": args.degree,
                        "d1_hat": "",
                        "d2_hat": "",
                        "dj_hat": "",
                        "p_hat": "",
                        "abs_error": "",
                        "success": 0,
                        "runtime_sec": f"{runtime:.2f}",
                        "detail_log": str(detail_log),
                        "error": err,
                    }
                    writer.writerow(row)
                    f_csv.flush()

                    print(f"    ERROR after {runtime/60:.1f} min: {err}")

    if args.make_plot:
        maybe_make_plot(csv_path, out_dir, "p_recovery_sweep.png")

    print("\nDone.")
    print(f"CSV saved at: {csv_path}")
    print(f"Detailed logs saved in: {detail_dir}")


if __name__ == "__main__":
    main()
