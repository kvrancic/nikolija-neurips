#!/usr/bin/env python3
"""Post-hoc Jaccard re-computation for an existing sweep CSV.

For each row of the input CSV, retrain flows for the recovered (d1_hat, d2_hat)
and recompute the Jaccard column using the new forward-sampling approach
(crl_sim/shared_coords.py:sample_zhat_paired). Writes a new CSV with the
updated Jaccard columns; leaves dim-recovery columns unchanged.

Usage:
    python scripts/rejaccard_csv.py results/p5/p_recovery_sweep.csv \
        --out results/p5/p_recovery_sweep_v2.csv \
        --gpus 0,1,2,3 \
        --jobs-per-gpu 1
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

# Lazy import torch/numpy in workers — keeps the parent process light.

JACCARD_OUTPUT_KEYS = [
    "jaccard", "shared_recovered",
    "jaccard_true_p", "shared_recovered_true_p",
    "p_hat_elbow", "jaccard_elbow", "shared_recovered_elbow", "elbow_gap",
    "p_count_error", "gap_at_true_p",
    "jaccard_aligned", "jaccard_aligned_true_p",
    "train_mmd_view1", "train_mmd_view2",
    "jaccard_reason",
]


def parse_gpus(s: str) -> List[Optional[int]]:
    if s.lower() in ("cpu", ""):
        return [None]
    return [int(x) for x in s.split(",") if x.strip()]


def worker(payload: Dict) -> Dict:
    """Recompute Jaccard for a single row. Runs in a child process so each
    worker can pin to one GPU via CUDA_VISIBLE_DEVICES.
    """
    gpu_id = payload["gpu_id"]
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Import inside the worker so the GPU env var takes effect.
    from crl_sim.shared_coords import jaccard_for_pipeline_run

    row = payload["row"]
    try:
        res = jaccard_for_pipeline_run(
            p_true=int(row["p_true"]),
            q1=int(row["q1"]),
            q2=int(row["q2"]),
            n=int(row["n"]),
            d_obs=int(row["d_obs"]),
            degree=int(row["degree"]),
            seed=int(row["seed"]),
            d1_hat=int(row["d1_hat"]),
            d2_hat=int(row["d2_hat"]),
            p_hat=int(row["p_hat"]),
            noise="exponential",
            gpu_id=0 if gpu_id is not None else None,  # CUDA_VISIBLE_DEVICES already remapped
            n_iter_train=payload["n_iter_train"],
            n_samples=payload["n_samples"],
            n_restarts=payload["n_restarts"],
            schur_K=payload["schur_K"],
            schur_n_eval=payload["schur_n_eval"],
        )
    except Exception as e:
        return {"row_idx": payload["row_idx"], "error": f"{type(e).__name__}: {e}"}

    out = {"row_idx": payload["row_idx"], "error": None}
    out["jaccard"] = res["jaccard"]
    out["shared_recovered"] = ",".join(str(x) for x in res["recovered"])
    out["jaccard_true_p"] = res["jaccard_true_p"]
    out["shared_recovered_true_p"] = ",".join(str(x) for x in res["recovered_true_p"])
    out["p_hat_elbow"] = res["p_hat_elbow"]
    out["jaccard_elbow"] = res["jaccard_elbow"]
    out["shared_recovered_elbow"] = ",".join(str(x) for x in res["recovered_elbow"])
    out["elbow_gap"] = res["gap_strength"]
    out["p_count_error"] = res["p_count_error"]
    out["gap_at_true_p"] = res["gap_at_true_p"]
    out["jaccard_aligned"] = res["jaccard_aligned"]
    out["jaccard_aligned_true_p"] = res["jaccard_aligned_true_p"]
    out["train_mmd_view1"] = res["train_mmd_view1"]
    out["train_mmd_view2"] = res["train_mmd_view2"]
    out["jaccard_reason"] = res.get("reason", "")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=str, help="Input sweep CSV.")
    parser.add_argument("--out", type=str, required=True, help="Output CSV path.")
    parser.add_argument("--gpus", type=str, default="0,1,2,3",
                        help="Comma-separated GPU IDs, or 'cpu'.")
    parser.add_argument("--n-iter-train", type=int, default=600,
                        help="Flow training iters per row (default 600).")
    parser.add_argument("--n-samples", type=int, default=768)
    parser.add_argument("--n-restarts", type=int, default=2)
    parser.add_argument("--schur-K", type=int, default=5)
    parser.add_argument("--schur-n-eval", type=int, default=4096)
    parser.add_argument("--only-status-ok", action="store_true", default=True,
                        help="Skip rows with status != 'ok'.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N rows (for testing).")
    args = parser.parse_args()

    in_path = Path(args.csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    print(f"loaded {len(rows)} rows from {in_path}")
    work_indices = []
    for i, row in enumerate(rows):
        if args.only_status_ok and row.get("status") != "ok":
            continue
        if args.limit is not None and len(work_indices) >= args.limit:
            break
        work_indices.append(i)
    print(f"will reprocess {len(work_indices)} rows")

    gpus = parse_gpus(args.gpus)
    n_workers = max(1, len(gpus))
    print(f"using {n_workers} workers (gpus={gpus})")

    payloads = []
    for k, idx in enumerate(work_indices):
        payloads.append({
            "row_idx": idx,
            "row": rows[idx],
            "gpu_id": gpus[k % len(gpus)],
            "n_iter_train": args.n_iter_train,
            "n_samples": args.n_samples,
            "n_restarts": args.n_restarts,
            "schur_K": args.schur_K,
            "schur_n_eval": args.schur_n_eval,
        })

    t0 = time.time()
    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(worker, p) for p in payloads]
        for fut in as_completed(futs):
            res = fut.result()
            completed += 1
            idx = res.pop("row_idx")
            err = res.pop("error", None)
            if err is not None:
                print(f"  [{completed}/{len(payloads)}] row {idx}: ERROR {err}")
                continue
            for k in JACCARD_OUTPUT_KEYS:
                if k in res:
                    rows[idx][k] = res[k]
            elapsed = time.time() - t0
            rate = completed / max(elapsed, 1e-6)
            eta = (len(payloads) - completed) / max(rate, 1e-6)
            row = rows[idx]
            print(f"  [{completed}/{len(payloads)}] n={row['n']} seed={row['seed']} "
                  f"jacc={row['jaccard']:.3f} elbow_p={row['p_hat_elbow']} "
                  f"gap={row['gap_at_true_p']:.2f}  ({elapsed/60:.1f}min, ETA {eta/60:.0f}min)")

    # Make sure all expected columns exist in the header
    for k in JACCARD_OUTPUT_KEYS:
        if k not in fieldnames:
            fieldnames = list(fieldnames) + [k]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
