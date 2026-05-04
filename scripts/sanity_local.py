#!/usr/bin/env python3
"""CPU-friendly smoke test: full pipeline + Theorem 1 / Jaccard at p=2.

Goal: validate end-to-end on a laptop in <30 minutes, no GPU. If this passes,
the cluster sweeps are extremely likely to produce sane numbers.

Uses the same code paths as the cluster sweep but with tiny iteration counts.

Pass criteria:
  - run_full_pipeline_once recovers p_hat == p_true (= 2)
  - jaccard_for_pipeline_run gives Jaccard >= 0.5 (loose; n is small and CPU
    is slow so we keep the bar low — the cluster runs are the real test).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

# Make the repo root importable regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from crl_sim.core import num_poly_features, run_full_pipeline_once
from crl_sim.shared_coords import jaccard_for_pipeline_run


def _build_args() -> SimpleNamespace:
    p_true, q1, q2 = 2, 2, 2
    degree = 2

    # d_obs needs to fit polynomial features for the joint case:
    #   joint feat count for dim p+q1+q2 = 6 with deg=2 is 6 + 21 = 27,
    #   so per-view d_obs must give 2*d_obs >= 27. We pick 30 with buffer.
    view_need = max(num_poly_features(p_true + q1, degree),
                    num_poly_features(p_true + q2, degree))
    joint_need_per_view = int(np.ceil(num_poly_features(p_true + q1 + q2, degree) / 2))
    d_obs = max(view_need, joint_need_per_view) + 10

    return SimpleNamespace(
        # Data
        p_true=p_true, q1=q1, q2=q2,
        d_obs=d_obs, degree=degree, n=2048,

        # Compute
        gpu_ids=[None],          # CPU only
        n_restarts=2,            # reduce from default 4 to keep CPU time down
        n_iter=200,              # reduce from default 400-800
        n_samples=400,
        n_samples_joint=512,
        lr=1e-2,
        max_val=1024,

        # Scan and selection
        max_dim_view=None, max_dim_joint=None, extra_dim_buffer=2,
        threshold=None,
        selection_abs_tol=2e-3, selection_rel_tol=0.05,
        view_floor_slack=5e-3, joint_floor_slack=5e-4,
        next_improvement_abs=2e-3, next_improvement_rel=0.10,
    )


def main() -> int:
    print("=" * 70)
    print("SANITY LOCAL — CPU smoke test for the CRL pipeline + Theorem 1")
    print("=" * 70)
    print(f"torch={torch.__version__}, cuda_available={torch.cuda.is_available()} (using CPU)")

    args = _build_args()
    seed = 42

    print(f"\nConfig: p={args.p_true}, q1={args.q1}, q2={args.q2}, "
          f"d_obs={args.d_obs}, degree={args.degree}, n={args.n}")
    print(f"Compute knobs: n_iter={args.n_iter}, n_restarts={args.n_restarts}, "
          f"n_samples={args.n_samples}")
    print(f"Expected: d1=d2={args.p_true + args.q1}, dj={args.p_true + args.q1 + args.q2}, "
          f"p_hat={args.p_true}; Jaccard >= 0.5 on this small CPU run.\n")

    # ---------------- Stage 1: full inclusion-exclusion pipeline ----------------
    print("-" * 70)
    print("Stage 1: run_full_pipeline_once (dim recovery + inclusion-exclusion)")
    print("-" * 70)

    t0 = time.time()
    passed, d1_hat, d2_hat, dj_hat, p_hat = run_full_pipeline_once(
        args, seed=seed, label="sanity_local"
    )
    t_stage1 = time.time() - t0

    print(f"\n[stage1] d1_hat={d1_hat}, d2_hat={d2_hat}, dj_hat={dj_hat}, p_hat={p_hat}")
    print(f"[stage1] dim-recovery passed={bool(passed)}, time={t_stage1/60:.2f} min")

    # ---------------- Stage 2: Theorem 1 / Jaccard ----------------
    print("\n" + "-" * 70)
    print("Stage 2: jaccard_for_pipeline_run (Theorem 1, Schur form)")
    print("-" * 70)

    t1 = time.time()
    jres = jaccard_for_pipeline_run(
        p_true=args.p_true, q1=args.q1, q2=args.q2,
        n=args.n, d_obs=args.d_obs, degree=args.degree,
        seed=seed,
        d1_hat=int(d1_hat), d2_hat=int(d2_hat), p_hat=int(p_hat),
        gpu_id=None,
        n_iter_train=args.n_iter,
        n_iter_inv=200,
        n_samples=args.n_samples,
        lr=args.lr,
        n_restarts=2,
    )
    t_stage2 = time.time() - t1

    print(f"\n[stage2] jaccard      = {jres['jaccard']}")
    print(f"[stage2] recovered    = {jres['recovered']}")
    print(f"[stage2] ground_truth = {jres['ground_truth']}")
    print(f"[stage2] sigma_diag   = {[round(v, 4) for v in jres['sigma_diag']]}")
    print(f"[stage2] view1 train MMD = {jres['train_mmd_view1']:.5f}, "
          f"view2 train MMD = {jres['train_mmd_view2']:.5f}")
    if jres["reason"]:
        print(f"[stage2] reason       = {jres['reason']}")
    print(f"[stage2] time={t_stage2/60:.2f} min")

    # ---------------- Verdict ----------------
    print("\n" + "=" * 70)
    print("FINAL")
    print("=" * 70)

    dim_ok = bool(passed)
    j = jres["jaccard"]
    j_ok = (j == j) and (j >= 0.5)  # NaN-safe

    print(f"  dimension recovery (p_hat == p_true):  {'OK' if dim_ok else 'FAIL'}")
    print(f"  Jaccard >= 0.5:                        {'OK' if j_ok else 'FAIL'} (got {j})")
    print(f"  total time: {(t_stage1 + t_stage2) / 60:.2f} min")

    if dim_ok and j_ok:
        print("\n  ALL SANITY CHECKS PASSED.")
        return 0

    print("\n  AT LEAST ONE CHECK FAILED — inspect output before launching cluster sweeps.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
