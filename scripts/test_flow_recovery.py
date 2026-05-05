#!/usr/bin/env python3
"""Diagnostic: check whether the trained flow + per-sample inversion recovers
Ẑ that's related to true Z by a permutation (the assumption Theorem 1 needs).

We:
  1. Generate paired X1, X2 from a known SCM.
  2. Train view-1 flow at d_hat = d1_true.
  3. Recover Ẑ_1 by per-sample inversion of X1.
  4. Compare corr(Ẑ_1, Z_1) — if it's a permutation matrix-ish pattern, good.
  5. Also compute Sigma_{1|2}(Ẑ_1, Ẑ_2) and check shared-coord recovery.

If correlation is diagonal-ish-permuted: flow is fine, alignment should work.
If correlation is dense/scrambled: flow is mixing latents, Theorem 1 won't fire.
"""
from __future__ import annotations

import sys
import numpy as np

from crl_sim.core import generate_scm, sample_er_dag
from crl_sim.shared_coords import (
    regenerate_paired_data,
    train_view_flow,
    recover_paired_z,
    sigma_1_given_2,
    recover_shared_indices,
    detect_p_via_elbow,
    spectral_gap_at,
    jaccard,
    greedy_alignment,
)


def run_one(p_true, q1, q2, n, seed, n_iter_train=400, n_iter_inv=300, n_restarts=2):
    print(f"\n{'='*70}")
    print(f"p={p_true} q1={q1} q2={q2} n={n} seed={seed}")
    print(f"{'='*70}")

    # Reuse the same recipe as the sweep (d_obs from the existing rule).
    n_features1 = (q1 + p_true) + (q1 + p_true) * (q1 + p_true + 1) // 2
    d_obs = max(int(0.5 * (n + n_features1)), 32)
    print(f"d_obs={d_obs}")

    rng = np.random.RandomState(seed)
    W = sample_er_dag(p_true, q1, q2, 0.4, rng)
    Z = generate_scm(W, n, rng, noise="exponential")
    z1_true = Z[:, : p_true + q1].astype(np.float64)
    z2_true = np.hstack([Z[:, :p_true], Z[:, p_true + q1 :]]).astype(np.float64)
    d1_true = z1_true.shape[1]
    d2_true = z2_true.shape[1]

    # Re-make the same X1, X2 the sweep would have used (uses regenerate_paired_data
    # which constructs X with its own RNG).
    x1, x2 = regenerate_paired_data(p_true, q1, q2, n, d_obs, degree=2, seed=seed,
                                     return_z=False)

    # ---- Sigma on TRUE Z ----
    sigma_true = sigma_1_given_2(z1_true, z2_true)
    diag_true = np.diag(sigma_true)
    print("\n[sanity] diag(Sigma_{1|2}) on TRUE Z:")
    print(f"  shared [0..{p_true-1}]: {[round(x,4) for x in diag_true[:p_true]]}")
    print(f"  private [{p_true}..]: {[round(x,4) for x in diag_true[p_true:]]}")

    # ---- Train flows ----
    print("\nTraining view-1 flow ...")
    flow1, G1, dev1, mmd1 = train_view_flow(
        x1, d1_true, degree=2, n_iter=n_iter_train, n_samples=768,
        gpu_id=None, seed=seed * 13 + 1, n_restarts=n_restarts,
    )
    print(f"  view-1 flow trained, train MMD = {mmd1:.6f}")
    print("Training view-2 flow ...")
    flow2, G2, dev2, mmd2 = train_view_flow(
        x2, d2_true, degree=2, n_iter=n_iter_train, n_samples=768,
        gpu_id=None, seed=seed * 13 + 2, n_restarts=n_restarts,
    )
    print(f"  view-2 flow trained, train MMD = {mmd2:.6f}")

    # ---- Invert ----
    print("\nPer-sample inversion ...")
    z1_hat = recover_paired_z(x1, flow1, G1, degree=2, dev=dev1,
                              n_iter=n_iter_inv, seed=seed * 13 + 3)
    z2_hat = recover_paired_z(x2, flow2, G2, degree=2, dev=dev2,
                              n_iter=n_iter_inv, seed=seed * 13 + 4)

    print(f"  z1_hat shape={z1_hat.shape}  z1_true shape={z1_true.shape}")
    print(f"  z1_hat std per col: {[round(float(s),3) for s in z1_hat.std(0)]}")

    # ---- Correlation: how related is Ẑ_1 to Z_1? ----
    z1_hat_n = (z1_hat - z1_hat.mean(0)) / (z1_hat.std(0) + 1e-8)
    z1_true_n = (z1_true - z1_true.mean(0)) / (z1_true.std(0) + 1e-8)
    corr = (z1_hat_n.T @ z1_true_n) / max(n, 1)
    print("\n|corr(Ẑ_1, Z_1)|  (rows=Ẑ_1 dims, cols=Z_1 dims with shared at [:p]):")
    abs_corr = np.abs(corr)
    for i in range(d1_true):
        line = "  " + " ".join(f"{abs_corr[i,j]:.2f}" for j in range(d1_true))
        line += f"   max={abs_corr[i].max():.2f} at col {abs_corr[i].argmax()}"
        print(line)

    # ---- Sigma on Ẑ ----
    sigma_hat = sigma_1_given_2(z1_hat.astype(np.float64), z2_hat.astype(np.float64))
    diag_hat = np.diag(sigma_hat)
    print("\ndiag(Sigma_{1|2}) on RECOVERED Ẑ (sorted):")
    sorted_idx = np.argsort(diag_hat)
    for rank, idx in enumerate(sorted_idx):
        marker = "  ← shared" if rank < p_true else ""
        # The ground-truth-aligned position of this Ẑ_1 dim:
        aligned_to = int(np.argmax(abs_corr[idx]))
        is_shared_in_truth = aligned_to < p_true
        truth_label = " (true: SHARED)" if is_shared_in_truth else " (true: private)"
        print(f"  rank {rank}: Ẑ_1[{idx:>2}] = {diag_hat[idx]:.4f}{marker}{truth_label}")

    # ---- Final scoring ----
    smallest = sorted(np.argsort(diag_hat)[:p_true].tolist())
    truth = list(range(p_true))
    j_idx = jaccard(smallest, truth)
    p_elbow, gap = detect_p_via_elbow(diag_hat)
    matching = greedy_alignment(z1_hat, z1_true)
    aligned_smallest = sorted({matching.get(int(i), -1) for i in smallest} - {-1})
    j_aln = jaccard(aligned_smallest, truth)
    print(f"\nJaccard (index-level, no alignment) = {j_idx:.3f}")
    print(f"Jaccard (alignment-based)            = {j_aln:.3f}")
    print(f"Elbow detection: p_hat={p_elbow}, gap={gap:.3f}")
    print(f"Spectral gap at true p: {spectral_gap_at(diag_hat, p_true):.3f}  (truth: ~13)")


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "small"
    if cfg == "small":
        run_one(p_true=2, q1=2, q2=2, n=2048, seed=11)
    elif cfg == "medium":
        run_one(p_true=5, q1=5, q2=5, n=2048, seed=11)
    elif cfg == "large":
        run_one(p_true=5, q1=5, q2=5, n=8192, seed=11)
