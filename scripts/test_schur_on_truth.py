#!/usr/bin/env python3
"""Diagnostic: validate the Schur-complement step on GROUND-TRUTH paired Z.

If Theorem 1 is implemented correctly, Sigma_{1|2}(z1_true, z2_true) should
have zero (or near-zero) diagonals at indices [0..p-1] and strictly positive
diagonals at [p..p+q1-1]. If THIS fails, the bug is in our math/code, not in
flow inversion. If this passes, the bug is downstream.

Why this matters: Nikolija pointed out Jaccard plateaus at ~0.4 even when
|p - p_hat| ≈ 0 — once dims are right, Jaccard should automatically be near 1.
"""

from __future__ import annotations

import numpy as np

from crl_sim.core import generate_scm, sample_er_dag, get_obs_and_G
from crl_sim.shared_coords import (
    sigma_1_given_2,
    recover_shared_indices,
    detect_p_via_elbow,
    spectral_gap_at,
    jaccard,
)


def regenerate_with_both_z(p_true, q1, q2, n, seed, noise="exponential", er_prob=0.4):
    """Return (z1_true, z2_true) with shared coords at indices [0..p-1] in both."""
    rng = np.random.RandomState(seed)
    W = sample_er_dag(p_true, q1, q2, er_prob, rng)
    Z = generate_scm(W, n, rng, noise=noise)
    zl = Z[:, :p_true]
    zi1 = Z[:, p_true : p_true + q1]
    zi2 = Z[:, p_true + q1 :]
    z1 = np.hstack([zl, zi1]).astype(np.float64)
    z2 = np.hstack([zl, zi2]).astype(np.float64)
    return z1, z2


def report(z1, z2, p_true, label):
    """Compute Sigma_{1|2}, print diag and shared-index recovery."""
    sigma = sigma_1_given_2(z1, z2, ridge=1e-8)
    diag = np.diag(sigma)

    sorted_idx = np.argsort(diag)
    smallest_p = sorted(sorted_idx[:p_true].tolist())
    truth = list(range(p_true))
    j = jaccard(smallest_p, truth)

    print(f"\n[{label}]  n={z1.shape[0]} d1={z1.shape[1]} d2={z2.shape[1]} p_true={p_true}")
    print(f"  diag(Sigma_1|2) at SHARED indices [0..{p_true-1}]:")
    for i in range(p_true):
        print(f"    [{i:2}]  {diag[i]:.6f}")
    print(f"  diag(Sigma_1|2) at PRIVATE indices [{p_true}..]:")
    for i in range(p_true, len(diag)):
        print(f"    [{i:2}]  {diag[i]:.6f}")
    print(f"  argsort(diag) = {sorted_idx.tolist()}")
    print(f"  smallest-{p_true} indices: {smallest_p}  truth: {truth}")
    print(f"  Jaccard = {j:.4f}")
    print(f"  spectral_gap_at(p_true): {spectral_gap_at(diag, p_true):.4f}")
    p_elbow, gap = detect_p_via_elbow(diag)
    print(f"  elbow detection: p_hat={p_elbow}, gap_strength={gap:.4f}")
    return j


def main():
    cases = [
        # (p, q1, q2, n)
        (2, 2, 2, 2048),  # tiny case from sanity_local
        (5, 5, 5, 2048),  # the p=5 sweep
        (5, 5, 5, 16384),
        (20, 20, 20, 4096),
    ]
    for p, q1, q2, n in cases:
        for seed in [11, 12, 13]:
            z1, z2 = regenerate_with_both_z(p, q1, q2, n, seed=seed)
            report(z1, z2, p, label=f"p={p} q1={q1} q2={q2} n={n} seed={seed}")


if __name__ == "__main__":
    main()
