#!/usr/bin/env python3
"""Does the flow recover Ẑ axis-aligned with Z if we train MUCH harder?

If yes, the original Jaccard plumbing is fine and just needs more iters.
If no, the algorithm is fundamentally not axis-alignment-preserving."""

import numpy as np
from crl_sim.core import generate_scm, sample_er_dag
from crl_sim.shared_coords import (
    regenerate_paired_data,
    train_view_flow,
    recover_paired_z,
    sigma_1_given_2,
    jaccard,
)


def run(p, q1, q2, n, seed, n_iter_train, n_iter_inv, n_restarts):
    rng = np.random.RandomState(seed)
    W = sample_er_dag(p, q1, q2, 0.4, rng)
    Z = generate_scm(W, n, rng, noise="exponential")
    z1_true = Z[:, : p + q1].astype(np.float64)
    z2_true = np.hstack([Z[:, :p], Z[:, p + q1 :]]).astype(np.float64)
    d1, d2 = z1_true.shape[1], z2_true.shape[1]

    n_features1 = d1 + d1 * (d1 + 1) // 2
    d_obs = max(int(0.5 * (n + n_features1)), 32)

    x1, x2 = regenerate_paired_data(p, q1, q2, n, d_obs, degree=2, seed=seed,
                                     return_z=False)

    flow1, G1, dev1, mmd1 = train_view_flow(
        x1, d1, degree=2, n_iter=n_iter_train, n_samples=768,
        gpu_id=None, seed=seed * 13 + 1, n_restarts=n_restarts,
    )
    flow2, G2, dev2, mmd2 = train_view_flow(
        x2, d2, degree=2, n_iter=n_iter_train, n_samples=768,
        gpu_id=None, seed=seed * 13 + 2, n_restarts=n_restarts,
    )
    z1_hat = recover_paired_z(x1, flow1, G1, degree=2, dev=dev1, n_iter=n_iter_inv, seed=seed*13+3)
    z2_hat = recover_paired_z(x2, flow2, G2, degree=2, dev=dev2, n_iter=n_iter_inv, seed=seed*13+4)

    # Correlation Ẑ ↔ Z
    z1n = (z1_hat - z1_hat.mean(0)) / (z1_hat.std(0) + 1e-8)
    z1tn = (z1_true - z1_true.mean(0)) / (z1_true.std(0) + 1e-8)
    corr = (z1n.T @ z1tn) / max(n, 1)
    abs_corr = np.abs(corr)
    max_corr_per_row = abs_corr.max(axis=1)

    # Schur
    sigma = sigma_1_given_2(z1_hat.astype(np.float64), z2_hat.astype(np.float64))
    diag = np.diag(sigma)
    smallest = sorted(np.argsort(diag)[:p].tolist())
    truth = list(range(p))
    j = jaccard(smallest, truth)

    print(f"  iters={n_iter_train:>5} restarts={n_restarts}: "
          f"mmd1={mmd1:.5f} mmd2={mmd2:.5f}  "
          f"max|corr| per row={[f'{c:.2f}' for c in max_corr_per_row]}  "
          f"jacc={j:.3f}")


if __name__ == "__main__":
    print("p=2 case, n=2048, seed=11")
    print("Goal: see if max|corr| per row → 1 with more training (= permutation-aligned).")
    print()
    for n_iter in [400, 1500, 5000]:
        run(p=2, q1=2, q2=2, n=2048, seed=11,
            n_iter_train=n_iter, n_iter_inv=500, n_restarts=2)
