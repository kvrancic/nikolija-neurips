#!/usr/bin/env python3
"""Diagnostic: if we use ORACLE alignment (least-squares fit of Ẑ to true Z
in simulation), does Theorem 1 fire correctly?

This tests: does Theorem 1 work IF we somehow had the right Λ_1, Λ_2?
- If yes → algorithm is missing an alignment step. We need to find Λ unsupervised.
- If no → there's a deeper issue.
"""
import numpy as np
from crl_sim.core import generate_scm, sample_er_dag
from crl_sim.shared_coords import (
    regenerate_paired_data,
    train_view_flow,
    recover_paired_z,
    sigma_1_given_2,
    detect_p_via_elbow,
    spectral_gap_at,
    jaccard,
)


def fit_affine(z_hat, z_true):
    """Solve z_hat = z_true @ A + b in least squares.
    Returns A (d_true × d_hat) and b (d_hat)."""
    Z = np.hstack([z_true, np.ones((z_true.shape[0], 1))])
    sol, *_ = np.linalg.lstsq(Z, z_hat, rcond=None)
    A = sol[:-1]
    b = sol[-1]
    return A, b


def main():
    p, q1, q2, n, seed = 2, 2, 2, 2048, 11

    rng = np.random.RandomState(seed)
    W = sample_er_dag(p, q1, q2, 0.4, rng)
    Z = generate_scm(W, n, rng, noise="exponential")
    z1_true = Z[:, : p + q1].astype(np.float64)
    z2_true = np.hstack([Z[:, :p], Z[:, p + q1:]]).astype(np.float64)
    d1, d2 = z1_true.shape[1], z2_true.shape[1]

    n_features1 = d1 + d1 * (d1 + 1) // 2
    d_obs = max(int(0.5 * (n + n_features1)), 32)

    x1, x2 = regenerate_paired_data(p, q1, q2, n, d_obs, degree=2, seed=seed,
                                     return_z=False)

    print("Training view-1 flow ...")
    flow1, G1, dev1, mmd1 = train_view_flow(x1, d1, degree=2, n_iter=400, n_samples=768,
                                              seed=seed*13+1, n_restarts=2)
    print("Training view-2 flow ...")
    flow2, G2, dev2, mmd2 = train_view_flow(x2, d2, degree=2, n_iter=400, n_samples=768,
                                              seed=seed*13+2, n_restarts=2)

    z1_hat = recover_paired_z(x1, flow1, G1, degree=2, dev=dev1, n_iter=300, seed=seed*13+3)
    z2_hat = recover_paired_z(x2, flow2, G2, degree=2, dev=dev2, n_iter=300, seed=seed*13+4)

    # Naive Schur on Ẑ
    sigma_naive = sigma_1_given_2(z1_hat.astype(np.float64), z2_hat.astype(np.float64))
    diag_naive = np.diag(sigma_naive)
    j_naive = jaccard(sorted(np.argsort(diag_naive)[:p].tolist()), list(range(p)))
    print(f"\n[naive Schur on Ẑ_1, Ẑ_2 from flow]")
    print(f"  diag: {[round(x,3) for x in diag_naive]}")
    print(f"  jaccard against [0..{p-1}]: {j_naive:.3f}")
    print(f"  spectral gap at p={p}: {spectral_gap_at(diag_naive, p):.3f}")

    # Oracle alignment: fit linear Ẑ = Z @ A + b, then invert to get Ẑ_aligned ≈ Z
    A1, b1 = fit_affine(z1_hat.astype(np.float64), z1_true)
    A2, b2 = fit_affine(z2_hat.astype(np.float64), z2_true)
    print(f"\nA1 (true→hat) shape: {A1.shape}, A1 cond: {np.linalg.cond(A1):.2f}")
    print(f"A2 (true→hat) shape: {A2.shape}, A2 cond: {np.linalg.cond(A2):.2f}")

    # Apply A1^{-1}: Ẑ_aligned = (Ẑ - b) @ A^{-1}, but only if A is square invertible.
    # If not, use pseudo-inverse.
    A1_inv = np.linalg.pinv(A1)
    A2_inv = np.linalg.pinv(A2)
    z1_aligned = (z1_hat.astype(np.float64) - b1) @ A1_inv  # ≈ z1_true if linear relation holds
    z2_aligned = (z2_hat.astype(np.float64) - b2) @ A2_inv

    err1 = np.mean((z1_aligned - z1_true) ** 2)
    err2 = np.mean((z2_aligned - z2_true) ** 2)
    var1 = np.var(z1_true)
    var2 = np.var(z2_true)
    print(f"\nResidual error after oracle linear alignment:")
    print(f"  view 1: MSE={err1:.4f}  /  var(z1)={var1:.4f}  →  R² = {1 - err1/var1:.3f}")
    print(f"  view 2: MSE={err2:.4f}  /  var(z2)={var2:.4f}  →  R² = {1 - err2/var2:.3f}")
    print("  (R² close to 1 → Ẑ is linearly related to Z; close to 0 → nonlinear/scrambled)")

    # Schur after oracle alignment
    sigma_aligned = sigma_1_given_2(z1_aligned, z2_aligned)
    diag_aligned = np.diag(sigma_aligned)
    j_aligned = jaccard(sorted(np.argsort(diag_aligned)[:p].tolist()), list(range(p)))
    print(f"\n[Schur on oracle-aligned Ẑ]")
    print(f"  diag: {[round(x,3) for x in diag_aligned]}")
    print(f"  jaccard: {j_aligned:.3f}")
    print(f"  spectral gap at p={p}: {spectral_gap_at(diag_aligned, p):.3f}")

    # Schur on TRUE Z (sanity)
    sigma_true = sigma_1_given_2(z1_true, z2_true)
    diag_true = np.diag(sigma_true)
    print(f"\n[sanity: Schur on TRUE Z]")
    print(f"  diag: {[round(x,4) for x in diag_true]}")


if __name__ == "__main__":
    main()
