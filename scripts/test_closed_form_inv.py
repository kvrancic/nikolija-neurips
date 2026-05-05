#!/usr/bin/env python3
"""Closed-form inversion: Ẑ = (X @ pinv(Ĝ^T))[:d_hat]
since poly's first d_hat features are linear (=Z) by construction."""
import numpy as np
import torch
from crl_sim.core import generate_scm, sample_er_dag, make_poly_fn, num_poly_features
from crl_sim.shared_coords import (
    regenerate_paired_data,
    train_view_flow,
    sigma_1_given_2,
    jaccard,
)


def closed_form_invert(x_np, G, d_hat, degree):
    """Ẑ = (X @ G^{T+})[:d_hat]  — direct projection through trained decoder."""
    G_np = G.detach().cpu().numpy().astype(np.float64)  # (d_obs, n_feat)
    # X = poly(Z) @ G^T  →  poly(Z) ≈ X @ pinv(G^T)
    G_T = G_np.T  # (n_feat, d_obs)
    pinv_GT = np.linalg.pinv(G_T)  # (d_obs, n_feat)  -- but pinv of n_feat×d_obs is d_obs×n_feat
    # Wait. X @ pinv(G^T): X is (n, d_obs), pinv(G^T) is what we want (d_obs, n_feat) so X @ pinv(G^T) gives (n, n_feat).
    # pinv of (n_feat, d_obs) matrix = (d_obs, n_feat). That's right.
    poly_z = x_np.astype(np.float64) @ pinv_GT  # (n, n_feat)
    return poly_z[:, :d_hat], poly_z


def main():
    p, q1, q2, n, seed = 2, 2, 2, 2048, 11

    rng = np.random.RandomState(seed)
    W = sample_er_dag(p, q1, q2, 0.4, rng)
    Z = generate_scm(W, n, rng, noise="exponential")
    z1_true = Z[:, : p + q1].astype(np.float64)
    z2_true = np.hstack([Z[:, :p], Z[:, p + q1:]]).astype(np.float64)
    d1, d2 = z1_true.shape[1], z2_true.shape[1]

    n_features1 = num_poly_features(d1, 2)
    d_obs = max(int(0.5 * (n + n_features1)), 32)
    print(f"d_obs={d_obs}, n_feat per view={n_features1}")

    x1, x2 = regenerate_paired_data(p, q1, q2, n, d_obs, degree=2, seed=seed,
                                     return_z=False)

    print("\nTraining flows ...")
    flow1, G1, dev1, mmd1 = train_view_flow(x1, d1, degree=2, n_iter=400, n_samples=768,
                                              seed=seed*13+1, n_restarts=2)
    flow2, G2, dev2, mmd2 = train_view_flow(x2, d2, degree=2, n_iter=400, n_samples=768,
                                              seed=seed*13+2, n_restarts=2)
    print(f"  view-1 MMD={mmd1:.6f}  view-2 MMD={mmd2:.6f}")

    z1_cf, poly_z1 = closed_form_invert(x1, G1, d1, 2)
    z2_cf, poly_z2 = closed_form_invert(x2, G2, d2, 2)

    # Check consistency: poly(z1_cf)[d_hat:] should equal the quadratic part recovered
    # (this checks if the closed-form inversion produces a CONSISTENT poly representation).
    # Compute the implied quadratic part from z1_cf:
    z1_cf_t = torch.tensor(z1_cf, dtype=torch.float32)
    phi = make_poly_fn(d1, 2, torch.device("cpu"))
    poly_implied = phi(z1_cf_t).numpy()
    # poly_z1 has shape (n, n_feat). poly_implied has same shape.
    quad_recov = poly_z1[:, d1:]
    quad_implied = poly_implied[:, d1:]
    consistency = np.corrcoef(quad_recov.ravel(), quad_implied.ravel())[0, 1]
    print(f"\nConsistency check (corr(implied quadratic, recovered quadratic)): {consistency:.4f}")

    # Compare z_cf to z_true
    z1n = (z1_cf - z1_cf.mean(0)) / (z1_cf.std(0) + 1e-8)
    z1tn = (z1_true - z1_true.mean(0)) / (z1_true.std(0) + 1e-8)
    corr = np.abs(z1n.T @ z1tn) / max(n, 1)
    print(f"\n|corr(Ẑ_1_closed_form, Z_1)|:")
    for i in range(d1):
        print(f"  Ẑ[{i}]: " + " ".join(f"{c:.2f}" for c in corr[i])
              + f"   max={corr[i].max():.2f}")

    # Schur via closed-form inversion
    sigma = sigma_1_given_2(z1_cf, z2_cf)
    diag = np.diag(sigma)
    smallest = sorted(np.argsort(diag)[:p].tolist())
    j = jaccard(smallest, list(range(p)))
    print(f"\nSchur diagonal (closed-form): {[round(x,4) for x in diag]}")
    print(f"  smallest-{p}: {smallest}  truth: {list(range(p))}  jaccard: {j:.3f}")

    # Linear regression: how well does Ẑ_cf fit Z linearly?
    Z_aug = np.hstack([z1_true, np.ones((n, 1))])
    sol, *_ = np.linalg.lstsq(Z_aug, z1_cf, rcond=None)
    z1_pred = Z_aug @ sol
    err = np.mean((z1_cf - z1_pred) ** 2)
    var = np.var(z1_cf)
    print(f"\nLinear fit Ẑ = Z @ A + b: MSE={err:.4f} / var(Ẑ)={var:.4f}  →  R² = {1 - err/var:.3f}")


if __name__ == "__main__":
    main()
