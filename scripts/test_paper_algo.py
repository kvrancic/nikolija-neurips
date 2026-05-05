#!/usr/bin/env python3
"""Test the EXACT Algorithm 4 step 7 from CRL paper:

  Ẑ⁽ᵉ⁾ ← (1/K) Σ_{k=1}^K f_e(ε_k),  ε_k ~ N(0, I),  for e = 1, 2

The key insight: ε_k is SHARED between f_1 and f_2 (no `e` subscript). With
shared noise input and independently trained flows, the implicit identifiability
makes the *positions* of shared coords in Ẑ⁽¹⁾ and Ẑ⁽²⁾ aligned (each flow's
first p output dims correspond to shared latents driven by the same ε[:p]).

This replaces the per-sample inversion approach in shared_coords.py.
"""
import numpy as np
import torch
from crl_sim.core import generate_scm, sample_er_dag, num_poly_features, set_all_seeds
from crl_sim.shared_coords import (
    regenerate_paired_data,
    train_view_flow,
    sigma_1_given_2,
    detect_p_via_elbow,
    spectral_gap_at,
    jaccard,
)


def sample_zhat_paired(flow1, flow2, dev1, dev2, n_eval, K, d1, d2, seed=0):
    """Sample Ẑ⁽¹⁾, Ẑ⁽²⁾ from both flows using SHARED ε_k batches, averaged over K.

    To couple flows: draw ε of dim max(d1, d2). Pass ε[:d1] to flow_1 and ε[:d2] to flow_2.
    The SHARED prefix of ε is the same for both flows -- this is what couples them.
    """
    set_all_seeds(seed)
    d_max = max(d1, d2)

    z1_acc = None
    z2_acc = None
    for k in range(K):
        # SAME eps batch for both flows
        eps_full = torch.randn(n_eval, d_max, device=dev1)  # assume same device
        eps1 = eps_full[:, :d1].to(dev1)
        eps2 = eps_full[:, :d2].to(dev2)
        with torch.no_grad():
            z1k = flow1(eps1).cpu().numpy().astype(np.float64)
            z2k = flow2(eps2).cpu().numpy().astype(np.float64)
        z1_acc = z1k if z1_acc is None else (z1_acc + z1k)
        z2_acc = z2k if z2_acc is None else (z2_acc + z2k)
    return z1_acc / K, z2_acc / K


def main():
    cfg_list = [
        (2, 2, 2, 2048, 11),
        (2, 2, 2, 2048, 12),
        (2, 2, 2, 2048, 13),
        (5, 5, 5, 2048, 11),
        (5, 5, 5, 4096, 11),
    ]
    for p, q1, q2, n, seed in cfg_list:
        run(p, q1, q2, n, seed)


def run(p, q1, q2, n, seed, K=5, n_eval=2000):
    rng = np.random.RandomState(seed)
    W = sample_er_dag(p, q1, q2, 0.4, rng)
    Z = generate_scm(W, n, rng, noise="exponential")
    z1_true = Z[:, : p + q1].astype(np.float64)
    z2_true = np.hstack([Z[:, :p], Z[:, p + q1 :]]).astype(np.float64)
    d1, d2 = z1_true.shape[1], z2_true.shape[1]
    n_features1 = num_poly_features(d1, 2)
    d_obs = max(int(0.5 * (n + n_features1)), 32)

    x1, x2 = regenerate_paired_data(p, q1, q2, n, d_obs, degree=2, seed=seed,
                                     return_z=False)

    print(f"\n=== p={p} q1={q1} q2={q2} n={n} seed={seed} ===")
    flow1, G1, dev1, mmd1 = train_view_flow(x1, d1, degree=2, n_iter=400, n_samples=768,
                                              seed=seed*13+1, n_restarts=2)
    flow2, G2, dev2, mmd2 = train_view_flow(x2, d2, degree=2, n_iter=400, n_samples=768,
                                              seed=seed*13+2, n_restarts=2)
    print(f"  MMD: view1={mmd1:.5f}  view2={mmd2:.5f}")

    z1_hat, z2_hat = sample_zhat_paired(flow1, flow2, dev1, dev2,
                                          n_eval=n_eval, K=K, d1=d1, d2=d2, seed=seed*17)

    # Schur
    sigma = sigma_1_given_2(z1_hat, z2_hat)
    diag = np.diag(sigma)
    print(f"  diag(Sigma_1|2): {[round(x,3) for x in diag]}")

    smallest = sorted(np.argsort(diag)[:p].tolist())
    j = jaccard(smallest, list(range(p)))
    print(f"  smallest-{p}: {smallest}  truth: {list(range(p))}  jaccard: {j:.3f}")
    p_elbow, gap = detect_p_via_elbow(diag)
    print(f"  elbow: p_hat={p_elbow}  gap={gap:.3f}")
    print(f"  spectral_gap at true p={p}: {spectral_gap_at(diag, p):.3f}")


if __name__ == "__main__":
    main()
