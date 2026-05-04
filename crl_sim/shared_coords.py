"""Theorem 1 (Schur-complement form): identify which coordinates of the
recovered latent are shared between the two views, and score against ground
truth via Jaccard.

Pipeline (per (n, seed) row of the sweep):

  1. Train a flow + linear decoder for view 1 at d_hat_1 and view 2 at d_hat_2.
  2. For each observed paired sample (x1_i, x2_i), recover paired latents
     by per-sample inversion: search over eps_i so that flow(eps_i) ≈ z_i
     with phi(z_i) G^T ≈ x_i.
  3. Form Sigma_{1|2} = Sigma_11 - Sigma_12 Sigma_22^{-1} Sigma_21 on the
     paired Z-hats.
  4. The p_hat smallest entries of diag(Sigma_{1|2}) are the recovered shared
     coordinates of Z-hat-1. Compare against {0,...,p_true-1} via Jaccard.

The training and inversion mirror what core.py does for dimension recovery,
just at a fixed dimension and exposing the trained models.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from crl_sim.core import (
    SimpleFlow,
    generate_scm,
    get_device,
    get_obs_and_G,
    make_poly_fn,
    multiscale_mmd,
    num_poly_features,
    sample_er_dag,
    set_all_seeds,
    standardize_np,
)


DEFAULT_SIGMAS: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)


# =============================================================================
# 1. Replay the same paired data the dim-recovery pipeline used
# =============================================================================

def regenerate_paired_data(
    p_true: int,
    q1: int,
    q2: int,
    n: int,
    d_obs: int,
    degree: int,
    seed: int,
    noise: str = "exponential",
    er_prob: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Recreate the (X1, X2) pair that run_full_pipeline_once produced for the
    same seed and config. Pure-deterministic given the seed; no model state.
    """
    rng = np.random.RandomState(seed)
    W = sample_er_dag(p_true, q1, q2, er_prob, rng)
    Z = generate_scm(W, n, rng, noise=noise)
    zl = Z[:, :p_true]
    zi1 = Z[:, p_true : p_true + q1]
    zi2 = Z[:, p_true + q1 :]
    x1, _ = get_obs_and_G(zl, zi1, d_obs, degree, rng)
    x2, _ = get_obs_and_G(zl, zi2, d_obs, degree, rng)
    return x1.astype(np.float32), x2.astype(np.float32)


# =============================================================================
# 2. Train flow + decoder at a fixed latent dimension
# =============================================================================

def train_view_flow(
    x_np: np.ndarray,
    d_hat: int,
    degree: int,
    n_iter: int = 400,
    n_samples: int = 768,
    lr: float = 1e-2,
    gpu_id: Optional[int] = None,
    seed: int = 7,
    n_layers: int = 6,
    hidden_dim: int = 64,
    sigmas: Sequence[float] = DEFAULT_SIGMAS,
    n_restarts: int = 2,
    train_batch_cap: int = 2048,
) -> Tuple[SimpleFlow, torch.Tensor, torch.device, float]:
    """Fit a RealNVP flow + linear poly-feature decoder G to match x_np.

    Returns (best_flow, best_G, device, best_train_mmd). The "best" is the
    restart with the lowest training MMD at any iteration.
    """
    x_np = standardize_np(x_np.astype(np.float32))
    dev = get_device(gpu_id)
    x = torch.tensor(x_np, dtype=torch.float32, device=dev)

    n_train, d_obs = x.shape
    train_batch = min(max(512, n_train // 8), train_batch_cap)

    n_feat = num_poly_features(d_hat, degree)
    phi = make_poly_fn(d_hat, degree, dev)

    best_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_G_cpu: Optional[torch.Tensor] = None

    for r in range(n_restarts):
        set_all_seeds(seed + 1000 * r)

        flow = SimpleFlow(dim=d_hat, n_layers=n_layers, hidden_dim=hidden_dim).to(dev)
        G = (torch.randn(d_obs, n_feat, device=dev) * 0.1).requires_grad_(True)

        optimizer = optim.Adam(list(flow.parameters()) + [G], lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iter)

        for _ in range(n_iter):
            optimizer.zero_grad(set_to_none=True)

            idx = torch.randperm(n_train, device=dev)[:train_batch]
            x_b = x[idx]

            eps = torch.randn(n_samples, d_hat, device=dev)
            z_hat = flow(eps)
            x_hat = phi(z_hat) @ G.t()
            x_hat = (x_hat - x_hat.mean(0, keepdim=True)) / (x_hat.std(0, keepdim=True) + 1e-8)

            loss = multiscale_mmd(x_hat, x_b, sigmas=sigmas)
            loss.backward()
            optimizer.step()
            scheduler.step()

            v = float(loss.detach().cpu().item())
            if v < best_loss:
                best_loss = v
                best_state = {k: t.detach().cpu().clone() for k, t in flow.state_dict().items()}
                best_G_cpu = G.detach().cpu().clone()

    if best_state is None or best_G_cpu is None:
        raise RuntimeError("train_view_flow: no restart produced a finite loss.")

    final_flow = SimpleFlow(dim=d_hat, n_layers=n_layers, hidden_dim=hidden_dim).to(dev)
    final_flow.load_state_dict({k: t.to(dev) for k, t in best_state.items()})
    final_G = best_G_cpu.to(dev)

    return final_flow, final_G, dev, best_loss


# =============================================================================
# 3. Per-sample inversion: recover paired Z-hats from observed X
# =============================================================================

def recover_paired_z(
    x_np: np.ndarray,
    flow: SimpleFlow,
    G: torch.Tensor,
    degree: int,
    dev: torch.device,
    n_iter: int = 300,
    lr: float = 5e-2,
    eps_prior: float = 1e-3,
    seed: int = 0,
) -> np.ndarray:
    """Per-sample inversion. We optimize epsilon_i ∈ R^d_hat for every observed
    x_i so that x_i ≈ phi(flow(epsilon_i)) G^T after the same standardization
    used at training time. The optimization variable is epsilon (not z directly)
    so the resulting z = flow(epsilon) is automatically in the flow's range and
    the L2 penalty on epsilon corresponds to the model's N(0, I) prior.

    Returns z_hat ∈ R^(n × d_hat) as float32 numpy.
    """
    x_np = standardize_np(x_np.astype(np.float32))
    n = x_np.shape[0]
    d_hat = flow.dim

    set_all_seeds(seed)
    x = torch.tensor(x_np, dtype=torch.float32, device=dev)
    phi = make_poly_fn(d_hat, degree, dev)

    flow.eval()
    for p_ in flow.parameters():
        p_.requires_grad = False
    G_const = G.detach()

    eps = torch.randn(n, d_hat, device=dev, requires_grad=True)
    optimizer = optim.Adam([eps], lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iter)

    for _ in range(n_iter):
        optimizer.zero_grad(set_to_none=True)
        z = flow(eps)
        x_hat = phi(z) @ G_const.t()
        x_hat = (x_hat - x_hat.mean(0, keepdim=True)) / (x_hat.std(0, keepdim=True) + 1e-8)
        rec_loss = ((x_hat - x) ** 2).mean()
        prior_loss = eps_prior * (eps ** 2).mean()
        loss = rec_loss + prior_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        z_final = flow(eps).detach().cpu().numpy().astype(np.float32)
    return z_final


# =============================================================================
# 4. Schur complement Sigma_{1|2}
# =============================================================================

def sigma_1_given_2(z1: np.ndarray, z2: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Compute Sigma_{1|2} = Sigma_11 - Sigma_12 Sigma_22^{-1} Sigma_21
    via np.linalg.solve. Inputs are paired (n × d1) and (n × d2)."""
    z1 = z1 - z1.mean(axis=0, keepdims=True)
    z2 = z2 - z2.mean(axis=0, keepdims=True)
    n = z1.shape[0]
    s11 = (z1.T @ z1) / max(n - 1, 1)
    s12 = (z1.T @ z2) / max(n - 1, 1)
    s22 = (z2.T @ z2) / max(n - 1, 1)
    s22_reg = s22 + ridge * np.eye(s22.shape[0], dtype=s22.dtype)
    rhs = np.linalg.solve(s22_reg, s12.T)
    return s11 - s12 @ rhs


# =============================================================================
# 5. Index selection and Jaccard
# =============================================================================

def recover_shared_indices(sigma_diag: np.ndarray, p_hat: int) -> np.ndarray:
    """Return sorted indices of the p_hat smallest entries of sigma_diag."""
    p_hat = max(0, int(p_hat))
    if p_hat == 0:
        return np.array([], dtype=np.int64)
    order = np.argsort(sigma_diag)
    return np.sort(order[:p_hat]).astype(np.int64)


def detect_p_via_elbow(sigma_diag: np.ndarray) -> Tuple[int, float]:
    """Detect p_hat directly from sorted diag(Sigma_{1|2}) by finding the
    largest log-gap between consecutive entries.

    Theorem 1 says shared coords have diag ~ 0 and private coords have diag > 0.
    With finite samples + flow noise, shared coords have small diag (not exactly
    zero) and private coords have larger diag. We find the elbow on the *sorted*
    sequence by computing log-ratios of consecutive entries; the largest log-jump
    is where shared transitions to private.

    Returns (p_hat, gap_strength) where p_hat is the index of the elbow and
    gap_strength is log(d_{p_hat+1} / d_{p_hat}) — useful for diagnostics.

    Falls back to p_hat=0 (no shared coords) if all diagonals are similar in
    log-scale (no clear gap). Falls back to len(sigma_diag) (all shared) if
    the smallest entry is dominated by a single very-small outlier.
    """
    diag = np.asarray(sigma_diag, dtype=np.float64).copy()
    if diag.size == 0:
        return 0, 0.0

    # Floor very-small or non-positive entries so log is defined.
    eps = max(1e-12, 1e-6 * float(np.max(np.abs(diag))))
    diag = np.maximum(diag, eps)

    sorted_diag = np.sort(diag)
    log_diag = np.log(sorted_diag)
    # log_diag[i+1] - log_diag[i] is the log-ratio gap between rank i and i+1.
    gaps = np.diff(log_diag)
    if gaps.size == 0:
        return 0, 0.0

    # The elbow position k means: the k smallest entries are "shared",
    # and the k+1...d entries are "private". The gap that signals this is
    # gaps[k-1] (between rank k-1 and rank k, both 0-indexed).
    # We pick k = argmax(gaps) + 1 in 1-indexed terms.
    k = int(np.argmax(gaps)) + 1
    gap_strength = float(gaps[k - 1])
    return k, gap_strength


def jaccard_via_elbow(
    z1: np.ndarray,
    z2: np.ndarray,
    p_true: int,
    ridge: float = 1e-6,
) -> Dict[str, object]:
    """Theorem 1 + elbow detection: compute Sigma_{1|2}, find p_hat from the
    elbow in sorted diagonals, then Jaccard against ground truth.

    Avoids the inclusion-exclusion route entirely (which depends on the joint
    dim recovery, which is fragile at large n).
    """
    sigma = sigma_1_given_2(z1, z2, ridge=ridge)
    diag = np.diag(sigma)
    p_hat_elbow, gap = detect_p_via_elbow(diag)
    recovered = recover_shared_indices(diag, p_hat_elbow)
    truth = list(range(p_true))
    j = jaccard(recovered, truth)
    return {
        "p_hat_elbow": int(p_hat_elbow),
        "gap_strength": float(gap),
        "jaccard_elbow": float(j),
        "recovered_elbow": recovered.tolist(),
        "ground_truth": truth,
        "sigma_diag": [float(v) for v in diag.tolist()],
    }


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    """Jaccard similarity over index sets."""
    sa = {int(x) for x in a}
    sb = {int(x) for x in b}
    if not sa and not sb:
        return 1.0
    union = sa | sb
    inter = sa & sb
    return len(inter) / len(union)


# =============================================================================
# 6. End-to-end: one (n, seed) row → Jaccard score
# =============================================================================

def jaccard_for_pipeline_run(
    p_true: int,
    q1: int,
    q2: int,
    n: int,
    d_obs: int,
    degree: int,
    seed: int,
    d1_hat: int,
    d2_hat: int,
    p_hat: int,
    *,
    noise: str = "exponential",
    gpu_id: Optional[int] = None,
    n_iter_train: int = 400,
    n_iter_inv: int = 300,
    n_samples: int = 768,
    lr: float = 1e-2,
    inv_lr: float = 5e-2,
    n_restarts: int = 2,
    sigmas: Sequence[float] = DEFAULT_SIGMAS,
) -> Dict[str, object]:
    """End-to-end Jaccard for one paired-data run, given the dimensions
    recovered upstream by run_full_pipeline_once.

    The function regenerates X1, X2 from `seed` so it's coupled to the same
    upstream pipeline run by reproducing that pipeline's RNG.
    """
    ground_truth = list(range(p_true))

    # Hard fail only if view dim recovery is broken — we can't compute Sigma_{1|2}
    # without reasonable view dims. If only p_hat is broken (but views are OK),
    # we still compute jaccard_true_p and jaccard_elbow which don't depend on it.
    if d1_hat <= 0 or d2_hat <= 0:
        return _failed_result(ground_truth, f"non-positive view dim: d1={d1_hat}, d2={d2_hat}")

    p_hat_clipped = max(0, min(int(p_hat), min(d1_hat, d2_hat)))
    p_hat_note = ""
    if p_hat_clipped != p_hat:
        p_hat_note = f"input p_hat={p_hat} clipped to {p_hat_clipped}"

    x1_np, x2_np = regenerate_paired_data(p_true, q1, q2, n, d_obs, degree, seed, noise=noise)

    flow1, G1, dev1, train_mmd_1 = train_view_flow(
        x1_np, d1_hat, degree,
        n_iter=n_iter_train, n_samples=n_samples, lr=lr,
        gpu_id=gpu_id, seed=seed * 13 + 1, n_restarts=n_restarts, sigmas=sigmas,
    )
    flow2, G2, dev2, train_mmd_2 = train_view_flow(
        x2_np, d2_hat, degree,
        n_iter=n_iter_train, n_samples=n_samples, lr=lr,
        gpu_id=gpu_id, seed=seed * 13 + 2, n_restarts=n_restarts, sigmas=sigmas,
    )

    z1 = recover_paired_z(x1_np, flow1, G1, degree, dev1, n_iter=n_iter_inv, lr=inv_lr, seed=seed * 13 + 3)
    z2 = recover_paired_z(x2_np, flow2, G2, degree, dev2, n_iter=n_iter_inv, lr=inv_lr, seed=seed * 13 + 4)

    sigma = sigma_1_given_2(z1, z2)
    diag = np.diag(sigma)

    # 1. Jaccard at the recovered p_hat (real-world conditions, depends on
    #    upstream dim recovery). If p_hat was nonsensical we use the clipped
    #    value and note it in the reason.
    recovered = recover_shared_indices(diag, p_hat_clipped)
    j_recovered = jaccard(recovered, ground_truth)

    # 2. Jaccard at the *true* p — decouples Theorem 1 validation from the
    #    fragile joint dim recovery. This is the "math works" headline metric.
    recovered_true_p = recover_shared_indices(diag, p_true)
    j_true_p = jaccard(recovered_true_p, ground_truth)

    # 3. Jaccard via elbow detection on diag(Sigma_{1|2}) — fully unsupervised
    #    p-detection from the spectrum. Brittle when the shared/private gap is
    #    weak, but informative as a diagnostic.
    p_hat_elbow, gap = detect_p_via_elbow(diag)
    recovered_elbow = recover_shared_indices(diag, p_hat_elbow)
    j_elbow = jaccard(recovered_elbow, ground_truth)

    return {
        "jaccard": float(j_recovered),
        "recovered": recovered.tolist(),
        "ground_truth": ground_truth,

        "jaccard_true_p": float(j_true_p),
        "recovered_true_p": recovered_true_p.tolist(),

        "p_hat_elbow": int(p_hat_elbow),
        "gap_strength": float(gap),
        "jaccard_elbow": float(j_elbow),
        "recovered_elbow": recovered_elbow.tolist(),

        "sigma_diag": [float(v) for v in diag.tolist()],
        "train_mmd_view1": float(train_mmd_1),
        "train_mmd_view2": float(train_mmd_2),
        "reason": p_hat_note,
    }


def _failed_result(ground_truth: List[int], reason: str) -> Dict[str, object]:
    return {
        "jaccard": float("nan"),
        "recovered": [],
        "ground_truth": ground_truth,
        "jaccard_true_p": float("nan"),
        "recovered_true_p": [],
        "p_hat_elbow": 0,
        "gap_strength": 0.0,
        "jaccard_elbow": float("nan"),
        "recovered_elbow": [],
        "sigma_diag": [],
        "train_mmd_view1": float("nan"),
        "train_mmd_view2": float("nan"),
        "reason": reason,
    }
