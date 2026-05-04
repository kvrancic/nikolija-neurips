"""
Fixed 4-GPU sanity checks for CRL dimension recovery.

Main fixes compared with the original sanity script:
  1. Uses multi-scale MMD instead of a single bandwidth.
  2. Selects the MINIMAL dimension that reaches the real-vs-real MMD noise floor.
  3. Uses separate slack values for marginal/view recovery and joint recovery.
  4. Aggregates restarts by median instead of mean.
  5. Avoids selecting a dimension that only barely passes threshold if the next
     dimension gives a large improvement.
  3. Does not clip p_hat with max(1, ...), so inclusion-exclusion failures are visible.
  4. Runs dimension-recovery restarts in parallel across up to 4 GPUs.
  5. Adds a full-pipeline multi-seed consistency check.

Example:
  python sanity_checks_param_v6.py

Optional:
  python sanity_checks_param_v6.py --gpus 0,1,2,3 --n-restarts 4 --n-iter 400
"""

import argparse
import concurrent.futures as futures
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# =============================================================================
# Utilities
# =============================================================================

def parse_gpu_ids(gpus: str) -> List[Optional[int]]:
    """
    Parses --gpus argument.
    Use --gpus cpu to force CPU.
    Use --gpus 0,1,2,3 to use four GPUs.
    """
    if gpus.lower() == "cpu":
        return [None]

    if not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.")
        return [None]

    visible = torch.cuda.device_count()
    requested = [int(x.strip()) for x in gpus.split(",") if x.strip() != ""]
    valid = [g for g in requested if 0 <= g < visible]

    if len(valid) == 0:
        print(f"No valid requested GPUs among {requested}; falling back to GPU 0.")
        return [0]

    return valid


def set_all_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def standardize_np(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + eps)


def get_device(gpu_id: Optional[int]) -> torch.device:
    if gpu_id is None or not torch.cuda.is_available():
        return torch.device("cpu")
    torch.cuda.set_device(gpu_id)
    return torch.device(f"cuda:{gpu_id}")


# =============================================================================
# Data generation
# =============================================================================

def sample_er_dag(p: int, q1: int, q2: int, er_prob: float, rng: np.random.RandomState) -> np.ndarray:
    """
    Samples a DAG with ordering 0,1,...,h-1.
    The first p nodes are shared, the next q1 are domain-specific for view 1,
    and the last q2 are domain-specific for view 2.

    We force each shared latent to have at least one child among the private nodes,
    which makes the shared structure nontrivial in small sanity checks.
    """
    h = p + q1 + q2
    W = np.zeros((h, h), dtype=np.float32)

    I1 = list(range(p, p + q1))
    I2 = list(range(p + q1, h))
    L = list(range(p))

    children_pool = I1 + I2
    rng.shuffle(children_pool)

    # Force simple signal from shared latents into private blocks when possible.
    # This assumes q1 + q2 >= 2p in the sanity settings.
    for idx, k in enumerate(L):
        if 2 * idx + 1 < len(children_pool):
            c1, c2 = children_pool[2 * idx], children_pool[2 * idx + 1]
            for c in (c1, c2):
                W[k, c] = rng.choice([-1.0, 1.0]) * rng.uniform(0.25, 1.0)

    # Add random DAG edges, avoiding direct cross-private edges.
    for i in range(h):
        for j in range(i + 1, h):
            if (i in I1 and j in I2) or (i in I2 and j in I1):
                continue
            if W[i, j] == 0 and rng.rand() < er_prob:
                W[i, j] = rng.choice([-1.0, 1.0]) * rng.uniform(0.25, 1.0)

    return W


def generate_scm(W: np.ndarray, n: int, rng: np.random.RandomState, noise: str = "exponential") -> np.ndarray:
    """
    Generates samples from a linear SCM on a DAG:
        Z_j = sum_{i<j} W_ij Z_i + eps_j.
    """
    h = W.shape[0]
    Z = np.zeros((n, h), dtype=np.float32)

    noise_types = ["gaussian", "laplace", "uniform", "mixture", "exponential"]

    for j in range(h):
        if noise == "random":
            node_noise = noise_types[rng.randint(len(noise_types))]
        else:
            node_noise = noise

        if node_noise == "gaussian":
            eps = rng.randn(n)
        elif node_noise == "laplace":
            eps = rng.laplace(0, 1 / np.sqrt(2), size=n)
        elif node_noise == "uniform":
            eps = rng.uniform(-np.sqrt(3), np.sqrt(3), size=n)
        elif node_noise == "mixture":
            idx = rng.randint(0, 2, size=n)
            eps = rng.randn(n) + idx * 2.0 - 1.0
        elif node_noise == "exponential":
            eps = rng.exponential(1.0, size=n) - 1.0
        else:
            raise ValueError(f"Unknown noise: {node_noise}")

        Z[:, j] = eps.astype(np.float32)
        if j > 0:
            Z[:, j] += Z[:, :j] @ W[:j, j]

    return Z


# =============================================================================
# Polynomial features and MMD
# =============================================================================

def make_poly_fn(dim: int, degree: int, dev: torch.device):
    """
    Polynomial feature map with linear and quadratic terms.
    No constant term, matching the original sanity script.
    """
    idx2 = torch.triu_indices(dim, dim, offset=0, device=dev) if degree >= 2 else None

    def phi(Z: torch.Tensor) -> torch.Tensor:
        features = [Z]
        if degree >= 2:
            outer = torch.bmm(Z.unsqueeze(2), Z.unsqueeze(1))
            features.append(outer[:, idx2[0], idx2[1]])
        return torch.cat(features, dim=1)

    return phi


def num_poly_features(dim: int, degree: int) -> int:
    if degree == 1:
        return dim
    if degree == 2:
        return dim + dim * (dim + 1) // 2
    raise NotImplementedError("This sanity script currently supports degree 1 or 2.")


def rbf_mmd(X: torch.Tensor, Y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Biased RBF MMD. Stable enough for sanity checks.
    """
    XX = torch.cdist(X, X).pow(2)
    YY = torch.cdist(Y, Y).pow(2)
    XY = torch.cdist(X, Y).pow(2)

    Kxx = torch.exp(-XX / (2 * sigma ** 2)).mean()
    Kyy = torch.exp(-YY / (2 * sigma ** 2)).mean()
    Kxy = torch.exp(-XY / (2 * sigma ** 2)).mean()

    return Kxx + Kyy - 2.0 * Kxy


def multiscale_mmd(
    X: torch.Tensor,
    Y: torch.Tensor,
    sigmas: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
) -> torch.Tensor:
    return sum(rbf_mmd(X, Y, sigma=s) for s in sigmas) / len(sigmas)


def estimate_real_mmd_floor(
    X_val_np: np.ndarray,
    gpu_id: Optional[int],
    sigmas: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
    n_repeats: int = 5,
    seed: int = 0,
) -> float:
    """
    Estimates the finite-sample MMD noise floor by comparing two real validation
    subsamples. This is the target level: a generated sample should not need to
    be much closer to validation data than another real validation sample is.

    This helps prevent over-selecting dimensions just because larger latent
    models can reduce MMD slightly.
    """
    rng = np.random.RandomState(seed)
    n = X_val_np.shape[0]
    if n < 4:
        return 0.0

    dev = get_device(gpu_id)
    vals = []

    for _ in range(n_repeats):
        perm = rng.permutation(n)
        half = n // 2
        A_np = X_val_np[perm[:half]]
        B_np = X_val_np[perm[half:2 * half]]

        A = torch.tensor(A_np, dtype=torch.float32, device=dev)
        B = torch.tensor(B_np, dtype=torch.float32, device=dev)

        with torch.no_grad():
            vals.append(float(multiscale_mmd(A, B, sigmas=sigmas).cpu().item()))

    if dev.type == "cuda":
        torch.cuda.empty_cache()

    return float(np.mean(vals))


# =============================================================================
# Flow model
# =============================================================================

class AffineCouplingLayer(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 64):
        super().__init__()
        self.d1 = dim // 2
        self.d2 = dim - self.d1

        self.net = nn.Sequential(
            nn.Linear(self.d1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.d2 * 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z1, z2 = z[:, : self.d1], z[:, self.d1 :]
        st = self.net(z1)
        s, t = st[:, : self.d2], st[:, self.d2 :]
        # tanh keeps scaling numerically stable.
        return torch.cat([z1, z2 * torch.exp(torch.tanh(s)) + t], dim=1)


class SimpleFlow(nn.Module):
    """
    Small RealNVP-style flow for sanity checks.

    Note: for dim=1, this uses a flexible neural map, not a guaranteed invertible flow.
    That is okay for the sanity check, because dim=1 is only used as an underfit candidate.
    """
    def __init__(self, dim: int, n_layers: int = 6, hidden_dim: int = 64):
        super().__init__()
        self.dim = dim

        if dim == 1:
            self.is_1d = True
            self.net = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.is_1d = False
            self.layers = nn.ModuleList(
                [AffineCouplingLayer(dim, hidden_dim=hidden_dim) for _ in range(n_layers)]
            )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.is_1d:
            return self.net(z)

        x = z
        for i, layer in enumerate(self.layers):
            if i % 2 == 1:
                x = torch.cat([x[:, self.dim // 2 :], x[:, : self.dim // 2]], dim=1)
            x = layer(x)
        return x


# =============================================================================
# Observation generation
# =============================================================================

def get_obs_and_G(
    zl: np.ndarray,
    zi: np.ndarray,
    d_out: int,
    degree: int,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates X = phi([zl, zi]) G^T with random full-rank G.
    Uses CPU torch for simple reproducibility.
    """
    dev = torch.device("cpu")
    z_comb = np.hstack([zl, zi]).astype(np.float32)
    dim = z_comb.shape[1]

    phi_gen = make_poly_fn(dim, degree, dev)
    z_t = torch.tensor(z_comb, dtype=torch.float32, device=dev)
    phi_val = phi_gen(z_t)
    n_phi = phi_val.shape[1]

    while True:
        G = rng.randn(d_out, n_phi).astype(np.float32)
        if np.linalg.matrix_rank(G) == min(d_out, n_phi):
            break

    G_t = torch.tensor(G, dtype=torch.float32, device=dev)
    X = (phi_val @ G_t.t()).detach().cpu().numpy()

    return X.astype(np.float32), G


# =============================================================================
# Parallel training workers
# =============================================================================

@dataclass
class RestartResult:
    penalized_val: float
    raw_val: float
    best_train: float
    full_rank: bool
    rank_G: int
    gpu_id: Optional[int]
    seed: int


def _train_one_restart_worker(args) -> RestartResult:
    """
    One restart for one candidate latent dimension.
    This function is designed to run inside a separate process.
    """
    (
        p_hat,
        degree,
        d_obs,
        X_train_np,
        X_val_np,
        n_samples,
        lr,
        n_iter,
        gpu_id,
        seed,
        sigmas,
        hidden_dim,
        n_layers,
    ) = args

    set_all_seeds(seed)
    dev = get_device(gpu_id)

    X_train = torch.tensor(X_train_np, dtype=torch.float32, device=dev)
    X_val = torch.tensor(X_val_np, dtype=torch.float32, device=dev)

    n_train = X_train.shape[0]
    train_batch = min(max(512, n_train // 8), 2048)

    phi = make_poly_fn(p_hat, degree, dev)
    n_feat = num_poly_features(p_hat, degree)

    flow = SimpleFlow(dim=p_hat, n_layers=n_layers, hidden_dim=hidden_dim).to(dev)
    G = (torch.randn(d_obs, n_feat, device=dev) * 0.1).requires_grad_(True)

    optimizer = optim.Adam(list(flow.parameters()) + [G], lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iter)

    best_train = float("inf")
    best_state = None
    best_G = None

    for _ in range(n_iter):
        optimizer.zero_grad(set_to_none=True)

        idx = torch.randperm(n_train, device=dev)[:train_batch]
        X_batch = X_train[idx]

        eps = torch.randn(n_samples, p_hat, device=dev)
        Z_hat = flow(eps)
        X_hat = phi(Z_hat) @ G.t()
        X_hat = (X_hat - X_hat.mean(0, keepdim=True)) / (X_hat.std(0, keepdim=True) + 1e-8)

        loss = multiscale_mmd(X_hat, X_batch, sigmas=sigmas)
        loss.backward()
        optimizer.step()
        scheduler.step()

        val = float(loss.detach().cpu().item())
        if val < best_train:
            best_train = val
            best_state = {k: v.detach().cpu().clone() for k, v in flow.state_dict().items()}
            best_G = G.detach().cpu().clone()

    # Validation using best train checkpoint.
    flow.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
    best_G = best_G.to(dev)

    with torch.no_grad():
        n_val = X_val.shape[0]
        eps_val = torch.randn(n_val, p_hat, device=dev)
        X_hat_val = phi(flow(eps_val)) @ best_G.t()
        X_hat_val = (X_hat_val - X_hat_val.mean(0, keepdim=True)) / (
            X_hat_val.std(0, keepdim=True) + 1e-8
        )

        val_mmd = float(multiscale_mmd(X_hat_val, X_val, sigmas=sigmas).cpu().item())
        rank_G = int(torch.linalg.matrix_rank(best_G).cpu().item())
        full_rank = rank_G == min(d_obs, n_feat)

    penalized = val_mmd if full_rank else val_mmd + 1e6

    if dev.type == "cuda":
        torch.cuda.empty_cache()

    return RestartResult(
        penalized_val=penalized,
        raw_val=val_mmd,
        best_train=best_train,
        full_rank=full_rank,
        rank_G=rank_G,
        gpu_id=gpu_id,
        seed=seed,
    )


def select_minimal_dimension(
    dims: Sequence[int],
    scores: Sequence[float],
    threshold: Optional[float] = None,
    abs_tol: float = 2e-3,
    rel_tol: float = 0.05,
    next_improvement_abs: float = 2e-3,
    next_improvement_rel: float = 0.10,
) -> Tuple[int, str]:
    """
    Selects the smallest dimension that fits sufficiently well.

    Priority:
      1. If a threshold is provided and any dim is below it, choose the smallest
         such dim. In this script the default threshold is the real-vs-real MMD
         floor plus a small slack.
      2. Otherwise choose the smallest dim within tolerance of the best score.

    This is closer to the theory, which identifies the minimal latent dimension.
    """
    dims = list(dims)
    scores = list(scores)

    if len(dims) == 0:
        raise RuntimeError("No candidate dimensions were evaluated.")

    if threshold is not None:
        for i, (d, s) in enumerate(zip(dims, scores)):
            if s <= threshold:
                # Guard against accidental under-selection:
                # if the next dimension improves a lot, this dimension only barely
                # passed due to finite-sample/optimisation noise, so continue.
                if i + 1 < len(scores):
                    improvement = s - scores[i + 1]
                    required = max(next_improvement_abs, next_improvement_rel * abs(s))
                    if improvement > required:
                        continue
                return d, (
                    f"minimal stable dim with score <= threshold {threshold:g}; "
                    f"next-improvement guard abs={next_improvement_abs:g}, rel={next_improvement_rel:g}"
                )

    best_score = min(scores)
    cutoff = best_score + max(abs_tol, rel_tol * abs(best_score))
    eligible = [d for d, s in zip(dims, scores) if s <= cutoff]

    return min(eligible), f"minimal dim within cutoff {cutoff:.6f} of best score {best_score:.6f}"


def recover_dimension_flow_parallel(
    X_np: np.ndarray,
    degree: int,
    max_dim: int,
    n_samples: int,
    lr: float,
    n_iter: int,
    gpu_ids: Sequence[Optional[int]],
    val_fraction: float = 0.2,
    n_restarts: int = 4,
    sigmas: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
    selection_threshold: Optional[float] = None,
    selection_abs_tol: float = 2e-3,
    selection_rel_tol: float = 0.05,
    selection_floor_slack: float = 5e-3,
    next_improvement_abs: float = 2e-3,
    next_improvement_rel: float = 0.10,
    hidden_dim: int = 64,
    n_layers: int = 6,
    base_seed: int = 123,
    label: str = "",
    max_val: int = 4096,
) -> Tuple[int, List[int], List[float]]:
    """
    Recovers the minimal latent dimension by scanning candidate dimensions.

    Restarts for each dimension are distributed across available GPUs.
    """
    X_np = standardize_np(X_np.astype(np.float32))
    n, d_obs = X_np.shape

    # Sweep-safe validation split:
    # - for small n, keep both train and validation non-empty;
    # - for huge n, cap validation because MMD is quadratic in validation size.
    if n < 4:
        raise ValueError(f"Need at least 4 samples, got n={n}.")
    desired_val = max(16, int(n * val_fraction))
    n_val = min(max_val, desired_val, max(1, n - 16))
    if n <= 64:
        n_val = max(1, n // 2)

    rng = np.random.RandomState(base_seed)
    idx = rng.permutation(n)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    X_val_np = X_np[val_idx]
    X_train_np = X_np[train_idx]

    dims: List[int] = []
    avg_scores: List[float] = []

    max_workers = max(1, min(len(gpu_ids), n_restarts))
    ctx = mp.get_context("spawn")

    # Data-dependent selection threshold.
    # If the user did not pass an explicit --threshold, estimate the finite-sample
    # real-vs-real MMD floor on validation data and add a small slack.
    effective_threshold = selection_threshold
    data_floor = None
    if effective_threshold is None:
        data_floor = estimate_real_mmd_floor(
            X_val_np,
            gpu_id=gpu_ids[0],
            sigmas=sigmas,
            n_repeats=5,
            seed=base_seed + 999,
        )
        effective_threshold = data_floor + selection_floor_slack

    print(f"\n[{label}] recover_dimension_flow_parallel")
    print(f"  n={n}, d_obs={d_obs}, degree={degree}, restarts={n_restarts}, workers={max_workers}")
    print(f"  GPUs used for restarts: {gpu_ids}")
    if data_floor is not None:
        print(f"  real-vs-real MMD floor={data_floor:.5f}; selection threshold={effective_threshold:.5f}")
    else:
        print(f"  user selection threshold={effective_threshold:.5f}")

    # Create a fresh pool per recovery call. This is simple and robust for CUDA.
    with futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        for p_hat in range(1, max_dim + 1):
            n_feat = num_poly_features(p_hat, degree)

            # If the feature dimension already exceeds the observed dimension,
            # the full-column-rank decoder condition cannot hold.
            if n_feat > d_obs:
                print(f"    stopping at dim={p_hat}: n_feat={n_feat} > d_obs={d_obs}")
                break

            tasks = []
            for r in range(n_restarts):
                gpu_id = gpu_ids[r % len(gpu_ids)]
                seed = base_seed + 10000 * p_hat + r
                tasks.append(
                    (
                        p_hat,
                        degree,
                        d_obs,
                        X_train_np,
                        X_val_np,
                        n_samples,
                        lr,
                        n_iter,
                        gpu_id,
                        seed,
                        tuple(sigmas),
                        hidden_dim,
                        n_layers,
                    )
                )

            results = list(executor.map(_train_one_restart_worker, tasks))

            penalized = [res.penalized_val for res in results]
            raw = [res.raw_val for res in results]
            ranks = [res.rank_G for res in results]
            oks = [res.full_rank for res in results]
            used = [res.gpu_id for res in results]

            avg = float(np.median(penalized))
            dims.append(p_hat)
            avg_scores.append(avg)

            raw_str = ", ".join(f"{v:.5f}" for v in raw)
            ok_str = "".join("✓" if x else "✗" for x in oks)
            gpu_str = ",".join("cpu" if g is None else str(g) for g in used)

            print(
                f"    dim={p_hat:<2d} n_feat={n_feat:<3d} med_pen={avg:.5f} "
                f"raw=[{raw_str}] rank={ranks} ok={ok_str} gpu=[{gpu_str}]"
            )

    selected_dim, reason = select_minimal_dimension(
        dims,
        avg_scores,
        threshold=effective_threshold,
        abs_tol=selection_abs_tol,
        rel_tol=selection_rel_tol,
        next_improvement_abs=next_improvement_abs,
        next_improvement_rel=next_improvement_rel,
    )

    print(f"    → selected dim={selected_dim} ({reason})")
    return selected_dim, dims, avg_scores


# =============================================================================
# Sanity checks
# =============================================================================

def get_max_dim_view(args) -> int:
    if args.max_dim_view is not None:
        return args.max_dim_view
    return max(args.p_true + args.q1, args.p_true + args.q2) + args.extra_dim_buffer


def get_max_dim_joint(args) -> int:
    if args.max_dim_joint is not None:
        return args.max_dim_joint
    return args.p_true + args.q1 + args.q2 + args.extra_dim_buffer


def print_dimension_targets(args, prefix: str = "") -> None:
    d1_true = args.p_true + args.q1
    d2_true = args.p_true + args.q2
    dj_true = args.p_true + args.q1 + args.q2
    print(f"{prefix}Expected: d1={d1_true}, d2={d2_true}, dj={dj_true}, p_hat={args.p_true}")
    print(f"{prefix}Using d_obs={args.d_obs}; joint observed dimension={2 * args.d_obs}")
    print(f"{prefix}max_dim_view={get_max_dim_view(args)}, max_dim_joint={get_max_dim_joint(args)}")


def check1_flow_exponential(
    gpu_id: Optional[int],
    n: int = 1024,
    dim: int = 3,
    n_iter: int = 800,
    seed: int = 42,
) -> bool:
    """
    Checks that the flow can fit standardized exponential noise.
    Runs on one GPU; dimension recovery itself uses all GPUs.
    """
    print("\n" + "=" * 70)
    print("CHECK 1: Flow fits exponential noise")
    print(f"dim={dim}, n={n}, n_iter={n_iter}")
    print("=" * 70)

    set_all_seeds(seed)
    dev = get_device(gpu_id)
    rng = np.random.RandomState(seed)

    Z_np = rng.exponential(1.0, size=(n, dim)).astype(np.float32) - 1.0
    X_real = torch.tensor(standardize_np(Z_np), dtype=torch.float32, device=dev)

    flow = SimpleFlow(dim=dim, n_layers=6, hidden_dim=64).to(dev)
    opt = optim.Adam(flow.parameters(), lr=1e-2)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter)

    losses = []
    for it in range(n_iter):
        opt.zero_grad(set_to_none=True)
        eps = torch.randn(n, dim, device=dev)
        loss = multiscale_mmd(flow(eps), X_real, sigmas=(0.5, 1.0, 2.0, 4.0))
        loss.backward()
        opt.step()
        sch.step()

        losses.append(float(loss.detach().cpu().item()))
        if (it + 1) % 200 == 0:
            print(f"  iter={it+1:<4d} MMD={losses[-1]:.5f}")

    final = losses[-1]
    print(f"Final MMD: {final:.5f}")

    # Multi-scale MMD has a different scale than single-bandwidth MMD.
    passed = final < 0.03
    print("✓ PASS" if passed else "✗ FAIL")
    return passed


def check2_dim_recovery(args) -> Tuple[bool, List[int], List[float], int]:
    print("\n" + "=" * 70)
    print("CHECK 2: Single-view dimension recovery")
    print(f"Expected: selected dim={args.p_true + args.q1}")
    print("=" * 70)

    rng = np.random.RandomState(42)
    r_true = args.p_true + args.q1
    d_obs = args.d_obs
    n = args.n

    Z_true = rng.exponential(1.0, size=(n, r_true)).astype(np.float32) - 1.0
    X_true, _ = get_obs_and_G(Z_true[:, :args.p_true], Z_true[:, args.p_true:], d_obs, args.degree, rng)

    d_hat, dims, scores = recover_dimension_flow_parallel(
        X_true,
        degree=args.degree,
        max_dim=get_max_dim_view(args),
        n_samples=args.n_samples,
        lr=args.lr,
        n_iter=args.n_iter,
        gpu_ids=args.gpu_ids,
        n_restarts=args.n_restarts,
        selection_threshold=args.threshold,
        selection_abs_tol=args.selection_abs_tol,
        selection_rel_tol=args.selection_rel_tol,
        selection_floor_slack=args.view_floor_slack,
        next_improvement_abs=args.next_improvement_abs,
        next_improvement_rel=args.next_improvement_rel,
        base_seed=202,
        label="check2/view",
        max_val=args.max_val,
    )

    passed = d_hat == r_true
    print(f"\nTrue dim={r_true}, recovered={d_hat}")
    print("✓ PASS" if passed else f"✗ FAIL (got {d_hat})")

    return passed, dims, scores, d_hat


def run_full_pipeline_once(args, seed: int, label: str = "full") -> Tuple[bool, int, int, int, int]:
    rng = np.random.RandomState(seed)

    p_true = args.p_true
    q1, q2 = args.q1, args.q2
    d_obs = args.d_obs
    n = args.n

    W = sample_er_dag(p_true, q1, q2, 0.4, rng)
    Z = generate_scm(W, n, rng, noise="exponential")

    ZL = Z[:, :p_true]
    ZI1 = Z[:, p_true : p_true + q1]
    ZI2 = Z[:, p_true + q1 :]

    X1, _ = get_obs_and_G(ZL, ZI1, d_obs, args.degree, rng)
    X2, _ = get_obs_and_G(ZL, ZI2, d_obs, args.degree, rng)
    X_joint = np.hstack([X1, X2]).astype(np.float32)

    print(f"\n[{label}] Fitting view 1 (true={p_true + q1})")
    d1_hat, _, _ = recover_dimension_flow_parallel(
        X1,
        degree=args.degree,
        max_dim=get_max_dim_view(args),
        n_samples=args.n_samples,
        lr=args.lr,
        n_iter=args.n_iter,
        gpu_ids=args.gpu_ids,
        n_restarts=args.n_restarts,
        selection_threshold=args.threshold,
        selection_abs_tol=args.selection_abs_tol,
        selection_rel_tol=args.selection_rel_tol,
        selection_floor_slack=args.view_floor_slack,
        next_improvement_abs=args.next_improvement_abs,
        next_improvement_rel=args.next_improvement_rel,
        base_seed=1000 + seed * 10 + 1,
        label=f"{label}/view1",
        max_val=args.max_val,
    )

    print(f"\n[{label}] Fitting view 2 (true={p_true + q2})")
    d2_hat, _, _ = recover_dimension_flow_parallel(
        X2,
        degree=args.degree,
        max_dim=get_max_dim_view(args),
        n_samples=args.n_samples,
        lr=args.lr,
        n_iter=args.n_iter,
        gpu_ids=args.gpu_ids,
        n_restarts=args.n_restarts,
        selection_threshold=args.threshold,
        selection_abs_tol=args.selection_abs_tol,
        selection_rel_tol=args.selection_rel_tol,
        selection_floor_slack=args.view_floor_slack,
        next_improvement_abs=args.next_improvement_abs,
        next_improvement_rel=args.next_improvement_rel,
        base_seed=1000 + seed * 10 + 2,
        label=f"{label}/view2",
        max_val=args.max_val,
    )

    print(f"\n[{label}] Fitting joint (true={p_true + q1 + q2})")
    dj_hat, _, _ = recover_dimension_flow_parallel(
        X_joint,
        degree=args.degree,
        max_dim=get_max_dim_joint(args),
        n_samples=args.n_samples_joint,
        lr=args.lr,
        n_iter=args.n_iter,
        gpu_ids=args.gpu_ids,
        n_restarts=args.n_restarts,
        selection_threshold=args.threshold,
        selection_abs_tol=args.selection_abs_tol,
        selection_rel_tol=args.selection_rel_tol,
        selection_floor_slack=args.joint_floor_slack,
        next_improvement_abs=args.next_improvement_abs,
        next_improvement_rel=args.next_improvement_rel,
        base_seed=1000 + seed * 10 + 3,
        label=f"{label}/joint",
        max_val=args.max_val,
    )

    # Important: no max(1, ...). We want to see failures.
    p_hat = d1_hat + d2_hat - dj_hat
    passed = p_hat == p_true

    print(f"\n[{label}] p_hat = {d1_hat} + {d2_hat} - {dj_hat} = {p_hat}")
    print(f"[{label}] true p={p_true}")
    print("✓ PASS" if passed else "✗ FAIL")

    return passed, d1_hat, d2_hat, dj_hat, p_hat


def check3_full_pipeline(args) -> Tuple[bool, int, int, int, int]:
    print("\n" + "=" * 70)
    print("CHECK 3: Full pipeline inclusion-exclusion")
    print_dimension_targets(args)
    print("=" * 70)

    return run_full_pipeline_once(args, seed=42, label="check3")


def check4_single_view_consistency(args) -> float:
    print("\n" + "=" * 70)
    print(f"CHECK 4: Single-view consistency over {args.n_seeds} seeds")
    print(f"Expected: recovered dim={args.p_true + args.q1}")
    print("=" * 70)

    r_true = args.p_true + args.q1
    d_obs = args.d_obs
    n = args.n

    results = []
    for seed in range(args.n_seeds):
        rng = np.random.RandomState(seed)
        Z_true = rng.exponential(1.0, size=(n, r_true)).astype(np.float32) - 1.0
        X_true, _ = get_obs_and_G(Z_true[:, :args.p_true], Z_true[:, args.p_true:], d_obs, args.degree, rng)

        d_hat, _, _ = recover_dimension_flow_parallel(
            X_true,
            degree=args.degree,
            max_dim=get_max_dim_view(args),
            n_samples=args.n_samples,
            lr=args.lr,
            n_iter=args.n_iter,
            gpu_ids=args.gpu_ids,
            n_restarts=args.n_restarts,
            selection_threshold=args.threshold,
            selection_abs_tol=args.selection_abs_tol,
            selection_rel_tol=args.selection_rel_tol,
            base_seed=3000 + seed,
            label=f"check4/seed{seed}",
            max_val=args.max_val,
        )

        ok = d_hat == r_true
        results.append(ok)
        print(f"  seed={seed:<2d} recovered={d_hat:<2d} {'✓' if ok else '✗'}")

    acc = float(np.mean(results))
    print(f"\nSingle-view accuracy: {acc:.1%} ({sum(results)}/{len(results)})")
    print("✓ PASS" if acc >= 0.8 else "✗ FAIL")
    return acc


def check5_full_pipeline_consistency(args) -> float:
    print("\n" + "=" * 70)
    print(f"CHECK 5: Full-pipeline consistency over {args.n_full_seeds} seeds")
    print_dimension_targets(args)
    print("=" * 70)

    results = []

    for seed in range(args.n_full_seeds):
        passed, d1, d2, dj, ph = run_full_pipeline_once(args, seed=500 + seed, label=f"check5/seed{seed}")
        results.append(passed)
        print(f"  seed={seed:<2d} d1={d1:<2d} d2={d2:<2d} dj={dj:<2d} p_hat={ph:<2d} {'✓' if passed else '✗'}")

    acc = float(np.mean(results))
    print(f"\nFull-pipeline accuracy: {acc:.1%} ({sum(results)}/{len(results)})")
    print("✓ PASS" if acc >= 0.8 else "✗ FAIL")
    return acc


# =============================================================================
# Main
# =============================================================================

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpus", type=str, default="0,1,2,3",
                        help="GPU ids to use, e.g. 0,1,2,3. Use 'cpu' for CPU.")
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--p-true", type=int, default=2,
                        help="True dimension of the shared latent space.")
    parser.add_argument("--q1", type=int, default=2,
                        help="True private dimension for view/domain 1.")
    parser.add_argument("--q2", type=int, default=2,
                        help="True private dimension for view/domain 2.")
    parser.add_argument("--d-obs", type=int, default=30,
                        help="Observed dimension per view. For degree 2, this should be at least r + r(r+1)/2 for the largest view latent dimension.")
    parser.add_argument("--max-dim-view", type=int, default=None,
                        help="Maximum candidate dimension for marginal/view recovery. Default: true view dimension + buffer.")
    parser.add_argument("--max-dim-joint", type=int, default=None,
                        help="Maximum candidate dimension for joint recovery. Default: true joint dimension + buffer.")
    parser.add_argument("--extra-dim-buffer", type=int, default=3,
                        help="Extra candidate dimensions above the true dimension when max dims are not explicitly provided.")
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--n-samples-joint", type=int, default=400)
    parser.add_argument("--n-iter", type=int, default=400)
    parser.add_argument("--n-restarts", type=int, default=4,
                        help="Use 4 to use 4 GPUs, one restart per GPU.")
    parser.add_argument("--lr", type=float, default=1e-2)

    # Dimension selection.
    # With multi-scale MMD, a fixed 0.015 threshold may be too strict/loose depending on data.
    # Default uses tolerance around best score. You can pass --threshold 0.03 if desired.
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--selection-abs-tol", type=float, default=2e-3)
    parser.add_argument("--selection-rel-tol", type=float, default=0.05)
    parser.add_argument("--floor-slack", dest="selection_floor_slack", type=float, default=None,
                        help="Legacy option. If set, uses the same slack for view and joint recovery.")
    parser.add_argument("--view-floor-slack", type=float, default=5e-3,
                        help="Slack added to the real-vs-real MMD floor for marginal/view dimension selection.")
    parser.add_argument("--joint-floor-slack", type=float, default=5e-4,
                        help="Slack added to the real-vs-real MMD floor for joint dimension selection.")
    parser.add_argument("--next-improvement-abs", type=float, default=2e-3,
                        help="If a dimension passes threshold but the next dimension improves by more than this, skip it.")
    parser.add_argument("--next-improvement-rel", type=float, default=0.10,
                        help="Relative version of --next-improvement-abs.")

    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-full-seeds", type=int, default=3)

    parser.add_argument("--skip-check5", action="store_true",
                        help="Skip full-pipeline multi-seed check to save time.")
    parser.add_argument("--quick", action="store_true",
                        help="Faster, rougher run: fewer iterations and seeds.")

    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    if args.quick:
        args.n_iter = min(args.n_iter, 200)
        args.n_seeds = min(args.n_seeds, 3)
        args.n_full_seeds = min(args.n_full_seeds, 1)
        args.n_samples = min(args.n_samples, 250)
        args.n_samples_joint = min(args.n_samples_joint, 300)

    args.gpu_ids = parse_gpu_ids(args.gpus)

    # Backward compatibility: if --floor-slack is passed, use it for both.
    # Otherwise, use separate defaults: views get more tolerance, joint is stricter.
    if args.selection_floor_slack is not None:
        args.view_floor_slack = args.selection_floor_slack
        args.joint_floor_slack = args.selection_floor_slack

    view_required_1 = num_poly_features(args.p_true + args.q1, args.degree)
    view_required_2 = num_poly_features(args.p_true + args.q2, args.degree)
    joint_required = num_poly_features(args.p_true + args.q1 + args.q2, args.degree)
    if args.d_obs < max(view_required_1, view_required_2):
        print("WARNING: d_obs may be too small for full-rank view polynomial features.")
        print(f"  required view features: max({view_required_1}, {view_required_2}), d_obs={args.d_obs}")
    if 2 * args.d_obs < joint_required:
        print("WARNING: 2*d_obs may be too small for full-rank joint polynomial features.")
        print(f"  required joint features: {joint_required}, 2*d_obs={2 * args.d_obs}")

    print("=" * 70)
    print("FIXED SANITY CHECKS v6 — sweep-ready parameterised p, q, d_obs")
    print("=" * 70)
    print(f"torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    print(f"Using devices: {args.gpu_ids}")
    print(f"n_restarts={args.n_restarts}, n_iter={args.n_iter}, n={args.n}")
    print(f"p_true={args.p_true}, q1={args.q1}, q2={args.q2}, d_obs={args.d_obs}, degree={args.degree}")
    print_dimension_targets(args)
    print(f"view_floor_slack={args.view_floor_slack}, joint_floor_slack={args.joint_floor_slack}")
    print(f"restart_aggregation=median, next_improvement_abs={args.next_improvement_abs}, next_improvement_rel={args.next_improvement_rel}")
    print("=" * 70)

    t_start = time.time()

    first_device = args.gpu_ids[0]
    p1 = check1_flow_exponential(gpu_id=first_device)

    p2, dims, scores, d_hat = check2_dim_recovery(args)
    p3, d1, d2, dj, ph = check3_full_pipeline(args)
    acc_single = check4_single_view_consistency(args)

    if args.skip_check5:
        acc_full = float("nan")
    else:
        acc_full = check5_full_pipeline_consistency(args)

    elapsed = time.time() - t_start

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Check 1 — Flow fits exponential:       {'✓' if p1 else '✗'}")
    print(f"Check 2 — Dim recovery r=4:            {'✓' if p2 else '✗'}")
    print(f"Check 3 — Full pipeline p=2:           {'✓' if p3 else '✗'}")
    print(f"Check 4 — Single-view consistency:     {acc_single:.1%}")

    if args.skip_check5:
        print("Check 5 — Full-pipeline consistency:   skipped")
        all_ok = p1 and p2 and p3 and acc_single >= 0.8
    else:
        print(f"Check 5 — Full-pipeline consistency:   {acc_full:.1%}")
        all_ok = p1 and p2 and p3 and acc_single >= 0.8 and acc_full >= 0.8

    print(f"\nTotal time: {elapsed / 60:.1f} minutes")
    print("=" * 70)

    if all_ok:
        print("✓ ALL CHECKS PASSED — reasonable to start the full sweep.")
    else:
        print("✗ SOME CHECKS FAILED — inspect scores before full sweep.")


if __name__ == "__main__":
    # This is important for CUDA + multiprocessing.
    main()
