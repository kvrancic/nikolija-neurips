"""crl_sim: simulation harness for causal representation learning, cross-modal transfer.

Modules:
- core: Algorithm 3 (RealNVP + multi-scale MMD) dimension recovery, plus the
  full inclusion-exclusion pipeline. Originally Nikolija's sanity_checks_param_v6.py.
- shared_coords: Theorem 1 (Schur-complement form) — recover which coordinates
  of Z_hat correspond to the shared latent space, plus Jaccard against ground truth.
- metrics: small helpers (Jaccard, abs error).
"""

__version__ = "0.1.0"
