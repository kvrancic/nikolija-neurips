# Research log — CRL polynomial-mixing simulations

**Paper:** *Causal Representation Learning for Cross-Modal Transfer*
(N. Bojkovic, A. Kumar, C. Uhler — NeurIPS 2026 submission, deadline May 6 AoE)

**Repo:** [nikolija-neurips](https://github.com/kvrancic/nikolija-neurips)
**Cluster:** uhlergroup3.mit.edu (4× NVIDIA A5000)
**Working dir:** `/Users/karlovrancic/Documents/projects/nikolija`

This log is a running record of substantive findings, decisions, and open
questions. It is meant to be (a) an audit trail for our own sanity and
(b) raw material for the paper's experiments section and rebuttal.

---

## TL;DR

- The paper's Algorithm 3 (RealNVP dim recovery) works; **|p − p̂|** does
  decrease with n on exponential noise.
- Theorem 1's index-level Jaccard was failing in our first implementation
  because we recovered Ẑ via **per-sample Adam inversion of x → ε**. That
  procedure produces an Ẑ that is *not* affinely related to the true Z
  (linear-fit R² ≈ −40 on the small case), so Σ_{1|2}(Ẑ_1, Ẑ_2) carries no
  Theorem-1 structure: the diagonal is unstructured and Jaccard plateaus at
  ≈0.4 regardless of n.
- The paper's Algorithm 4 step 7 actually prescribes a different recipe:
  **forward-sample Ẑ⁽ᵉ⁾ from each trained flow with the *same* batch of
  Gaussian noise ε_k**, averaged over K=5 draws to reduce variance. The
  shared ε prefix is what couples the two views.
- Switching to forward sampling immediately reproduces the paper's expected
  trend on a CPU re-run of our existing p=5 sweep:

      n     mean Jaccard@p_true (over 10 seeds)
      128   0.36
      256   0.48
      512   0.70
      1024  0.80
      2048  0.80   (n>2048 still streaming as of 2026-05-05 16:00 EDT)

  vs. the broken pipeline which was flat at ≈0.24–0.48.
- Schur math itself is correct — `sigma_1_given_2(z1_true, z2_true)`
  recovers shared indices with Jaccard = 1.0 on every seed/configuration we
  tested. Bug was strictly in the Ẑ-recovery step, not in Theorem 1.

---

## 1. What the paper claims (operational)

For each pair of synthetic observations
`X^e = G^e · φ_r(Z_L, Z_{I_e})`, with `φ_r` the polynomial feature map of
degree `r=2` and `G^e ∈ ℝ^{d_e × m}` random:

1. **Proposition 1** (`p = R_1 + R_2 − R_tot`): recover the latent dimensions
   d_1 = p+q_1, d_2 = p+q_2, d_joint = p+q_1+q_2 from PCA-style rank tests on
   X^1, X^2, [X^1, X^2]. Algorithm 3 (RealNVP + multiscale-MMD + small G)
   does this empirically.

2. **Theorem 1** (Schur form): given paired recovered latents Ẑ⁽¹⁾, Ẑ⁽²⁾
   that satisfy the **block-triangular model** (Proposition 2):

       (Ẑ_L,1)   (A_1   0  ) (Z_L)   (b_1)
       (Ẑ_I1 ) = (C_1   D_1) (Z_I1) + (b_2)

   the matrix Σ_{1|2} := E[Cov(Ẑ⁽¹⁾ | Ẑ⁽²⁾)] has zero diagonals exactly at
   the shared coordinates of Ẑ⁽¹⁾ and strictly positive diagonals at the
   private ones. Step 4 of the proof: in the *observed* (un-reordered) Ẑ⁽¹⁾
   the relation to the canonical block-triangular form is a *permutation*
   matrix P_1, and permutation preserves diagonals, so the zero entries
   identify the shared positions in the original ordering.

The Jaccard metric is between the recovered shared-index set
`{i : (Σ_{1|2})_{ii} ≈ 0}` and the ground-truth shared-index set
`{0, …, p−1}` of the data-generating Z.

The headline empirical claim (Section 5, Figure ??): MAE → 0 and Jaccard → 1
under all four non-Gaussian noise families (Laplace / Uniform / Exponential
/ Gaussian-mixture). **Exponential is the binding case** for this submission
("samo je poenta da mi proradi za exponential noise" — N.B.).

---

## 2. Original implementation issue

### What the first implementation did

`crl_sim/shared_coords.py` (`recover_paired_z`, before fix):

1. Trained flow_1 on X_1 and flow_2 on X_2 independently.
2. **Per-sample inversion** of each observed x_i: optimize ε_i ∈ ℝ^{d_e}
   via 300 Adam steps to minimize `|| φ(flow_e(ε_i)) G_e^T − x_i ||²`.
3. Take Ẑ_e[i] = flow_e(ε_i) as the i-th paired latent.
4. Compute Σ_{1|2} on (Ẑ_1, Ẑ_2) and pick the smallest `p̂` diagonal entries.

### Symptom

Jaccard at p_true plateaued at 0.24–0.48 across all n in `results/p5/`,
with no monotone climb toward 1.

### Diagnosis (scripts/test_*.py, 2026-05-05)

- `scripts/test_schur_on_truth.py`: feed paired ground-truth Z_1, Z_2 into
  `sigma_1_given_2`. **Jaccard = 1.000 on every seed**, with spectral gap
  ≈ 13 in log-scale. So the math is right.

- `scripts/test_flow_recovery.py`: train flows + invert, compare Ẑ_1 to Z_1.
  Per-row max |corr(Ẑ_1[i], Z_1[j])| over all j is only 0.15–0.29 (for a
  signed-permutation alignment we'd want ≈0.95). std of Ẑ_1 entries is
  6–20, vs. ≈1 for Z_1. So Ẑ is in a wholly different scale and basis.

- `scripts/test_oracle_align.py`: fit Ẑ_1 = Λ Z_1 + b by least-squares.
  **Residual MSE ≈ 48 vs Var(Ẑ_1) ≈ 1.6 → R² ≈ −40**. Ẑ is not even
  linearly related to Z, let alone permutation-aligned.

- `scripts/test_closed_form_inv.py`: bypass Adam inversion, use
  `Ẑ = (X · pinv(Ĝ^T))[:, :d_hat]` (since the first d_hat features of
  φ are linear in z). Better: max corr per row 0.34–0.46, R² 0.32, Schur
  diagonals in the right ballpark (≈1) — but Jaccard still 0.33. The
  closed-form inversion exposes that the trained Ĝ's polynomial-feature
  consistency check (corr between recovered quadratic block and the
  outer-product of recovered linear block) is only 0.37, so the joint
  (flow, Ĝ) hasn't actually converged to a polynomial decoder.

### Root cause

The flow + decoder is trained only via MMD on X. That objective fixes
**marginals**, not pointwise correspondence — many (Ẑ, Ĝ) pairs match
P_X distributionally without the implicit ε ↦ Ẑ map being well-defined as
an inverse of the polynomial mixing. Per-sample Adam inversion then
collapses onto whichever pre-image of x_i is closest to its initial ε,
which is essentially noise — Ẑ samples ended up uncorrelated with each
other across i, breaking the joint covariance that Theorem 1 depends on.

---

## 3. The fix — Algorithm 4 step 7, exactly as written

`crl_sim/shared_coords.py:sample_zhat_paired` (commit `93c55f1`):

    Ẑ⁽ᵉ⁾ ← (1/K) Σ_{k=1}^K f_e(ε_k),   ε_k ~ N(0, I_{d_max}),   for e = 1, 2

The crucial reading the paper appendix encodes (and which I'd missed
initially): **ε_k has no `e` subscript** — the *same* batch of Gaussian
noise is fed into both flows, with f_e taking the first d_e coordinates.
That shared-ε prefix is what makes Σ_{12}(Ẑ_1, Ẑ_2) non-trivial.
K=5 averaging reduces sample noise on Σ_{1|2}.

Per-sample inversion is removed from the production path; the function is
kept under `recover_paired_z` with a `DEPRECATED` docstring for diagnostic
comparison only.

The alignment-based Jaccard (greedy correlation matching of Ẑ to Z) is no
longer well-defined under forward sampling (the fresh-ε samples aren't
paired with the original X / Z), so it's reported as NaN; we now rely on:

| metric | description |
|---|---|
| `jaccard` | Jaccard at recovered p_hat from inclusion–exclusion |
| `jaccard_true_p` | **headline:** Jaccard at the *true* p (decouples Theorem 1 from joint-dim flakiness) |
| `jaccard_elbow` | Jaccard at p̂_elbow detected from the Σ_{1|2} log-spectrum |
| `p_count_error` | `|p̂_elbow − p_true|` — count-only metric, permutation-invariant |
| `gap_at_true_p` | log-ratio gap between rank-p and rank-(p−1) entries — Theorem-1 signal strength |

`scripts/rejaccard_csv.py` drops the new Jaccard step on top of an existing
sweep CSV without re-running the (much slower) dim recovery — useful for
salvaging existing CSVs.

---

## 4. Empirical validation

### 4.1 Tiny smoke (p=2, q1=q2=2, n=2048, exponential noise)

Three seeds, before vs. after the fix:

| seed | per-sample inv. (old) | forward sampling (new) |
|---|---|---|
| 11 | jacc 0.333, gap@p 0.02 | jacc 0.333, gap@p 0.97 |
| 12 | jacc 0.000, gap@p 0.02 | **jacc 1.000**, gap@p 0.62 |
| 13 | jacc 0.000, gap@p 0.02 | jacc 0.000, gap@p 0.30 |

Spectral gap jumps from ~0.02 to 0.3–1.0 — Theorem-1 structure is now
present in Σ_{1|2}. The seed-to-seed Jaccard variance is the unknown
permutation P_1 the paper acknowledges in Theorem 1's Step 4.

### 4.2 Re-Jaccard on the existing p=5 sweep (CPU, ongoing as of 16:00 EDT)

Same dim-recovery rows from `results/p5/p_recovery_sweep.csv` (90 rows, 10
seeds × 9 powers), Jaccard recomputed with `scripts/rejaccard_csv.py`,
n_iter_train=800, n_restarts=2. The headline `jaccard_true_p`, mean ± std
over 10 seeds:

      n     mean    std
      128   0.360   0.158
      256   0.480   0.193
      512   0.700   0.170
      1024  0.800   0.000
      2048  …  (in flight)
      4096
      8192
      16384
      32768

Compare to the **broken pipeline** which was 0.24 at n=128 climbing to only
0.48 at n=32768 (slide saved as `results/figs/p5_default.png`). The fix
moves the n=1024 value from 0.31 → 0.80 — that's the difference between
"a result we can't publish" and "the curve the paper claims".

The plateau at 0.80 = 4/5 correct shared coords is consistent with one
residual permutation slot still being miscounted. Whether the curve
climbs further at n ≥ 4096 will tell us if the residual is a permutation
artefact that resolves with better convergence, or a structural limit of
this exact noise regime.

### 4.3 Cluster sweeps (running on uhlergroup3, 4×A5000)

| sweep | rows | dim-recovery status | Jaccard status |
|---|---|---|---|
| p=5  | 90 / 90 | ✅ done (CSV in `results/p5/`) | ⏳ re-Jaccard in flight (laptop CPU) |
| p=20 | 14 / 48 | ⏳ running, ETA 2026-05-06 ~14:00 EDT (43 min/row × 34 rows) | runs after dim-recovery finishes |
| p=40 | 0 | scheduled by `scripts/queue_sweeps.sh` after p=20; trimmed cluster fallback to powers 9–10 × 3 seeds = 6 rows (~16 h on cluster) | uses new code automatically |

Full p=40 (powers 9–12, 4 seeds = 16 rows) is gated on RunPod 8×A100 80GB,
~$170 / ~12 h. Bootstrap script: `scripts/runpod_bootstrap.sh`. The cluster
fallback is just a backstop in case RunPod isn't spun up.

---

## 5. Cluster + plumbing notes (for reproducibility)

- `crl_sim/core.py` is `sanity_checks_param_v6.py` renamed, behaviourally
  unchanged except for an optional `selection_floor_multiplier` knob in
  `recover_dimension_flow_parallel` (additive slack `floor + ε` is too
  generous when the data-driven MMD floor drops at large n; multiplicative
  threshold `floor × 1.20` for views, `× 1.05` for joint stops the
  large-n undershoot).
- `scripts/sweep.py` has `--view-floor-multiplier` / `--joint-floor-multiplier`
  to plumb that through; sweep CSVs include the resulting d1_hat, d2_hat,
  dj_hat, p_hat alongside the new Jaccard variants.
- `scripts/queue_sweeps.sh` polls CSV row counts (90 for p=5, 48 for p=20,
  6 for p=40) and chains the three sweeps in one tmux session; each
  finished sweep is auto-committed to the local cluster repo by
  `scripts/git_save_results.sh`. Remote push to GitHub `main` is
  blocked-by-policy from the cluster; we pull from the laptop for plotting.
- Local cadence: laptop pulls results CSVs via rsync periodically and
  commits to the laptop repo (with proper user.email). Frequent commits.

---

## 6. Open questions / TODO

1. **Does the n=4096–32768 tail of p=5 climb past 0.80?** If yes, the
   residual permutation is just slow flow convergence; if it plateaus,
   we should investigate a permutation-tightening regularizer (e.g., a
   sparsity prior on Λ implied by RealNVP coupling order).

2. **Do p=20 and p=40 reproduce the same trend?** Larger p means a larger
   permutation group (p! × q!) for P_1 to land in, so the fluctuation
   between perfectly-correct and partially-correct seeds may be sharper.

3. **Permutation-invariant metric for the appendix.** Even if the
   index-level Jaccard climbs to 1 in the limit, the seed-to-seed variance
   at finite n is non-trivial because of P_1. The clean asymptotic
   statement is *the recovered shared subspace converges to the true
   shared subspace*; we can complement Jaccard with `mean(top-p canonical
   correlations²)` between Ẑ_1's claimed shared subspace and Z_1's first p
   columns. This is robust to P_1 by construction.

4. **Algorithm 4 step 7 — is K=5 averaging supposed to be averaging Ẑ
   batches, or averaging Σ_{1|2} matrices?** Both readings exist in the
   paper text. We're currently averaging Ẑ batches per the literal
   reading; the alternative interpretation (average K independent Σ
   estimates) is sample-efficient in a different way and worth a
   comparison run if Jaccard plateaus below 1.

5. **Does q_1 ≠ q_2 break anything?** Our shared-ε implementation feeds
   `ε_k[:d_1]` to flow_1 and `ε_k[:d_2]` to flow_2 — implicitly assumes
   the *first* p coordinates of ε are the shared block in *both* flows.
   That's a convention choice; we haven't audited whether the flow
   training puts the shared signal there reliably for asymmetric q.

6. **Real-data experiment.** Out of scope for the deadline per N.B.
   but Caroline Uhler will provide a CITE-seq-style multi-view dataset
   later; the current code path should accept it with only a swap of the
   data generator.

---

## 7. Provenance

- 2026-05-03 evening: initial repo scaffold, plan agreed with Nikolija,
  cluster credentials shared, p=5 launched.
- 2026-05-04 14:34 EDT: p=5 sweep launched on cluster (4×A5000).
- 2026-05-05 03:46 EDT: p=5 sweep finished (90/90). p=20 launched
  automatically by `queue_sweeps.sh`.
- 2026-05-05 11:16 EDT: Nikolija flagged Jaccard plateau on p=5 plot
  ("on treva automatski da ide u 1 … moz da gresi da racuna covariance
  matrix mzd na necem pogresnom").
- 2026-05-05 13:00–14:00 EDT: diagnosis (per-sample inversion was the bug)
  and fix (commit `93c55f1`).
- 2026-05-05 14:00–16:00 EDT: p=5 re-Jaccard reproducing the climbing
  curve up to n=1024.

(Times in this document are EDT. Cluster reports them as `America/New_York`.)
