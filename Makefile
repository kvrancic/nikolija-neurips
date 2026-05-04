# Makefile for Nikolija's CRL simulations.
# Sweep configs live here (single source of truth). The Python scripts only
# take CLI args; this file packages those into named targets.

# ---- Where things live ----
PY        := python3
RESULTS   := results
FIGS      := $(RESULTS)/figs

# ---- Compute knobs (override on the command line) ----
GPUS      ?= 0,1,2,3      # comma-separated; pass GPUS=cpu to force CPU
RESTARTS  ?= 4            # set to 8 for 8-GPU pods
JOBS      ?= 1            # not used for now; placeholder

# Per-sweep dim recovery knobs
N_ITER    ?= 800
N_SAMPLES ?= 768

# ---- Help ----
.PHONY: help
help:
	@echo "Targets:"
	@echo "  make sanity-local           CPU smoke test (~15-30 min)"
	@echo "  make smoke-cluster          ~10 min cluster smoke (1 GPU, 2 seeds @ 2^7)"
	@echo "  make sweep-p5 [GPUS=...]    full p=5 sweep (~12h on 4xA5000)"
	@echo "  make sweep-p20 [GPUS=...]   full p=20 sweep (~30h on 4xA5000)"
	@echo "  make sweep-p40 [GPUS=...]   full p=40 sweep (~60-80h on 4xA5000)"
	@echo "  make plots                  generate all figures from existing CSVs"
	@echo "  make plot-p{5,20,40}        single-p figure"
	@echo "  make commit-results         git commit anything new under $(RESULTS)/"
	@echo ""
	@echo "Override compute knobs: GPUS=0,1,2,3,4,5,6,7 RESTARTS=8 N_ITER=400"

# ---- Local smoke ----
.PHONY: sanity-local
sanity-local:
	$(PY) scripts/sanity_local.py

# ---- Cluster smoke ----
.PHONY: smoke-cluster
smoke-cluster:
	$(PY) scripts/sweep.py \
	    --p-true 2 --q1 2 --q2 2 --degree 2 \
	    --powers 7 --seeds-per-n 2 \
	    --gpus $(GPUS) --n-restarts 2 \
	    --n-iter 400 --n-samples 512 \
	    --jaccard-n-iter-train 200 --jaccard-n-iter-inv 200 --jaccard-n-restarts 1 \
	    --out-dir $(RESULTS)/smoke --csv-name smoke.csv

# ---- Real sweeps ----
# Powers and seed counts match the plan; tune via env vars per Nikolija's
# call (q1=q2=p, degree=2, exponential noise hard-coded in core.py).
.PHONY: sweep-p5
sweep-p5:
	$(PY) scripts/sweep.py \
	    --p-true 5 --q1 5 --q2 5 --degree 2 \
	    --powers 5-17 --seeds-per-n 20 \
	    --gpus $(GPUS) --n-restarts $(RESTARTS) \
	    --n-iter $(N_ITER) --n-samples $(N_SAMPLES) \
	    --out-dir $(RESULTS)/p5 --csv-name p_recovery_sweep.csv
	@./scripts/git_save_results.sh $(RESULTS)/p5 "results: p5 sweep done"

.PHONY: sweep-p20
sweep-p20:
	$(PY) scripts/sweep.py \
	    --p-true 20 --q1 20 --q2 20 --degree 2 \
	    --powers 7-17 --seeds-per-n 10 \
	    --gpus $(GPUS) --n-restarts $(RESTARTS) \
	    --n-iter $(N_ITER) --n-samples $(N_SAMPLES) \
	    --out-dir $(RESULTS)/p20 --csv-name p_recovery_sweep.csv
	@./scripts/git_save_results.sh $(RESULTS)/p20 "results: p20 sweep done"

.PHONY: sweep-p40
sweep-p40:
	$(PY) scripts/sweep.py \
	    --p-true 40 --q1 40 --q2 40 --degree 2 \
	    --powers 9-17 --seeds-per-n 5 \
	    --gpus $(GPUS) --n-restarts $(RESTARTS) \
	    --n-iter $(N_ITER) --n-samples $(N_SAMPLES) \
	    --out-dir $(RESULTS)/p40 --csv-name p_recovery_sweep.csv
	@./scripts/git_save_results.sh $(RESULTS)/p40 "results: p40 sweep done"

# ---- Plotting ----
.PHONY: plots plot-p5 plot-p20 plot-p40
plots: plot-p5 plot-p20 plot-p40

plot-p5:
	@if [ -f $(RESULTS)/p5/p_recovery_sweep.csv ]; then \
	    $(PY) scripts/plot.py $(RESULTS)/p5/p_recovery_sweep.csv --out $(FIGS)/p5.png; \
	else echo "skipping plot-p5: no CSV at $(RESULTS)/p5/p_recovery_sweep.csv"; fi

plot-p20:
	@if [ -f $(RESULTS)/p20/p_recovery_sweep.csv ]; then \
	    $(PY) scripts/plot.py $(RESULTS)/p20/p_recovery_sweep.csv --out $(FIGS)/p20.png; \
	else echo "skipping plot-p20: no CSV at $(RESULTS)/p20/p_recovery_sweep.csv"; fi

plot-p40:
	@if [ -f $(RESULTS)/p40/p_recovery_sweep.csv ]; then \
	    $(PY) scripts/plot.py $(RESULTS)/p40/p_recovery_sweep.csv --out $(FIGS)/p40.png; \
	else echo "skipping plot-p40: no CSV at $(RESULTS)/p40/p_recovery_sweep.csv"; fi

# ---- Result safety: commit any CSV/plot updates under results/ to git ----
# Forces results/ into the working tree even though it's gitignored, so we keep
# every completed (n, seed) row durable across crashes.
.PHONY: commit-results
commit-results:
	@./scripts/git_save_results.sh $(RESULTS) "results: snapshot"
