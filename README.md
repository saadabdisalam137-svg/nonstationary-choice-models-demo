# nonstationary-choice-models-demo
**Simulation-only demo: nonstationary choice behaviour in a tone-guided rule-learning / reversal task**

This repository is a lightweight, reproducible demo of a **synthetic** nonstationary choice-modelling pipeline:

1) **Synthetic simulation** of a tone-guided rule-learning / reversal environment  
2) **Analysis + visualization** of behaviour around reversals (rolling accuracy, rolling action frequencies, reversal markers, reversal-aligned views)

**No experimental mouse data** is included.  
**No model-parameter estimation** is performed in this repo (simulation + plotting only).

> GitHub renders `.ipynb` notebooks as **static HTML** (viewable, not executable). :contentReference[oaicite:1]{index=1}  
> To run in the browser, use Binder, or run locally. :contentReference[oaicite:2]{index=2}

<!-- Optional: add a Binder badge once you set it up -->
<!-- [![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/<YOUR_GITHUB_USER>/<YOUR_REPO>/<BRANCH>?filepath=notebooks/demo1.ipynb) -->
<!-- Binder URL patterns like ?filepath=... are commonly used to open a specific notebook. :contentReference[oaicite:3]{index=3} -->

---

## What’s in the demo

### 1) Fixed-parameter simulation (Q-learning + softmax)
- Generates **states (tones), actions (L/R/N), rewards**, and **reversal events**.
- A reversal is armed when recent performance reaches **≥19 correct in the last 20 trials**, then executed **250 trials later** (countdown).  
- Plots: rolling accuracy + rolling action frequencies; and reversal-aligned overlays (rel_t = 0 at the reversal marker).

### 2) Dynamic-parameter simulation (random-walk β(t) and biases)
- Same task, but **β(t)** and action biases **b_L(t), b_R(t), b_N(t)** evolve over time via random walks.
- Supports **multiple serial reversals**, and the plotting utilities can draw **all reversal flip times** (not just the first).

### 3) Analysis + plotting utilities
- Rolling empirical choice probabilities P̂(L), P̂(R), P̂(N)
- (Optional) policy probabilities π(L), π(R), π(N) if logged
- Parameter trajectories (β(t), biases)
- Reversal-aligned windows and reversal markers (first vs all flips)

---

## Repository structure
- `src/non_stationary_demo/simulate.py` — synthetic generators (task + fixed and dynamic agent simulation)
- `src/non_stationary_demo/analysis.py` — rolling metrics + reversal alignment + plotting helpers
- `src/non_stationary_demo/__init__.py` — small public API
- `notebooks/demo1.ipynb` — main demo notebook
- `results/figures/` — saved figures (optional)

---

## Reproducibility
- Simulations are **deterministic given a seed** (e.g., `seed=0`).  
  Change the seed to generate different reversal times and trajectories.

---

## Quick start (local)

```bash
# 1) Create and activate a clean environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# 2) Install dependencies
pip install -U pip
pip install -e .

# 3) Run the notebook
jupyter notebook notebooks/demo1.ipynb
