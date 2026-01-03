# nonstationary-choice-models-demo
**Simulation-only demo: nonstationary choice behavior in a rule-learning / reversal task**

This repository is a lightweight, reproducible demo of a **synthetic** nonstationary choice-modeling pipeline:

1) **Synthetic simulation** of a tone-guided rule-learning / reversal environment  
2) **Analysis + visualization** of behavior around reversal (rolling accuracy, rolling action frequencies, reversal-aligned overlays)

✅ **No experimental mouse data** is included.  
✅ **No model-parameter estimation** is performed in this repo (simulation + plotting only).

> GitHub renders `.ipynb` notebooks as **static HTML** (viewable, but not executable).  
> To run the notebook in the browser, use Binder (instructions below), or run locally.

---

## What’s in the demo

### 1) Synthetic simulation (no experimental data required)
- Generates **states (tones), actions (L/R/N), rewards**, and a **reversal** event.
- Lets you vary simulation parameters (e.g., learning rate / inverse temperature / action biases) and see how behavior changes.

### 2) Analysis + reversal-aligned visualization
- Rolling accuracy
- Rolling action frequencies: P(L), P(R), P(N)
- Reversal-aligned overlays (centered so the reversal marker is at `rel_t = 0`)

---

## Repository structure
- `src/nonstationary_demo/simulate.py` — synthetic generators (task + agent simulation)
- `src/nonstationary_demo/analysis.py` — rolling metrics + reversal alignment + plotting helpers
- `src/nonstationary_demo/__init__.py` — small public API
- `notebooks/demo1.ipynb` — the main demo notebook
- `results/figures/` — saved figures (optional)

---

## Quick start (local)

```bash
# 1) Create and activate a clean environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# 2) Install dependencies
pip install -U pip
pip install -e ".[dev]"

# 3) Run the notebook
jupyter notebook notebooks/demo1.ipynb
