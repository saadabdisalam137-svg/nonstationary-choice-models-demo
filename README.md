# nonstationary-choice-models-demo
**An agent in a cognitive rule-learning environment (reversal tasks) + parameter-estimation demo**

This repository is a lightweight, reproducible demo of a nonstationary choice-modeling pipeline:
1) **Synthetic simulation** of a tone-guided rule-learning / reversal environment, and  
2) **Sliding-window parameter estimation** (e.g., \alpha, \beta, and biases) with visualization around reversal.

> Note: GitHub renders `.ipynb` notebooks as **static HTML** (viewable, but not executable). To run the notebook in the browser, use Binder (instructions below). :contentReference[oaicite:1]{index=1}

---

## What’s in the demo

### 1) Synthetic simulation (no experimental data required)
- Demonstrates how the synthetic environment generates **states, actions, rewards**, and how behavior changes with parameters.
- Includes plots like rolling accuracy and rolling action frequencies.

### 2) Parameter estimation + reversal-aligned visualization
- Demonstrates how estimated parameters evolve across time windows.
- Includes “reversal neighborhood” overlays (all mice on the same axes), aligned so the **first reversal marker** is at **rel_idx = 0**.

---

## Repository structure (suggested)
- `src/`  
  - Core Python modules (model + estimation utilities)
- `notebooks/`  
  - `demo.ipynb` — the main demo notebook
- `requirements.txt` — Python dependencies

---

## Quick start (local)

```bash
# 1) Create and activate a clean environment (recommended)
python -m venv .venv
source .venv/bin/activate  # (macOS/Linux)
# .venv\Scripts\activate   # (Windows)

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the notebook
jupyter notebook
