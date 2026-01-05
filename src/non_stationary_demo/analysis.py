# src/non_stationary_demo/analysis.py
from __future__ import annotations

from typing import Optional, Sequence, List, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Helpers for learning curves
# -------------------------

def rolling_mean(x: pd.Series, window: int) -> pd.Series:
    return x.rolling(window=window, min_periods=max(1, window // 5)).mean()


def add_action_indicators(df: pd.DataFrame, actions: Sequence[str] = ("L", "R", "N")) -> pd.DataFrame:
    out = df.copy()
    for a in actions:
        out[f"is_{a}"] = (out["action"].astype(str).str.upper() == a).astype(int)
    return out


def compute_rolling_summaries(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """
    Adds:
      - roll_acc
      - roll_pL, roll_pR, roll_pN
    """
    out = add_action_indicators(df)
    out["roll_acc"] = rolling_mean(out["correct"], window)
    out["roll_pL"] = rolling_mean(out["is_L"], window)
    out["roll_pR"] = rolling_mean(out["is_R"], window)
    out["roll_pN"] = rolling_mean(out["is_N"], window) if "is_N" in out.columns else np.nan
    return out


# -------------------------
# Reversal timing helpers
# -------------------------

def get_reversal_trigger_times(df: pd.DataFrame) -> List[int]:
    """
    Times where criterion was met and the countdown was armed.
    Uses df['reversal_triggered']==1 if present.
    """
    d = df.reset_index(drop=True)
    if "reversal_triggered" not in d.columns:
        return []
    m = d["reversal_triggered"].astype(int) == 1
    if not m.any():
        return []
    return [int(i) for i in d.index[m].tolist()]


def get_reversal_flip_times(df: pd.DataFrame) -> List[int]:
    """
    Times where the rule actually flipped.
    Preferred: detect from df['rule'] step changes (diff>0).
    Fallback: df['reversal_t'] if present.
    """
    d = df.reset_index(drop=True)

    if "rule" in d.columns:
        # diff() is the standard discrete change detector for step-like series. :contentReference[oaicite:3]{index=3}
        flips = d.index[d["rule"].astype(float).diff().fillna(0.0) > 0.0].tolist()
        return [int(i) for i in flips]

    if "reversal_t" in d.columns:
        vals = d["reversal_t"].dropna()
        if len(vals) > 0:
            # could be scalar or repeated; accept first if scalar-like
            if np.isscalar(vals.iloc[0]) and np.isfinite(vals.iloc[0]):
                return [int(vals.iloc[0])]
    return []


def _first_reversal_index(df: pd.DataFrame, *, kind: Literal["trigger", "flip"] = "trigger") -> Optional[int]:
    """
    Backward-compatible: returns FIRST trigger or FIRST flip time.
    """
    if kind == "flip":
        xs = get_reversal_flip_times(df)
    else:
        xs = get_reversal_trigger_times(df)

    return int(xs[0]) if len(xs) > 0 else None


def _draw_reversal_lines(
    ax: plt.Axes,
    xs: Sequence[int],
    *,
    label: str,
    linestyle: str = "--",
    linewidth: float = 1.0,
    alpha: float = 1.0,
) -> None:
    """
    Draw many vertical lines efficiently.
    Uses vlines() which accepts array-like x positions. :contentReference[oaicite:4]{index=4}
    """
    if xs is None:
        return
    xs = [int(x) for x in xs]
    if len(xs) == 0:
        return

    ymin, ymax = ax.get_ylim()
    ax.vlines(xs, ymin=ymin, ymax=ymax, linestyles=linestyle, linewidth=linewidth, alpha=alpha, label=label)
    # Keep original limits (vlines can sometimes expand autoscale depending on call order)
    ax.set_ylim(ymin, ymax)


def reversal_align(
    df: pd.DataFrame,
    marker_t: Optional[int] = None,
    before: int = 120,
    after: int = 120,
    *,
    marker_kind: Literal["trigger", "flip"] = "trigger",
) -> pd.DataFrame:
    """
    Returns a slice aligned so rel_t = 0 at a reversal marker time.
    Output index is rel_t in [-before, ..., +after].

    marker_kind:
      - "trigger": align to criterion/arming time
      - "flip": align to actual rule flip time
    """
    d = df.reset_index(drop=True).copy()

    if marker_t is None:
        marker_t = _first_reversal_index(d, kind=marker_kind)
        if marker_t is None:
            raise ValueError("No reversal marker found. Provide marker_t explicitly or ensure df has reversal info.")

    marker_t = int(marker_t)
    start = max(0, marker_t - int(before))
    end = min(len(d), marker_t + int(after) + 1)

    out = d.iloc[start:end].copy()
    out["rel_t"] = np.arange(start - marker_t, end - marker_t)
    out = out.set_index("rel_t", drop=True)
    return out


# -------------------------
# Fixed-model plots (unchanged defaults; now can show all reversals too)
# -------------------------

def plot_learning_curves(df: pd.DataFrame, window: int = 50, title: str = "Synthetic learning curves") -> plt.Figure:
    d = compute_rolling_summaries(df, window=window)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(d["roll_acc"].values, label="rolling accuracy")
    ax.plot(d["roll_pL"].values, label="P(L)")
    ax.plot(d["roll_pR"].values, label="P(R)")
    ax.plot(d["roll_pN"].values, label="P(N)")
    ax.set_xlabel("trial")
    ax.set_ylabel("rolling mean")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_reversal_aligned(
    df: pd.DataFrame,
    window: int = 50,
    before: int = 120,
    after: int = 120,
    *,
    marker_kind: Literal["trigger", "flip"] = "trigger",
) -> plt.Figure:
    d = compute_rolling_summaries(df, window=window)
    a = reversal_align(d, before=before, after=after, marker_kind=marker_kind)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(a["roll_acc"].values, label="rolling accuracy")
    ax.plot(a["roll_pL"].values, label="P(L)")
    ax.plot(a["roll_pR"].values, label="P(R)")
    ax.plot(a["roll_pN"].values, label="P(N)")
    ax.axvline(x=0, linestyle="--", linewidth=1)
    ax.set_xlabel("reversal-aligned trial (rel_t)")
    ax.set_ylabel("rolling mean")
    ax.set_title(f"Reversal-aligned synthetic curves ({marker_kind})")
    ax.legend()
    fig.tight_layout()
    return fig


# -------------------------
# Dynamic-parameter plotting
# -------------------------

DYN_PARAM_COLS = ["beta_t", "bias_L_t", "bias_R_t", "bias_N_t"]
ReversalMode = Literal["none", "first", "all"]
ReversalKind = Literal["trigger", "flip"]


def _get_x(d: pd.DataFrame) -> pd.Series:
    return d["t"] if "t" in d.columns else pd.Series(d.index, index=d.index)


def _reversal_xs(d: pd.DataFrame, *, reversals: ReversalMode, kind: ReversalKind) -> List[int]:
    if reversals == "none":
        return []
    xs = get_reversal_flip_times(d) if kind == "flip" else get_reversal_trigger_times(d)
    if reversals == "first":
        return xs[:1]
    return xs


def plot_dynamic_params(
    df: pd.DataFrame,
    *,
    smooth_window: Optional[int] = 25,
    reversals: ReversalMode = "first",
    reversal_kind: ReversalKind = "trigger",
    title: str = "Dynamic parameter trajectories",
) -> plt.Figure:
    """
    Plot beta_t and biases over trials (optionally smoothed).

    reversals:
      - "none": no markers
      - "first": one marker
      - "all": draw markers for every reversal
    reversal_kind:
      - "trigger": criterion/arming time
      - "flip": actual rule flip time
    """
    missing = [c for c in DYN_PARAM_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing dynamic columns in df: {missing}")

    d = df.reset_index(drop=True).copy()
    x = _get_x(d)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for c in DYN_PARAM_COLS:
        y = d[c].astype(float)
        if smooth_window is not None and smooth_window >= 2:
            y = y.rolling(smooth_window, min_periods=1).mean()
        ax.plot(x, y, label=c)

    xs = _reversal_xs(d, reversals=reversals, kind=reversal_kind)
    if len(xs) > 0:
        _draw_reversal_lines(ax, xs, label=f"reversal {reversal_kind}", linestyle="--", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("trial")
    ax.set_ylabel("value")
    ax.legend(ncols=2, fontsize=9)
    fig.tight_layout()
    return fig


def plot_dynamic_params_reversal_aligned(
    df: pd.DataFrame,
    *,
    before: int = 150,
    after: int = 150,
    smooth_window: Optional[int] = 15,
    marker_kind: ReversalKind = "flip",
    title: str = "Dynamic parameters aligned to reversal",
) -> plt.Figure:
    """
    Plot beta_t and biases vs rel_t around the FIRST reversal marker.
    marker_kind defaults to "flip" (actual contingency change).
    """
    nbhd = reversal_align(df, before=before, after=after, marker_kind=marker_kind)

    missing = [c for c in DYN_PARAM_COLS if c not in nbhd.columns]
    if missing:
        raise KeyError(f"Missing dynamic columns in df: {missing}")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for c in DYN_PARAM_COLS:
        y = nbhd[c].astype(float)
        if smooth_window is not None and smooth_window >= 2:
            y = y.rolling(smooth_window, min_periods=1).mean()
        ax.plot(nbhd.index, y, label=c)

    ax.axvline(0, linestyle="--", linewidth=1, label=f"reversal ({marker_kind}, rel_t=0)")
    ax.set_title(title)
    ax.set_xlabel("rel_t")
    ax.set_ylabel("value")
    ax.legend(ncols=2, fontsize=9)
    fig.tight_layout()
    return fig


def plot_choice_prob_evolution(
    df: pd.DataFrame,
    *,
    window: int = 50,
    show_policy: bool = True,
    reversals: ReversalMode = "first",
    reversal_kind: ReversalKind = "trigger",
    title: str = "Choice probabilities over time (dynamic model)",
) -> plt.Figure:
    """
    Shows:
      - Rolling empirical frequencies of chosen actions: P̂(L), P̂(R), P̂(N)
      - (Optional) Rolling mean of policy probabilities: π(L), π(R), π(N) (if present)
      - Reversal markers (first or all)
    """
    d = df.reset_index(drop=True).copy()
    x = _get_x(d)

    a = d["action"].astype(str).str.upper()
    d["is_L"] = (a == "L").astype(float)
    d["is_R"] = (a == "R").astype(float)
    d["is_N"] = (a == "N").astype(float)

    pL = d["is_L"].rolling(window, min_periods=1).mean()
    pR = d["is_R"].rolling(window, min_periods=1).mean()
    pN = d["is_N"].rolling(window, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(x, pL, label="rolling P̂(L) (empirical)")
    ax.plot(x, pR, label="rolling P̂(R) (empirical)")
    ax.plot(x, pN, label="rolling P̂(N) (empirical)")

    if show_policy and {"pi_L", "pi_R", "pi_N"}.issubset(d.columns):
        ax.plot(x, d["pi_L"].rolling(window, min_periods=1).mean(), linestyle="--", label="rolling π(L) (policy)")
        ax.plot(x, d["pi_R"].rolling(window, min_periods=1).mean(), linestyle="--", label="rolling π(R) (policy)")
        ax.plot(x, d["pi_N"].rolling(window, min_periods=1).mean(), linestyle="--", label="rolling π(N) (policy)")

    xs = _reversal_xs(d, reversals=reversals, kind=reversal_kind)
    if len(xs) > 0:
        _draw_reversal_lines(ax, xs, label=f"reversal {reversal_kind}", linestyle="--", linewidth=1)

    ax.set_xlabel("trial")
    ax.set_ylabel("probability")
    ax.set_title(title)
    ax.legend(ncols=2, fontsize=9)
    fig.tight_layout()
    return fig


def plot_bias_increment_drift_to_zero(
    df: pd.DataFrame,
    *,
    tau_bias: float,
    window: int = 25,
    reversals: ReversalMode = "first",
    reversal_kind: ReversalKind = "trigger",
    title: str = "Bias increment drift-to-zero (mixture weight + increments)",
) -> plt.Figure:
    """
    Diagnostics for your drift-to-zero design.

    Plots:
      - w(t)=exp(-t/tau_bias) (or df['w_bias'] if present)
      - rolling means of d_bias_L, d_bias_R, d_bias_N
      - reversal markers (first or all)
    """
    d = df.reset_index(drop=True).copy()
    x = _get_x(d)

    if "w_bias" in d.columns:
        w = d["w_bias"].astype(float)
    else:
        tau_bias = float(tau_bias)
        w = np.exp(x.astype(float) / (-tau_bias)) if tau_bias > 0 else np.ones_like(x, dtype=float)

    needed = {"d_bias_L", "d_bias_R", "d_bias_N"}
    if not needed.issubset(d.columns):
        raise KeyError(f"Need columns {sorted(needed)} in df to plot increments.")

    dL = d["d_bias_L"].astype(float).rolling(window, min_periods=1).mean()
    dR = d["d_bias_R"].astype(float).rolling(window, min_periods=1).mean()
    dN = d["d_bias_N"].astype(float).rolling(window, min_periods=1).mean()

    fig, ax1 = plt.subplots(figsize=(9, 4.5))

    ax1.plot(x, dL, label="roll mean d_bias_L")
    ax1.plot(x, dR, label="roll mean d_bias_R")
    ax1.plot(x, dN, label="roll mean d_bias_N")
    ax1.set_xlabel("trial")
    ax1.set_ylabel("increment (rolling mean)")

    ax2 = ax1.twinx()
    ax2.plot(x, w, linestyle="--", label="w(t)=exp(-t/tau_bias)")
    ax2.set_ylabel("mixture weight w(t)")
    ax2.set_ylim(-0.05, 1.05)

    # Combine legends from both axes using get_legend_handles_labels(). :contentReference[oaicite:5]{index=5}
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, ncols=2, fontsize=9)

    # Add reversal markers on ax1 (so they match x-axis)
    xs = _reversal_xs(d, reversals=reversals, kind=reversal_kind)
    if len(xs) > 0:
        _draw_reversal_lines(ax1, xs, label=f"reversal {reversal_kind}", linestyle="--", linewidth=1, alpha=0.9)

    ax1.set_title(title)
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, path, *, dpi: int = 200) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
