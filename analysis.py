# src/nonstationary_demo/analysis.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    if "is_N" in out.columns:
        out["roll_pN"] = rolling_mean(out["is_N"], window)
    else:
        out["roll_pN"] = np.nan
    return out


def reversal_align(df: pd.DataFrame, marker_t: Optional[int] = None, before: int = 120, after: int = 120) -> pd.DataFrame:
    """
    Returns a slice aligned so rel_t = 0 at reversal trigger time.
    """
    if marker_t is None:
        if "reversal_triggered" in df.columns and (df["reversal_triggered"] == 1).any():
            marker_t = int(df.index[df["reversal_triggered"] == 1].min())
        else:
            raise ValueError("No reversal marker found. Provide marker_t explicitly.")

    start = max(0, marker_t - before)
    end = min(len(df), marker_t + after + 1)

    out = df.iloc[start:end].copy()
    out["rel_t"] = np.arange(start - marker_t, end - marker_t)
    out = out.set_index("rel_t", drop=True)
    return out


def plot_learning_curves(df: pd.DataFrame, window: int = 50, title: str = "Synthetic learning curves") -> plt.Figure:
    d = compute_rolling_summaries(df, window=window)

    fig = plt.figure()
    plt.plot(d["roll_acc"].values, label="rolling accuracy")
    plt.plot(d["roll_pL"].values, label="P(L)")
    plt.plot(d["roll_pR"].values, label="P(R)")
    if "roll_pN" in d.columns:
        plt.plot(d["roll_pN"].values, label="P(N)")
    plt.xlabel("trial")
    plt.ylabel("rolling mean")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    return fig


def plot_reversal_aligned(df: pd.DataFrame, window: int = 50, before: int = 120, after: int = 120) -> plt.Figure:
    d = compute_rolling_summaries(df, window=window)
    a = reversal_align(d, before=before, after=after)

    fig = plt.figure()
    plt.plot(a["roll_acc"].values, label="rolling accuracy")
    plt.plot(a["roll_pL"].values, label="P(L)")
    plt.plot(a["roll_pR"].values, label="P(R)")
    if "roll_pN" in a.columns:
        plt.plot(a["roll_pN"].values, label="P(N)")
    plt.axvline(x=before, linestyle="--")  # rel_t=0 is at index 'before'
    plt.xlabel("reversal-aligned trial (rel_t)")
    plt.ylabel("rolling mean")
    plt.title("Reversal-aligned synthetic curves")
    plt.legend()
    plt.tight_layout()
    return fig
