# src/non_stationary_demo/simulate.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import random

import numpy as np
import pandas as pd


# =========================
# Task specification helpers
# =========================

@dataclass(frozen=True)
class TaskSpec:
    """
    Defines the task/environment.

    reward_probs: mapping (state, action) -> P(reward=1)
    correct_pairs: mapping (state, action) -> 1 if "correct" under current rule, else 0
    """
    state_space: Sequence[str]
    action_space: Sequence[str]  # e.g. ["L","R","N"]
    reward_probs: Dict[Tuple[str, str], float]
    correct_pairs: Dict[Tuple[str, str], int]


def make_correct_pairs_from_reward_probs(
    reward_probs: Dict[Tuple[str, str], float],
    threshold: float = 0.5,
) -> Dict[Tuple[str, str], int]:
    """(s,a) is 'correct' if P(reward|s,a) > threshold."""
    correct: Dict[Tuple[str, str], int] = {}
    for k, p in reward_probs.items():
        correct[k] = 1 if float(p) > threshold else 0
    return correct


def reverse_reward_probs(
    reward_probs: Dict[Tuple[str, str], float],
    action_space: Sequence[str],
) -> Dict[Tuple[str, str], float]:
    """
    Flip reward probabilities for all actions except N (kept at 0).
    """
    flipped: Dict[Tuple[str, str], float] = {}
    for (s, a), p in reward_probs.items():
        if str(a).upper() == "N":
            flipped[(s, a)] = 0.0
        else:
            flipped[(s, a)] = 1.0 - float(p)

    # Ensure every (s, a) exists (optional, but safer)
    for s in {k[0] for k in reward_probs}:
        for a in action_space:
            flipped.setdefault((s, a), 0.0 if str(a).upper() == "N" else 0.5)
    return flipped


# =========================
# Agent parameters (fixed)
# =========================

@dataclass(frozen=True)
class AgentParams:
    """
    Simulation-only parameters (not estimated in Repo 1).

    NOTE:
      In the dynamic simulator, these act as INITIAL CONDITIONS / fixed parts:
        - alpha is fixed (Q-update learning rate)
        - beta, bias_L, bias_R, bias_N are initial values for the random walks
    """
    alpha: float
    beta: float
    bias_L: float = 0.0
    bias_R: float = 0.0
    bias_N: float = 0.0


# =========================
# Dynamic random-walk params
# =========================

@dataclass(frozen=True)
class DynamicRWParams:
    """
    Random-walk hyperparameters for dynamic biases and beta.

    Bias increments:
      For each action a in {L,R,N}, sample increment from mixture:
        w(t) * N(mu_a, sigma_a^2) + (1-w(t)) * N(0, sigma_a^2)
      where w(t) = exp(-t/tau_bias).

      Then enforce sum-to-zero (translation invariance fix):
        d <- d - mean(d)  so that dL + dR + dN = 0 each step.

    Beta dynamics:
      Default: random walk in log-space so beta stays positive.
      Optional: add OU-like mean reversion (tau_logbeta + beta0) and rare jumps.
    """

    # --- Bias mixture parameters ---
    mu_L: float = 0.0
    mu_R: float = 0.0
    mu_N: float = 1e-4          # selectable small non-zero drift for N increment (early)
    sigma_L: float = 0.02
    sigma_R: float = 0.02
    sigma_N: float = 1e-3       # very small noise for N increment
    tau_bias: float = 600.0     # time constant in trials; w(t)=exp(-t/tau_bias)

    # --- Beta increments in log-space ---
    mu_logbeta: float = 0.0
    sigma_logbeta: float = 0.01

    # Optional mean reversion for log(beta)
    tau_logbeta: Optional[float] = None   # if set, do OU-like pull to log(beta0)
    beta0: Optional[float] = None         # target level for mean reversion

    # Optional rare jumpiness for step-like behavior
    p_beta_jump: float = 0.0
    sigma_logbeta_jump: float = 0.25

    # Safety clamp for beta
    beta_min: float = 1e-3
    beta_max: float = 50.0


def _mix_normal(
    rng: np.random.Generator,
    *,
    w: float,
    mu: float,
    sigma: float,
) -> float:
    """
    Sample from mixture: w*N(mu, sigma^2) + (1-w)*N(0, sigma^2).
    """
    w = float(np.clip(w, 0.0, 1.0))
    if rng.random() < w:
        return float(rng.normal(loc=mu, scale=sigma))
    return float(rng.normal(loc=0.0, scale=sigma))


def rw_update_biases_sum0(
    bL: float,
    bR: float,
    bN: float,
    rw: DynamicRWParams,
    rng: np.random.Generator,
    *,
    t: int,
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Update biases with mixture-drift-to-zero increments, then project to sum-to-zero:

      raw d_a ~ w(t) N(mu_a, sigma_a^2) + (1-w(t)) N(0, sigma_a^2)
      d <- d - mean(d) so dL+dR+dN=0.

    Returns:
      (bL_new, bR_new, bN_new, dL, dR, dN, w_bias)
    """
    tau = float(rw.tau_bias)
    w = float(np.exp(-float(t) / tau)) if (tau is not None and tau > 0) else 1.0

    dL_raw = _mix_normal(rng, w=w, mu=rw.mu_L, sigma=rw.sigma_L)
    dR_raw = _mix_normal(rng, w=w, mu=rw.mu_R, sigma=rw.sigma_R)
    dN_raw = _mix_normal(rng, w=w, mu=rw.mu_N, sigma=rw.sigma_N)

    m = (dL_raw + dR_raw + dN_raw) / 3.0
    dL = dL_raw - m
    dR = dR_raw - m
    dN = dN_raw - m

    return bL + dL, bR + dR, bN + dN, float(dL), float(dR), float(dN), float(w)


def rw_update_beta_logspace(
    beta: float,
    rw: DynamicRWParams,
    rng: np.random.Generator,
    *,
    t: int,
) -> float:
    """
    Random walk on log(beta), with optional mean reversion and optional rare jumps.
    """
    beta = max(float(beta), rw.beta_min)
    logb = float(np.log(beta))

    # OU-like mean reversion (if configured)
    if rw.tau_logbeta is not None and rw.beta0 is not None and rw.tau_logbeta > 0:
        target = float(np.log(max(float(rw.beta0), rw.beta_min)))
        logb += (target - logb) / float(rw.tau_logbeta)

    # baseline RW increment
    logb += float(rng.normal(loc=rw.mu_logbeta, scale=rw.sigma_logbeta))

    # occasional bigger jump -> step-like variability
    if rw.p_beta_jump > 0.0 and rng.random() < float(rw.p_beta_jump):
        logb += float(rng.normal(loc=0.0, scale=rw.sigma_logbeta_jump))

    beta_new = float(np.exp(logb))
    beta_new = min(max(beta_new, rw.beta_min), rw.beta_max)
    return beta_new


# =========================================
# Simulator 1: Constant parameters (fixed)
# =========================================

class ConstantQLearnerReversalSimulator:
    """
    Q-learning + softmax policy with optional reversal trigger.

    - Q update: Q[a,s] <- (1-alpha) Q[a,s] + alpha * reward
    - Policy: softmax(beta*Q[a,s] + bias_a)
    - Reversal trigger: if last perf_window trials have >= perf_threshold correct, start countdown.
      When countdown hits 0, flip reward_probs (N stays 0).
    """

    def __init__(
        self,
        agent: AgentParams,
        task: TaskSpec,
        n_trials: int,
        *,
        seed: int = 0,
        reversal_countdown: int = 250,
        perf_window: int = 20,
        perf_threshold: int = 19,
        allow_multiple_reversals: bool = False,
    ):
        self.agent = agent
        self.task = task
        self.n_trials = int(n_trials)

        self.seed = int(seed)
        self.reversal_countdown_init = int(reversal_countdown)
        self.perf_window = int(perf_window)
        self.perf_threshold = int(perf_threshold)
        self.allow_multiple_reversals = bool(allow_multiple_reversals)

        self._rng = random.Random(self.seed)

        self._bias_map = {
            "L": float(agent.bias_L),
            "R": float(agent.bias_R),
            "N": float(agent.bias_N),
        }

        # mutable copies for reversal logic
        self.reward_probs = dict(task.reward_probs)
        self.correct_pairs = dict(task.correct_pairs)

        self.action_space = [str(a).upper() for a in task.action_space]
        self.state_space = [str(s) for s in task.state_space]

    def _init_q(self) -> Dict[Tuple[str, str], float]:
        return {(a, s): 0.0 for a in self.action_space for s in self.state_space}

    def _softmax_probs(self, Q: Dict[Tuple[str, str], float], state: str) -> np.ndarray:
        u = np.array(
            [self.agent.beta * float(Q[(a, state)]) + self._bias_map.get(a, 0.0) for a in self.action_space],
            dtype=float,
        )
        u -= u.max()
        e = np.exp(u)
        return e / (e.sum() + 1e-12)

    def _sample_reward(self, state: str, action: str) -> int:
        p = float(self.reward_probs.get((state, action), 0.0))
        return 1 if self._rng.random() < p else 0

    def _is_correct(self, state: str, action: str) -> int:
        return 1 if int(self.correct_pairs.get((state, action), 0)) == 1 else 0

    def _apply_reversal(self):
        self.reward_probs = reverse_reward_probs(self.reward_probs, self.action_space)
        self.correct_pairs = make_correct_pairs_from_reward_probs(self.reward_probs, threshold=0.5)

    def run(self) -> pd.DataFrame:
        Q = self._init_q()

        rows = []
        last_k_correct: List[int] = []
        reversal_armed = False
        countdown = None

        reversal_events: List[int] = []
        rule = 0

        for t in range(self.n_trials):
            state = self._rng.choice(self.state_space)
            probs = self._softmax_probs(Q, state)
            action = self._rng.choices(self.action_space, weights=probs, k=1)[0]

            # policy probs (useful for analysis plots)
            pmap = {a: float(probs[i]) for i, a in enumerate(self.action_space)}

            reward = self._sample_reward(state, action)
            Q[(action, state)] = (1.0 - self.agent.alpha) * Q[(action, state)] + self.agent.alpha * float(reward)

            corr = self._is_correct(state, action)
            last_k_correct.append(corr)
            if len(last_k_correct) > self.perf_window:
                last_k_correct.pop(0)

            # reversal trigger
            if (not reversal_armed) and (len(last_k_correct) == self.perf_window) and (sum(last_k_correct) >= self.perf_threshold):
                reversal_armed = True
                countdown = self.reversal_countdown_init
                reversal_events.append(t)

            # countdown and flip
            if reversal_armed and countdown is not None:
                countdown -= 1
                if countdown == 0:
                    self._apply_reversal()
                    rule += 1

                    # allow new reversals after this flip
                    reversal_armed = False
                    countdown = None

                    # IMPORTANT: reset performance buffer so the next reversal requires a new 20-trial criterion
                    last_k_correct = []

                    # OPTIONAL SAFETY: if you truly want only one reversal, stop arming forever:
                    if not self.allow_multiple_reversals:
                        # simplest: set a guard variable
                        self._stop_reversals = True

            rows.append(
                {
                    "t": t,
                    "state": state,
                    "action": action,
                    "reward": int(reward),
                    "correct": int(corr),
                    "rule": int(rule),
                    "reversal_triggered": 1 if (len(reversal_events) > 0 and reversal_events[-1] == t) else 0,
                    "pi_L": pmap.get("L", np.nan),
                    "pi_R": pmap.get("R", np.nan),
                    "pi_N": pmap.get("N", np.nan),
                }
            )

        df = pd.DataFrame(rows)
        df["reversal_t"] = df.index[df["reversal_triggered"] == 1].min() if (df["reversal_triggered"] == 1).any() else np.nan
        return df


# =========================================
# Simulator 2: Dynamic parameters (RW)
# =========================================

class DynamicQLearnerReversalSimulator:
    """
    Q-learning + softmax policy with reversal trigger, BUT:

    - bias_L(t), bias_R(t), bias_N(t) evolve via drift-to-zero mixture increments,
      projected each step to satisfy dL+dR+dN=0.
    - beta(t) evolves in log-space (optionally mean-reverting + jumpy).
    """

    def __init__(
        self,
        agent: AgentParams,
        task: TaskSpec,
        n_trials: int,
        *,
        seed: int = 0,
        rw_params: Optional[DynamicRWParams] = None,
        reversal_countdown: int = 250,
        perf_window: int = 20,
        perf_threshold: int = 19,
        allow_multiple_reversals: bool = False,
    ):
        self.agent = agent
        self.task = task
        self.n_trials = int(n_trials)

        self.seed = int(seed)
        self.rw = rw_params if rw_params is not None else DynamicRWParams()

        self.reversal_countdown_init = int(reversal_countdown)
        self.perf_window = int(perf_window)
        self.perf_threshold = int(perf_threshold)
        self.allow_multiple_reversals = bool(allow_multiple_reversals)

        # RNGs
        self._py_rng = random.Random(self.seed)
        self._np_rng = np.random.default_rng(self.seed)

        # mutable copies for reversal logic
        self.reward_probs = dict(task.reward_probs)
        self.correct_pairs = dict(task.correct_pairs)

        self.action_space = [str(a).upper() for a in task.action_space]
        self.state_space = [str(s) for s in task.state_space]

        # dynamic state (initial conditions from agent)
        self._bL = float(agent.bias_L)
        self._bR = float(agent.bias_R)
        self._bN = float(agent.bias_N)
        self._beta_t = float(agent.beta)

    def _init_q(self) -> Dict[Tuple[str, str], float]:
        return {(a, s): 0.0 for a in self.action_space for s in self.state_space}

    def _softmax_probs(self, Q: Dict[Tuple[str, str], float], state: str) -> np.ndarray:
        bias_map = {"L": self._bL, "R": self._bR, "N": self._bN}
        u = np.array(
            [self._beta_t * float(Q[(a, state)]) + float(bias_map.get(a, 0.0)) for a in self.action_space],
            dtype=float,
        )
        u -= u.max()
        e = np.exp(u)
        return e / (e.sum() + 1e-12)

    def _sample_reward(self, state: str, action: str) -> int:
        p = float(self.reward_probs.get((state, action), 0.0))
        return 1 if self._py_rng.random() < p else 0

    def _is_correct(self, state: str, action: str) -> int:
        return 1 if int(self.correct_pairs.get((state, action), 0)) == 1 else 0

    def _apply_reversal(self):
        self.reward_probs = reverse_reward_probs(self.reward_probs, self.action_space)
        self.correct_pairs = make_correct_pairs_from_reward_probs(self.reward_probs, threshold=0.5)

    def run(self) -> pd.DataFrame:
        Q = self._init_q()

        rows = []
        last_k_correct: List[int] = []
        reversal_armed = False
        countdown = None

        reversal_events: List[int] = []
        rule = 0

        for t in range(self.n_trials):
            # values used THIS trial (before updates)
            beta_used = float(self._beta_t)
            bL_used = float(self._bL)
            bR_used = float(self._bR)
            bN_used = float(self._bN)

            state = self._py_rng.choice(self.state_space)
            probs = self._softmax_probs(Q, state)
            action = self._py_rng.choices(self.action_space, weights=probs, k=1)[0]

            pmap = {a: float(probs[i]) for i, a in enumerate(self.action_space)}

            reward = self._sample_reward(state, action)
            Q[(action, state)] = (1.0 - self.agent.alpha) * Q[(action, state)] + self.agent.alpha * float(reward)

            corr = self._is_correct(state, action)
            last_k_correct.append(corr)
            if len(last_k_correct) > self.perf_window:
                last_k_correct.pop(0)

            # reversal trigger
            if (not reversal_armed) and (len(last_k_correct) == self.perf_window) and (sum(last_k_correct) >= self.perf_threshold):
                reversal_armed = True
                countdown = self.reversal_countdown_init
                reversal_events.append(t)

            # countdown and flip
            if reversal_armed and countdown is not None:
                countdown -= 1
                if countdown == 0:
                    self._apply_reversal()
                    rule += 1

                    # allow new reversals after this flip
                    reversal_armed = False
                    countdown = None

                    # IMPORTANT: reset performance buffer so the next reversal requires a new 20-trial criterion
                    last_k_correct = []

                    # OPTIONAL SAFETY: if you truly want only one reversal, stop arming forever:
                    if not self.allow_multiple_reversals:
                        # simplest: set a guard variable
                        self._stop_reversals = True


            # dynamic updates (apply AFTER recording "used" values)
            self._bL, self._bR, self._bN, dL, dR, dN, w_bias = rw_update_biases_sum0(
                self._bL, self._bR, self._bN, self.rw, self._np_rng, t=t
            )
            self._beta_t = rw_update_beta_logspace(self._beta_t, self.rw, self._np_rng, t=t)

            rows.append(
                {
                    "t": t,
                    "state": state,
                    "action": action,
                    "reward": int(reward),
                    "correct": int(corr),
                    "rule": int(rule),
                    "reversal_triggered": 1 if (len(reversal_events) > 0 and reversal_events[-1] == t) else 0,

                    # policy probs
                    "pi_L": pmap.get("L", np.nan),
                    "pi_R": pmap.get("R", np.nan),
                    "pi_N": pmap.get("N", np.nan),

                    # dynamic trajectories used at this trial
                    "beta_t": beta_used,
                    "bias_L_t": bL_used,
                    "bias_R_t": bR_used,
                    "bias_N_t": bN_used,

                    # increments applied after this trial
                    "d_bias_L": float(dL),
                    "d_bias_R": float(dR),
                    "d_bias_N": float(dN),
                    "w_bias": float(w_bias),
                }
            )

        df = pd.DataFrame(rows)
        df["reversal_t"] = df.index[df["reversal_triggered"] == 1].min() if (df["reversal_triggered"] == 1).any() else np.nan
        return df


# =========================
# One-call convenience funcs
# =========================

def simulate_reversal_session(
    *,
    seed: int,
    n_trials: int,
    agent: AgentParams,
    task: TaskSpec,
    reversal_countdown: int = 250,
    perf_window: int = 20,
    perf_threshold: int = 19,
    allow_multiple_reversals: bool = True,   # <-- ADD
) -> pd.DataFrame:
    sim = ConstantQLearnerReversalSimulator(
        agent=agent,
        task=task,
        n_trials=n_trials,
        seed=seed,
        reversal_countdown=reversal_countdown,
        perf_window=perf_window,
        perf_threshold=perf_threshold,
        allow_multiple_reversals=allow_multiple_reversals,  # <-- PASS THROUGH
    )
    return sim.run()



def simulate_dynamic_reversal_session(
    *,
    seed: int,
    n_trials: int,
    agent: AgentParams,
    task: TaskSpec,
    rw_params: Optional[DynamicRWParams] = None,
    reversal_countdown: int = 250,
    perf_window: int = 20,
    perf_threshold: int = 19,
    allow_multiple_reversals: bool = True,   # <-- ADD
) -> pd.DataFrame:
    sim = DynamicQLearnerReversalSimulator(
        agent=agent,
        task=task,
        n_trials=n_trials,
        seed=seed,
        rw_params=rw_params,
        reversal_countdown=reversal_countdown,
        perf_window=perf_window,
        perf_threshold=perf_threshold,
        allow_multiple_reversals=allow_multiple_reversals,  # <-- PASS THROUGH
    )
    return sim.run()
