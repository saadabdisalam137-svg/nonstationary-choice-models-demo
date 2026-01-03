# src/nonstationary_demo/simulate.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from collections import defaultdict
import random
import numpy as np
import pandas as pd


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
    """A simple convention: (s,a) is 'correct' if P(reward|s,a) > threshold."""
    correct = {}
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
    flipped = {}
    for (s, a), p in reward_probs.items():
        if str(a).upper() == "N":
            flipped[(s, a)] = 0.0
        else:
            flipped[(s, a)] = 1.0 - float(p)
    # Ensure every (s, a) exists (optional, but safer)
    for s in {k[0] for k in reward_probs}:
        for a in action_space:
            flipped.setdefault((s, a), 0.0 if a.upper() == "N" else 0.5)
    return flipped


@dataclass(frozen=True)
class AgentParams:
    """
    Simulation-only parameters (not estimated in Repo 1).
    """
    alpha: float
    beta: float
    bias_L: float = 0.0
    bias_R: float = 0.0
    bias_N: float = 0.0


class ConstantQLearnerReversalSimulator:
    """
    Simulation-only Q-learning + softmax policy with optional reversal trigger.

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
        np.random.seed(self.seed)

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
        self._a2i = {a: i for i, a in enumerate(self.action_space)}

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
        rule = 0  # 0 before flip, 1 after flip, 2 after second flip, etc.

        for t in range(self.n_trials):
            state = self._rng.choice(self.state_space)
            probs = self._softmax_probs(Q, state)
            action = self._rng.choices(self.action_space, weights=probs, k=1)[0]

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
                reversal_events.append(t)  # index of trigger
            # countdown and flip
            if reversal_armed and countdown is not None:
                countdown -= 1
                if countdown == 0:
                    self._apply_reversal()
                    rule += 1
                    reversal_armed = False if not self.allow_multiple_reversals else False
                    countdown = None

            rows.append(
                {
                    "t": t,
                    "state": state,
                    "action": action,
                    "reward": int(reward),
                    "correct": int(corr),
                    "rule": int(rule),
                    "reversal_triggered": 1 if (len(reversal_events) > 0 and reversal_events[-1] == t) else 0,
                }
            )

        df = pd.DataFrame(rows)
        # convenience marker: first reversal trigger t (if any)
        df["reversal_t"] = df.index[df["reversal_triggered"] == 1].min() if (df["reversal_triggered"] == 1).any() else np.nan
        return df


def simulate_reversal_session(
    *,
    seed: int,
    n_trials: int,
    agent: AgentParams,
    task: TaskSpec,
    reversal_countdown: int = 250,
    perf_window: int = 20,
    perf_threshold: int = 19,
) -> pd.DataFrame:
    """
    One-call function your notebook can use.
    """
    sim = ConstantQLearnerReversalSimulator(
        agent=agent,
        task=task,
        n_trials=n_trials,
        seed=seed,
        reversal_countdown=reversal_countdown,
        perf_window=perf_window,
        perf_threshold=perf_threshold,
        allow_multiple_reversals=False,
    )
    return sim.run()
