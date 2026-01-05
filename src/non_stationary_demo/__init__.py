# src/nonstationary_demo/__init__.py

from .simulate import (
    AgentParams,
    TaskSpec,
    make_correct_pairs_from_reward_probs,
    reverse_reward_probs,
    simulate_reversal_session,
    ConstantQLearnerReversalSimulator,
)

from .analysis import (
    compute_rolling_summaries,
    reversal_align,
    plot_learning_curves,
    plot_reversal_aligned,
)

__all__ = [
    "AgentParams",
    "TaskSpec",
    "make_correct_pairs_from_reward_probs",
    "reverse_reward_probs",
    "simulate_reversal_session",
    "ConstantQLearnerReversalSimulator",
    "compute_rolling_summaries",
    "reversal_align",
    "plot_learning_curves",
    "plot_reversal_aligned",
]
