import os
import sys

sys.path.append(
    os.path.join(
        os.path.abspath(__file__),
        os.pardir,
        os.pardir,
        os.pardir,
    )
)
import operator
from typing import Iterator

import numpy as np

from rl.constants import ACTION, STATE
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.policy import FiniteDeterministicPolicy, FinitePolicy
from rl.states import NonTerminal, State

DEFAULT_TOLERANCE = 1e-5
VALUE_FUNCTION = dict[NonTerminal[STATE], float]


def extended_vf(v: VALUE_FUNCTION[STATE], s: State[STATE]) -> float:
    def _non_terminal_vf(st: NonTerminal[STATE], v=v) -> float:
        return v[st]

    return s.on_on_terminal(_non_terminal_vf, 0.0)


def almost_equal_numpy_arrays(
    left: np.ndarray,
    right: np.ndarray,
    tolerance: float = DEFAULT_TOLERANCE,
) -> bool:
    return max(abs(left - right)) < tolerance


def greedy_policy_from_vf(
    mdp: FiniteMarkovDecisionProcess[STATE, ACTION],
    vf: VALUE_FUNCTION[STATE],
    gamma: float,
) -> FiniteDeterministicPolicy[STATE, ACTION]:
    greedy_policy: dict[STATE, ACTION] = dict()

    for state in mdp.non_terminal_states:
        q_values: Iterator[tuple[ACTION, float]] = (
            (
                action,
                mdp.mapping[state][action].expectation(
                    lambda next_state, reward: reward
                    + gamma * extended_vf(v=vf, s=next_state)
                ),
            )
            for action in mdp.actions(state)
        )

        greedy_policy[state.state] = max(q_values, key=operator.itemgetter(1))[0]

    return FiniteDeterministicPolicy(action_for=greedy_policy)


def almost_equal_vf_pis(
    left: tuple[VALUE_FUNCTION[STATE], FinitePolicy[STATE, ACTION]],
    right: tuple[VALUE_FUNCTION[STATE], FinitePolicy[STATE, ACTION]],
    tolerance: float = DEFAULT_TOLERANCE,
) -> bool:
    left_vf, _ = left
    right_vf, _ = right
    return max(abs(left_vf[state] - right_vf[state]) for state in left_vf) < tolerance
