import operator
from typing import Iterator

import numpy as np

from rl.constants import ACTION, STATE
from rl.distributions import Categorical, Choose
from rl.iterate import converged, iterate
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.markov_process import FiniteMarkovRewardProcess
from rl.policy import FiniteDeterministicPolicy, FinitePolicy
from rl.states import NonTerminal, State

DEFAULT_TOLERANCE = 1e-5
VALUE_FUNCTION = dict[NonTerminal[STATE], float]


def extended_vf(v: VALUE_FUNCTION[STATE], s: State[STATE]) -> float:
    def _non_terminal_vf(st: NonTerminal[STATE], v=v) -> float:
        return v[st]

    return s.on_on_terminal(_non_terminal_vf, 0.0)


def evaluate_mrp(
    mrp: FiniteMarkovRewardProcess[STATE],
    gamma: float,
) -> Iterator[np.ndarray]:
    transition_matrix: np.ndarray = mrp.compute_transition_matrix()

    def _update(v: np.ndarray) -> np.ndarray:
        return mrp.reward_function_vector + gamma * transition_matrix @ v

    v_0: np.ndarray = np.zeros(len(mrp.non_terminal_states))
    return iterate(step=_update, start=v_0)


def almost_equal_numpy_arrays(
    left: np.ndarray,
    right: np.ndarray,
    tolerance: float = DEFAULT_TOLERANCE,
) -> bool:
    return max(abs(left - right)) < tolerance


def evaluate_mrp_result(
    mrp: FiniteMarkovRewardProcess[STATE],
    gamma: float,
) -> VALUE_FUNCTION[STATE]:
    v_star: np.ndarray = converged(
        values=evaluate_mrp(mrp=mrp, gamma=gamma),
        done=almost_equal_numpy_arrays,
    )

    return {state: value for state, value in zip(mrp.non_terminal_states, v_star)}


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


def policy_iteration(
    mdp: FiniteMarkovDecisionProcess[STATE, ACTION],
    gamma: float,
    matrix_method_for_mrp_eval: bool = False,
) -> Iterator[tuple[VALUE_FUNCTION[STATE], FinitePolicy[STATE, ACTION]]]:
    def _update(
        vf_policy: tuple[VALUE_FUNCTION[STATE], FinitePolicy[STATE, ACTION]],
    ) -> tuple[VALUE_FUNCTION[STATE], FiniteDeterministicPolicy[STATE, ACTION]]:
        _, policy = vf_policy
        mrp: FiniteMarkovRewardProcess[STATE] = mdp.apply_finite_policy(policy)

        improved_vf: VALUE_FUNCTION[STATE] = (
            {
                mrp.non_terminal_states[i]: v
                for i, v in enumerate(mrp.compute_value_function_vector(gamma))
            }
            if matrix_method_for_mrp_eval
            else evaluate_mrp_result(mrp=mrp, gamma=gamma)
        )

        improved_policy: FiniteDeterministicPolicy[STATE, ACTION] = (
            greedy_policy_from_vf(mdp=mdp, vf=improved_vf, gamma=gamma)
        )

        return improved_vf, improved_policy

    v_0: VALUE_FUNCTION[STATE] = {s: 0.0 for s in mdp.non_terminal_states}

    policy_0: FinitePolicy[STATE, ACTION] = FinitePolicy(
        {s.state: Choose(mdp.actions(s)) for s in mdp.non_terminal_states}
    )

    return iterate(step=_update, start=(v_0, policy_0))


def almost_equal_vf_pis(
    left: tuple[VALUE_FUNCTION[STATE], FinitePolicy[STATE, ACTION]],
    right: tuple[VALUE_FUNCTION[STATE], FinitePolicy[STATE, ACTION]],
    tolerance: float = DEFAULT_TOLERANCE,
) -> bool:
    left_vf, _ = left
    right_vf, _ = right
    return max(abs(left_vf[state] - right_vf[state]) for state in left_vf) < tolerance


def policy_iteration_result(
    mdp: FiniteMarkovDecisionProcess[STATE, ACTION],
    gamma: float,
) -> tuple[VALUE_FUNCTION[STATE], FiniteDeterministicPolicy[STATE, ACTION]]:
    return converged(policy_iteration(mdp=mdp, gamma=gamma), done=almost_equal_vf_pis)
