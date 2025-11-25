from typing import Iterator

import numpy as np

from rl.constants import ACTION, STATE
from rl.distributions import Choose
from rl.dynamic_programming import (
    VALUE_FUNCTION,
    almost_equal_numpy_arrays,
    almost_equal_vf_pis,
    greedy_policy_from_vf,
)
from rl.iterate import converged, iterate
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.markov_process import FiniteMarkovRewardProcess
from rl.policy import FiniteDeterministicPolicy, FinitePolicy


def evaluate_mrp(
    mrp: FiniteMarkovRewardProcess[STATE],
    gamma: float,
) -> Iterator[np.ndarray]:
    transition_matrix: np.ndarray = mrp.compute_transition_matrix()

    def _update(v: np.ndarray) -> np.ndarray:
        return mrp.reward_function_vector + gamma * transition_matrix @ v

    v_0: np.ndarray = np.zeros(len(mrp.non_terminal_states))
    return iterate(step=_update, start=v_0)


def evaluate_mrp_result(
    mrp: FiniteMarkovRewardProcess[STATE],
    gamma: float,
) -> VALUE_FUNCTION[STATE]:
    v_star: np.ndarray = converged(
        values=evaluate_mrp(mrp=mrp, gamma=gamma),
        done=almost_equal_numpy_arrays,
    )

    return {state: value for state, value in zip(mrp.non_terminal_states, v_star)}


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


def policy_iteration_result(
    mdp: FiniteMarkovDecisionProcess[STATE, ACTION],
    gamma: float,
) -> tuple[VALUE_FUNCTION[STATE], FiniteDeterministicPolicy[STATE, ACTION]]:
    return converged(policy_iteration(mdp=mdp, gamma=gamma), done=almost_equal_vf_pis)
