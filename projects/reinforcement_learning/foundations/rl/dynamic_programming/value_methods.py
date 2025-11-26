from typing import Iterator

from rl.constants import ACTION, STATE
from rl.dynamic_programming import (
    VALUE_FUNCTION,
    almost_equal_dictionaries,
    extended_vf,
    greedy_policy_from_vf,
)
from rl.iterate import converged, iterate
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.policy import FiniteDeterministicPolicy


def value_iteration(
    mdp: FiniteMarkovDecisionProcess[STATE, ACTION],
    gamma: float,
) -> Iterator[VALUE_FUNCTION[STATE]]:
    def _update(v: VALUE_FUNCTION[STATE]) -> VALUE_FUNCTION[STATE]:
        return {
            state: max(
                mdp.mapping[state][action].expectation(
                    lambda next_state_reward: next_state_reward[1]
                    + gamma * extended_vf(v=v, s=next_state_reward[0])
                )
                for action in mdp.actions(state)
            )
            for state in v
        }

    v_0: VALUE_FUNCTION[STATE] = {s: 0.0 for s in mdp.non_terminal_states}
    return iterate(step=_update, start=v_0)


def value_iteration_result(
    mdp: FiniteMarkovDecisionProcess[STATE, ACTION],
    gamma: float,
) -> tuple[VALUE_FUNCTION[STATE], FiniteDeterministicPolicy[STATE, ACTION]]:
    optimal_vf: VALUE_FUNCTION[STATE] = converged(
        values=value_iteration(mdp=mdp, gamma=gamma),
        done=almost_equal_dictionaries,
    )

    optimal_policy: FiniteDeterministicPolicy[STATE, ACTION] = greedy_policy_from_vf(
        mdp=mdp,
        vf=optimal_vf,
        gamma=gamma,
    )

    return optimal_vf, optimal_policy
