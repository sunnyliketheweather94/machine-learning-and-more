from dataclasses import dataclass, replace
from itertools import groupby
from operator import itemgetter
from typing import Generic, Iterator

from rl.constants import ACTION, STATE
from rl.distributions import FiniteDistribution
from rl.dynamic_programming import VALUE_FUNCTION, extended_vf
from rl.markov_decision_process import (
    ActionMapping,
    FiniteMarkovDecisionProcess,
    StateActionMapping,
)
from rl.markov_process import FiniteMarkovRewardProcess, RewardTransition, StateReward
from rl.policy import FiniteDeterministicPolicy
from rl.states import NonTerminal, State, Terminal


@dataclass(frozen=True)
class WithTime(Generic[STATE]):
    state: STATE
    time: int = 0

    def step_time(self):
        return replace(self, time=self.time + 1)


RewardOutCome = FiniteDistribution[tuple[WithTime[STATE], float]]


def construct_finite_horizon_mrp(
    process: FiniteMarkovRewardProcess[STATE],
    limit: int,
) -> FiniteMarkovRewardProcess[WithTime[STATE]]:
    transition_map: dict[WithTime[STATE], RewardOutCome] = dict()

    for time in range(limit):
        for state in process.non_terminal_states:
            result: StateReward[STATE] = process.transition_reward(state=state)
            current_state = WithTime(state=state.state, time=time)

            transition_map[current_state] = result.map(
                lambda sr: (
                    WithTime(state=sr[0].state, time=time + 1),
                    sr[1],
                )
            )

    return FiniteMarkovRewardProcess(transition_map)


def deconstruct_finite_horizon_mrp(
    process: FiniteMarkovRewardProcess[WithTime[STATE]],
) -> list[RewardTransition[STATE]]:
    def _extract_time(state: WithTime[STATE]) -> int:
        return state.time

    def _create_next_state_reward_without_time(
        s_r: tuple[State[WithTime[STATE]], float],
    ) -> tuple[State[STATE], float]:
        next_state, reward = s_r

        if isinstance(next_state, NonTerminal):
            pair = (NonTerminal(next_state.state.state), reward)

        else:
            pair = (Terminal(next_state.state.state), reward)

        return pair

    def _remove_time(arg: StateReward[WithTime[STATE]]) -> StateReward[STATE]:
        return arg.map(_create_next_state_reward_without_time)

    return [
        {
            NonTerminal(state.state): _remove_time(
                process.transition_reward(NonTerminal(state))
            )
            for state in states
        }
        for _, states in groupby(
            sorted((nt.state for nt in process.non_terminal_states), key=_extract_time),
            key=_extract_time,
        )
    ]


def compute_vf_at_each_time_step(
    steps: list[RewardTransition[STATE]],
    gamma: float,
) -> Iterator[VALUE_FUNCTION[STATE]]:
    v: list[VALUE_FUNCTION[STATE]] = list()

    for step in reversed(steps):
        v.append(
            {
                state: res.expectation(
                    lambda s_r: s_r[1]
                    + gamma * (extended_vf(v[-1], s=s_r[0]) if len(v) > 0 else 0.0)
                )
                for state, res in step.items()
            }
        )

    return reversed(v)


def construct_finite_horizon_mdp(
    process: FiniteMarkovDecisionProcess[STATE, ACTION],
    limit: int,
) -> FiniteMarkovDecisionProcess[WithTime[STATE], ACTION]:
    mapping: dict[
        WithTime[STATE],
        dict[
            ACTION,
            FiniteDistribution[
                tuple[
                    WithTime[STATE],
                    float,
                ]
            ],
        ],
    ] = dict()

    for time in range(limit):
        for state in process.non_terminal_states:
            current_state = WithTime(state=state.state, time=time)

            mapping[current_state] = {
                action: result.map(
                    lambda sr: (WithTime(state=sr[0].state, time=time + 1), sr[1])
                )
                for action, result in process.mapping[state].items()
            }

    return FiniteMarkovDecisionProcess(mapping)


def deconstruct_finite_horizon_mdp(
    process: FiniteMarkovDecisionProcess[WithTime[STATE], ACTION],
) -> list[StateActionMapping[STATE, ACTION]]:
    def _extract_time(state: WithTime[STATE]) -> int:
        return state.time

    def _create_next_state_reward_without_time(
        s_r: tuple[State[WithTime[STATE]], float],
    ) -> tuple[State[STATE], float]:
        next_state, reward = s_r

        if isinstance(next_state, NonTerminal):
            pair = (NonTerminal(next_state.state.state), reward)

        else:
            pair = (Terminal(next_state.state.state), reward)

        return pair

    def _remove_time(
        arg: ActionMapping[ACTION, WithTime[STATE]],
    ) -> ActionMapping[ACTION, STATE]:
        return {
            action: sr_dist.map(_create_next_state_reward_without_time)
            for action, sr_dist in arg.items()
        }

    return [
        {
            NonTerminal(state.state): _remove_time(process.mapping[NonTerminal(state)])
            for state in states
        }
        for _, states in groupby(
            sorted((nt.state for nt in process.non_terminal_states), key=_extract_time),
            key=_extract_time,
        )
    ]


def calculate_optimal_vf_and_policies(
    steps: list[StateActionMapping[STATE, ACTION]],
    gamma: float,
) -> Iterator[tuple[VALUE_FUNCTION[STATE], FiniteDeterministicPolicy[STATE, ACTION]]]:
    v_p: list[
        tuple[VALUE_FUNCTION[STATE], FiniteDeterministicPolicy[STATE, ACTION]]
    ] = []

    for step in reversed(steps):
        current_vf: dict[NonTerminal[STATE], float] = {}
        current_actions: dict[STATE, ACTION] = {}

        for state, actions_map in step.items():
            action_values = (
                (
                    sr_dist.expectation(
                        lambda s_r: s_r[1]
                        + gamma
                        * (extended_vf(v=v_p[-1][0], s=s_r[0]) if len(v_p) > 0 else 0.0)
                    ),
                    action,
                )
                for action, sr_dist in actions_map.items()
            )

            v_star, a_star = max(action_values, key=itemgetter(0))
            current_vf[state] = v_star
            current_actions[state] = a_star

        v_p.append((current_vf, FiniteDeterministicPolicy(current_actions)))

    return reversed(v_p)
