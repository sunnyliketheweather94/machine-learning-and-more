from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Generic, Iterable, Mapping, Sequence, Set, TypeVar

import numpy as np

from rl.distributions import (
    Categorical,
    Distribution,
    FiniteDistribution,
    SampledDistribution,
)

S = TypeVar("S")
X = TypeVar("X")


class State(ABC, Generic[S]):
    state: S

    def on_on_terminal(self, f: Callable, default: X) -> X:
        """f takes a "NonTerminal[S]" arg and returns X"""
        if isinstance(self, NonTerminal):
            return f(self)

        return default


@dataclass(frozen=True)
class Terminal(State[S]):
    state: S


@dataclass(frozen=True)
class NonTerminal(State[S]):
    state: S


class MarkovProcess(ABC, Generic[S]):
    @abstractmethod
    def transition(self, state: NonTerminal[S]) -> Distribution[State[S]]:
        pass

    def simulate(
        self,
        start_state_distribution: Distribution[NonTerminal[S]],
    ) -> Iterable[State[S]]:
        state: State[S] = start_state_distribution.sample()
        yield state

        while isinstance(state, NonTerminal):
            state = self.transition(state=state).sample()
            yield state


Transition = Mapping[NonTerminal[S], FiniteDistribution[State[S]]]


class FiniteMarkovProcess(MarkovProcess[S]):
    non_terminal_states: Sequence[NonTerminal[S]]
    transition_map: Transition[S]

    def __init__(self, transition_map: Mapping[S, FiniteDistribution[S]]):
        non_terminals: Set[S] = set(transition_map.keys())

        self.transition_map = {
            NonTerminal(state=source_state): Categorical(
                {
                    (
                        NonTerminal(destination_state)
                        if destination_state in non_terminals
                        else Terminal(destination_state)
                    ): prob
                    for destination_state, prob in dist
                }
            )
            for source_state, dist in transition_map.items()
        }

        self.non_terminal_states = list(self.transition_map.keys())

    def __repr__(self) -> str:
        display = ""

        for source_state, dist in self.transition_map.items():
            display += f"From State {source_state.state}:\n"
            for dest_state, prob in dist:
                opt = "Terminal " if isinstance(dest_state, Terminal) else ""
                display += (
                    f"\tTo {opt}State {dest_state.state} with Probability {prob:.3f}\n"
                )

        return display

    def transition(self, state: NonTerminal[S]) -> FiniteDistribution[State[S]]:
        return self.transition_map[state]

    def compute_transition_matrix(self) -> np.ndarray:
        num_states = len(self.non_terminal_states)
        trans_matrix = np.zeros((num_states, num_states))

        for row, source in enumerate(self.non_terminal_states):
            for col, dest in enumerate(self.non_terminal_states):
                trans_matrix[row, col] = self.transition(state=source).probability(
                    outcome=dest
                )

        return trans_matrix

    def compute_stationary_distribution(self) -> FiniteDistribution[S]:
        eigen_vals, eigen_vecs = np.linalg.eig(self.compute_transition_matrix().T)
        index_of_unit_eigenvalue = np.argmin(np.abs(eigen_vals - 1))
        unit_eigenvector = eigen_vecs[:, index_of_unit_eigenvalue].real

        return Categorical(
            {
                self.non_terminal_states[i].state: prob.item()
                for i, prob in enumerate(unit_eigenvector)
            }
        )


@dataclass(frozen=True)
class TransitionStep(Generic[S]):
    state: NonTerminal[S]
    next_state: State[S]
    reward: float


class MarkovRewardProcess(MarkovProcess[S]):
    @abstractmethod
    def transition_reward(
        self,
        state: NonTerminal[S],
    ) -> Distribution[tuple[State[S], float]]:
        pass

    def transition(self, state: NonTerminal[S]) -> Distribution[State[S]]:
        distribution = self.transition_reward(state=state)

        def sample_next_state(
            distribution: Distribution[tuple[State[S], float]] = distribution,
        ):
            next_state, _ = distribution.sample()
            return next_state

        return SampledDistribution(sampler=sample_next_state)

    def simulate_reward(
        self,
        start_state_distribution: Distribution[NonTerminal[S]],
    ) -> Iterable[TransitionStep[S]]:
        state: State[S] = start_state_distribution.sample()
        reward: float = 0.0

        while isinstance(state, NonTerminal):
            next_distribution = self.transition_reward(state=state)
            next_state, reward = next_distribution.sample()

            yield TransitionStep(state=state, next_state=next_state, reward=reward)
            state = next_state


StateReward = FiniteDistribution[tuple[State[S], float]]
RewardTransition = Mapping[NonTerminal[S], StateReward[S]]


class FiniteMarkovRewardProcess(FiniteMarkovProcess[S], MarkovRewardProcess[S]):
    transition_reward_map: RewardTransition[S]
    reward_function_vector: np.ndarray

    def __init__(
        self,
        transition_reward_map: Mapping[S, FiniteDistribution[tuple[S, float]]],
    ):
        transition_map: dict[S, FiniteDistribution[S]] = {}

        for state, distribution in transition_reward_map.items():
            probabilities: dict[S, float] = defaultdict(float)

            for (next_state, _), probability in distribution:
                probabilities[next_state] += probability

            transition_map[state] = Categorical(probabilities)

        super().__init__(transition_map=transition_map)

        nt: set[S] = set(transition_reward_map.keys())

        self.transition_reward_map = {
            NonTerminal(current_state): Categorical(
                {
                    (
                        NonTerminal(next_state)
                        if next_state in nt
                        else Terminal(next_state),
                        reward,
                    ): probability
                    for (next_state, reward), probability in distribution
                }
            )
            for current_state, distribution in transition_reward_map.items()
        }

        self.reward_function_vector = np.array(
            [
                sum(
                    probability * reward
                    for (_, reward), probability in self.transition_reward_map[state]
                )
                for state in self.non_terminal_states
            ]
        )

    def transition_reward(self, state: NonTerminal[S]) -> StateReward[S]:
        return self.transition_reward_map[state]
