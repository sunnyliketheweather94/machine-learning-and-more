from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, Generic, Iterable, Mapping, Sequence, Set, Tuple

from rl.constants import ACTION, STATE
from rl.distributions import Categorical, Distribution, FiniteDistribution
from rl.markov_process import (
    FiniteMarkovRewardProcess,
    MarkovRewardProcess,
    NonTerminal,
    StateReward,
)
from rl.policy import FinitePolicy, Policy
from rl.states import NonTerminal, State, Terminal


@dataclass(frozen=True)
class TransitionStep(Generic[STATE, ACTION]):
    state: NonTerminal[STATE]
    action: ACTION
    next_state: State[STATE]
    reward: float

    def add_return(self, discount_factor: float, total_return: float):
        return ReturnStep(
            state=self.state,
            next_state=self.next_state,
            action=self.action,
            reward=self.reward,
            total_return=self.reward + discount_factor * total_return,
        )


@dataclass(frozen=True)
class ReturnStep(TransitionStep[STATE, ACTION]):
    total_return: float


class MarkovDecisionProcess(ABC, Generic[STATE, ACTION]):
    @abstractmethod
    def actions(self, state: NonTerminal[STATE]) -> Iterable[ACTION]:
        pass

    @abstractmethod
    def step(
        self,
        state: NonTerminal[STATE],
        action: ACTION,
    ) -> Distribution[tuple[State[STATE], float]]:
        """given state and action, return a distribution of (next state, reward)"""
        pass

    def apply_policy(self, policy: Policy[STATE, ACTION]) -> MarkovRewardProcess[STATE]:
        mdp = self

        class RewardProcess(MarkovRewardProcess[STATE]):
            def transition_reward(
                self,
                state: NonTerminal[STATE],
            ) -> Distribution[tuple[State[STATE], float]]:
                actions: Distribution[ACTION] = policy.act(state=state)
                return actions.apply(lambda a: mdp.step(state=state, action=a))

        return RewardProcess()

    def simulate_actions(
        self,
        start_states: Distribution[NonTerminal[STATE]],
        policy: Policy[STATE, ACTION],
    ) -> Iterable[TransitionStep[STATE, ACTION]]:
        state: State[STATE] = start_states.sample()

        while isinstance(state, NonTerminal):
            action_distribution: Distribution[ACTION] = policy.act(state=state)
            action: ACTION = action_distribution.sample()
            next_state_distribution = self.step(state=state, action=action)
            next_state, reward = next_state_distribution.sample()

            yield TransitionStep(
                state=state,
                action=action,
                next_state=next_state,
                reward=reward,
            )

            state = next_state


# {state -> {action -> {(next_state, reward) : probability}}}
ActionMapping = dict[ACTION, StateReward[STATE]]
StateActionMapping = dict[NonTerminal[STATE], ActionMapping[ACTION, STATE]]


class FiniteMarkovDecisionProcess(MarkovDecisionProcess[STATE, ACTION]):
    mapping: StateActionMapping[STATE, ACTION]
    non_terminal_states: list[NonTerminal[STATE]]

    def __init__(
        self,
        mapping: dict[STATE, dict[ACTION, FiniteDistribution[tuple[STATE, float]]]],
    ):
        non_terminals: set[STATE] = set(mapping.keys())
        self.mapping = {
            NonTerminal(state): {
                action: Categorical(
                    {
                        (
                            NonTerminal(next_state)
                            if next_state in non_terminals
                            else Terminal(next_state),
                            reward,
                        ): prob
                        for (next_state, reward), prob in next_state_reward_dist
                    }
                )
                for action, next_state_reward_dist in action_next_state_reward_mapping.items()
            }
            for state, action_next_state_reward_mapping in mapping.items()
        }

        self.non_terminal_states = list(self.mapping.keys())

    def __repr__(self) -> str:
        display = ""

        for state, action_next_state_reward_mapping in self.mapping.items():
            display += f"From State {state.state}:\n"
            for (
                action,
                next_state_reward_dist,
            ) in action_next_state_reward_mapping.items():
                display += f"  With Action {action}:\n"
                for (next_state, reward), prob in next_state_reward_dist:
                    optional = "Terminal " if isinstance(next_state, Terminal) else ""
                    display += f"    To [{optional}State {next_state.state}"
                    display += f" and Reward {reward:.3f}] "
                    display += f"with Probability {prob:.3f}"

        return display

    def step(self, state: NonTerminal[STATE], action: ACTION) -> StateReward[STATE]:
        action_map: ActionMapping[ACTION, STATE] = self.mapping[state]
        return action_map[action]

    def actions(self, state: NonTerminal[STATE]) -> Iterable[ACTION]:
        return self.mapping[state].keys()

    def apply_finite_policy(
        self, policy: FinitePolicy[STATE, ACTION]
    ) -> FiniteMarkovRewardProcess[STATE]:
        transition_mapping: dict[STATE, FiniteDistribution[tuple[STATE, float]]] = (
            dict()
        )

        for state in self.mapping:
            action_map: ActionMapping[STATE, ACTION] = self.mapping[state]
            outcomes: dict[tuple[STATE, float], float] = defaultdict(float)
            actions: FiniteDistribution[ACTION] = policy.act(state)
            for action, prob_action in actions:
                for (next_state, reward), prob in action_map[action]:
                    outcomes[(next_state.state, reward)] += prob_action * prob

            transition_mapping[state.state] = Categorical(outcomes)

        return FiniteMarkovRewardProcess(transition_mapping)
