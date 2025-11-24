from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import (
    DefaultDict,
    Dict,
    Generic,
    Iterable,
    Mapping,
    Sequence,
    Set,
    Tuple,
    TypeVar,
)

from rl.distributions import Categorical, Distribution, FiniteDistribution
from rl.markov_process import (
    FiniteMarkovRewardProcess,
    MarkovRewardProcess,
    NonTerminal,
    State,
    StateReward,
    Terminal,
)
from rl.policy import Policy

ACTION = TypeVar("Action")
STATE = TypeVar("State")


@dataclass(frozen=True)
class TransitionStep(Generic[STATE, ACTION]):
    state: NonTerminal[STATE]
    action: ACTION
    next_state: State[STATE]
    reward: float

    def add_return(
        self, discount_factor: float, total_return: float
    ) -> ReturnStep[STATE, ACTION]:
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
