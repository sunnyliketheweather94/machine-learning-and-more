from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Generic, Iterable, TypeVar

from rl.distributions import Choose, Constant, Distribution, FiniteDistribution
from rl.markov_process import NonTerminal

ACTION = TypeVar("Action")  # denotes an action
STATE = TypeVar("State")  # denotes a state


class Policy(ABC, Generic[STATE, ACTION]):
    @abstractmethod
    def act(self, state: NonTerminal[STATE]) -> Distribution[ACTION]:
        pass


@dataclass(frozen=True)
class DeterministicPolicy(Policy[STATE, ACTION]):
    action_for: Callable[[STATE], ACTION]

    def act(self, state: NonTerminal[STATE]) -> Constant[ACTION]:
        return Constant(self.action_for(state.state))


@dataclass(frozen=True)
class RandomPolicy(Policy[STATE, ACTION]):
    policy_choices: Distribution[Policy[STATE, ACTION]]

    def act(self, state: NonTerminal[STATE]) -> Distribution[ACTION]:
        policy: Policy[STATE, ACTION] = self.policy_choices.sample()
        return policy.act(state=state)


@dataclass(frozen=True)
class UniformPolicy(Policy[STATE, ACTION]):
    valid_actions: Callable[[STATE], Iterable[ACTION]]

    def act(self, state: NonTerminal[STATE]) -> Choose[ACTION]:
        return Choose(options=self.valid_actions(state.state))


class Always(DeterministicPolicy[STATE, ACTION]):
    action: ACTION

    def __init__(self, action: ACTION):
        self.action = action
        super().__init__(action_for=lambda _: action)


@dataclass(frozen=True)
class FinitePolicy(Policy[STATE, ACTION]):
    policy_map: dict[STATE, FiniteDistribution[ACTION]]

    def __repr__(self) -> str:
        display = ""
        for state, action_dist in self.policy_map.items():
            display += f"For State {state}:\n"
            for action, prob in action_dist:
                display += f"  Do Action {action} "
                display += f"with Probability {prob:.3f}\n"

        return display

    def act(self, state: NonTerminal[STATE]) -> FiniteDistribution[ACTION]:
        return self.policy_map[state.state]


class FiniteDeterministicPolicy(FinitePolicy[STATE, ACTION]):
    action_for: dict[STATE, ACTION]

    def __init__(self, action_for: dict[STATE, ACTION]):
        self.action_for = action_for
        super().__init__(
            policy_map={
                state: Constant(action) for state, action in self.action_for.items()
            }
        )

    def __repr__(self) -> str:
        display = ""
        for state, action in self.action_for.items():
            display += f"For State {state}: Do Action {action}\n"

        return display
