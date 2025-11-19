from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Generic, Iterable, TypeVar

from rl.distributions import Distribution

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
