from abc import ABC
from dataclasses import dataclass
from typing import Callable, Generic

from rl.constants import STATE, X


class State(ABC, Generic[STATE]):
    state: STATE

    def on_on_terminal(self, f: Callable, default: X) -> X:
        """f takes a "NonTerminal[STATE]" arg and returns X"""
        if isinstance(self, NonTerminal):
            return f(self)

        return default


@dataclass(frozen=True)
class Terminal(State[STATE]):
    state: STATE


@dataclass(frozen=True)
class NonTerminal(State[STATE]):
    state: STATE
