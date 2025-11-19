import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.abspath(__file__),
            os.pardir,
            os.pardir,
        )
    )
)

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from rl.distributions import Categorical, Constant
from rl.markov_process import MarkovProcess, NonTerminal, State


@dataclass(frozen=True)
class StateMP2:
    price: int
    delta: Literal[-1, 1, None]


@dataclass
class StockPriceMP2(MarkovProcess[StateMP2]):
    alpha: float  # strength of pull (must be in [0, 1])

    def up_prob(self, state: StateMP2) -> float:
        if state.delta is None:
            return 0.5

        return 0.5 * (1.0 - self.alpha * state.delta)

    def transition(self, state: NonTerminal[StateMP2]) -> Categorical[State[StateMP2]]:
        up_prob = self.up_prob(state=state.state)

        return Categorical(
            distribution={
                NonTerminal(StateMP2(price=state.state.price + 1, delta=1)): up_prob,
                NonTerminal(StateMP2(price=state.state.price - 1, delta=-1)): 1.0
                - up_prob,
            }
        )


def process2_price_traces(
    start_price: int,
    alpha: float,
    time_steps: int,
    num_traces: int,
) -> np.ndarray:
    mp = StockPriceMP2(alpha=alpha)

    start_state_distribution = Constant(
        NonTerminal(StateMP2(price=start_price, delta=None))
    )

    return np.vstack(
        [
            np.fromiter(
                (
                    s.state.price
                    for s in itertools.islice(
                        mp.simulate(start_state_distribution=start_state_distribution),
                        time_steps + 1,
                    )
                ),
                float,
            )
            for _ in range(num_traces)
        ]
    )


if __name__ == "__main__":
    traces = process2_price_traces(
        start_price=20,
        alpha=0.2,
        time_steps=1000,
        num_traces=100,
    )

    traces = traces.T
    traces = np.mean(traces, axis=1)

    plt.subplots(figsize=(10, 6))
    plt.plot(range(1, traces.shape[0] + 1), traces)

    plt.title("Dynamics of Stock Price for Process 2")
    plt.xlabel("Time, $t$")
    plt.ylabel("Stock price, $X_t$ (in dollars)")

    plt.savefig(Path(__file__).parent / "images" / "process2.png")
    plt.close()
