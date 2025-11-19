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

import matplotlib.pyplot as plt
import numpy as np

from rl.distributions import Categorical, Constant
from rl.markov_process import MarkovProcess, NonTerminal, State
from rl.math import get_logistic_function


@dataclass(frozen=True)
class StateMP1:
    price: int


@dataclass
class StockPriceMP1(MarkovProcess[StateMP1]):
    level: int  # mean-reversion level
    alpha: float  # strength of mean-reversion (must be non-negative)

    def up_prob(self, state: StateMP1) -> float:
        return get_logistic_function(alpha=self.alpha)(self.level - state.price)

    def transition(self, state: NonTerminal[StateMP1]) -> Categorical[State[StateMP1]]:
        up_prob = self.up_prob(state=state.state)

        return Categorical(
            distribution={
                NonTerminal(StateMP1(price=state.state.price + 1)): up_prob,
                NonTerminal(StateMP1(price=state.state.price - 1)): 1.0 - up_prob,
            }
        )


def process1_price_traces(
    start_price: int,
    level: int,
    alpha: float,
    time_steps: int,
    num_traces: int,
) -> np.ndarray:
    mp = StockPriceMP1(level=level, alpha=alpha)

    start_state_distribution = Constant(NonTerminal(StateMP1(price=start_price)))

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
    traces = process1_price_traces(
        start_price=20,
        level=20,
        alpha=0.2,
        time_steps=1_000,
        num_traces=100,
    )

    traces = traces.T
    traces = np.mean(traces, axis=1)

    plt.subplots(figsize=(10, 6))
    plt.plot(range(1, traces.shape[0] + 1), traces)
    plt.axhline(20, linestyle="--", color="black", label="Mean-reversion Level")

    plt.title("Dynamics of Stock Price for Process 1")
    plt.xlabel("Time, $t$")
    plt.ylabel("Stock price, $X_t$ (in dollars)")

    plt.legend()
    plt.savefig(Path(__file__).parent / "images" / "process1.png")
    plt.close()
