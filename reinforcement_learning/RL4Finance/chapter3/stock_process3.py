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
from rl.math import get_unit_sigmoid_function


@dataclass(frozen=True)
class StateMP3:
    num_up_moves: int
    num_down_moves: int


@dataclass
class StockPriceMP3(MarkovProcess[StateMP3]):
    alpha: float = 1.0  # strength of reverse-pull (non-negative value)

    def calculate_up_prob(self, state: StateMP3) -> float:
        total = state.num_up_moves + state.num_down_moves

        if total:
            return get_unit_sigmoid_function(alpha=self.alpha)(
                state.num_down_moves / total
            )

        else:
            return 0.5

    def transition(self, state: NonTerminal[StateMP3]) -> Categorical[State[StateMP3]]:
        up_prob = self.calculate_up_prob(state=state.state)

        return Categorical(
            {
                NonTerminal(
                    StateMP3(
                        num_up_moves=state.state.num_up_moves + 1,
                        num_down_moves=state.state.num_down_moves,
                    )
                ): up_prob,
                NonTerminal(
                    StateMP3(
                        num_up_moves=state.state.num_up_moves,
                        num_down_moves=state.state.num_down_moves + 1,
                    )
                ): 1.0 - up_prob,
            }
        )


def process3_price_traces(
    start_price: int,
    alpha: float,
    time_steps: int,
    num_traces: int,
) -> np.ndarray:
    mp = StockPriceMP3(alpha=alpha)

    start_state_distribution = Constant(
        NonTerminal(StateMP3(num_up_moves=0, num_down_moves=0))
    )

    return np.vstack(
        [
            np.fromiter(
                (
                    start_price + s.state.num_up_moves - s.state.num_down_moves
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
    traces = process3_price_traces(
        start_price=20,
        alpha=0.2,
        time_steps=1000,
        num_traces=100,
    )

    traces = traces.T
    traces = np.mean(traces, axis=1)

    plt.subplots(figsize=(10, 6))
    plt.plot(range(1, traces.shape[0] + 1), traces)
    plt.axhline(20, linestyle="--", color="black", label="Starting price = $20")

    plt.title("Dynamics of Stock Price for Process 3")
    plt.xlabel("Time, $t$")
    plt.ylabel("Stock price, $X_t$ (in dollars)")
    plt.legend()

    plt.savefig(Path(__file__).parent / "images" / "process3.png")
    plt.close()
