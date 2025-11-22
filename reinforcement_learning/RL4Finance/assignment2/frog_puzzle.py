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
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from rl.distributions import Categorical, Constant, FiniteDistribution
from rl.markov_process import FiniteMarkovProcess, NonTerminal


@dataclass(frozen=True)
class State:
    position: int


class FrogPuzzle(FiniteMarkovProcess[State]):
    def __init__(self, width: int) -> None:
        self.width = width
        logger.info(f"Setting up the Frog Puzzle with width {self.width}")

        super().__init__(transition_map=self.compute_transition_map())

    def compute_transition_map(self) -> dict[State, FiniteDistribution[State]]:
        distribution: dict[State, FiniteDistribution[State]] = defaultdict(float)

        for current_position in range(0, self.width):
            distribution[State(position=current_position)] = Categorical(
                {
                    State(position=next_position): 1.0
                    for next_position in range(current_position + 1, self.width + 1)
                }
            )

        return distribution


if __name__ == "__main__":
    W = 5

    puzzle = FrogPuzzle(width=W)

    # print out the transition probabilities
    print(puzzle)

    traces = [
        sum(
            1
            for _ in itertools.islice(
                puzzle.simulate(
                    start_state_distribution=Constant(
                        value=NonTerminal(State(position=0))
                    )
                ),
                # skipping 1st position bc that's always the beginning
                # position and we want the expected # of steps
                1,
                None,
            )
        )
        for _ in range(100_000)
    ]

    print(f"Expected number of steps taken: {np.mean(traces):.3f}")

    plt.hist(traces)
    plt.savefig(Path(__file__).parent / "images" / f"frog_puzzle_width{W}.png")
