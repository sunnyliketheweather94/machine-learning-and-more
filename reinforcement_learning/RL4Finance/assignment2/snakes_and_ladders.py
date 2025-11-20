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

from dataclasses import dataclass

import numpy as np

from rl.distributions import Categorical, FiniteDistribution
from rl.markov_process import FiniteMarkovProcess


@dataclass(frozen=True)
class State:
    position: int


class SnakesAndLadders(FiniteMarkovProcess[State]):
    num_rolls: int
    num_squares: int
    snakes: dict[int, int]
    ladders: dict[int, int]

    def __init__(
        self,
        snakes: dict[int, int],
        ladders: dict[int, int],
        num_rolls: int = 6,
        num_squares: int = 100,
    ):
        self.num_rolls = num_rolls
        self.num_squares = num_squares
        self.snakes = snakes
        self.ladders = ladders

        super().__init__(transition_map=self.compute_transitions())

    def compute_transitions(self) -> dict[State, FiniteDistribution[State]]:
        transitions: dict[State, FiniteDistribution[State]] = {}

        for current_position in range(1, self.num_squares + 1):
            if (
                current_position in self.snakes
                or current_position in self.ladders
                or current_position == self.num_squares
            ):
                continue

            tmp_dist: dict[State, float] = dict()
            current_state = State(position=current_position)

            for direction in range(1, self.num_rolls + 1):
                next_position = current_position + direction

                # the player needs to roll the exact
                # number in order to reach 10
                if next_position > self.num_squares:
                    continue

                # travel down the snake or the ladder
                next_position = self.snakes.get(next_position, next_position)
                next_position = self.ladders.get(next_position, next_position)

                # assign a value of 1
                # (the Categorical init method will normalize it)
                tmp_dist[State(position=next_position)] = (
                    tmp_dist.get(State(position=next_position), 0.0) + 1.0
                )

            transitions[current_state] = Categorical(tmp_dist)

        return transitions


def generate_snake_connections(n_snakes: int, n_squares: int) -> int:
    snakes = {}

    while len(snakes) < n_snakes:
        source = np.random.choice(range(2, n_squares))
        if (
            source.item() not in snakes.keys()
            and source.item() not in snakes.values()
            and source.item() != 2
        ):
            destination = np.random.choice(range(1, source.item()))

            if source.item() != destination.item():
                snakes[source.item()] = destination.item()

    print("===== SNAKES =====")
    for s, d in snakes.items():
        print(f"{s} ---> {d}")
    print()

    return snakes


if __name__ == "__main__":
    ROLLS = 6
    SQUARES = 100
    N_LADDERS = 3
    N_SNAKES = 15

    snakes = generate_snake_connections(n_snakes=N_SNAKES, n_squares=SQUARES)
    ladders = {}

    game = SnakesAndLadders(
        ladders=ladders,
        snakes=snakes,
        num_rolls=ROLLS,
        num_squares=SQUARES,
    )

    # print(game)

    # stationary_distribution = game.compute_stationary_distribution()

    # for state, prob in stationary_distribution:
    #     print(f"{state=}: {prob=:.3f}")
