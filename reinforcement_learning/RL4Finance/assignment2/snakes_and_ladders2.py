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
import heapq
import random
from dataclasses import dataclass
from operator import itemgetter
from pathlib import Path
from typing import Dict, Set

import matplotlib.pyplot as plt
from loguru import logger

from rl.distributions import Categorical, Constant, FiniteDistribution
from rl.markov_process import FiniteMarkovRewardProcess, NonTerminal


@dataclass(frozen=True)
class State:
    """
    Represents the state of the player on the Snakes and Ladders board.

    The state is defined solely by the player's position. It is frozen
    to ensure immutability, which is necessary when using it as a key
    in a dictionary (e.g., in the transition map).

    Attributes:
        position: The square number the player is currently on (1 to num_squares).
    """

    position: int


class SnakesAndLadders(FiniteMarkovRewardProcess[State]):
    """
    Models the game of Snakes and Ladders as a Finite Markov Process (FMP).

    The game state is the player's position on the board. The process
    is defined by the transition probabilities after a single die roll.

    Attributes:
        num_rolls (int): The maximum value of the die (e.g., 6 for a standard die).
        num_squares (int): The total number of squares on the board (e.g., 100).
        snakes (dict[int, int]): Mapping of snake heads (source) to tails (destination).
        ladders (dict[int, int]): Mapping of ladder bases (source) to tops (destination).
    """

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

        super().__init__(transition_reward_map=self.compute_transitions())

    def compute_transitions(self) -> dict[State, FiniteDistribution[State]]:
        """
        Computes the transition probabilities for all non-terminal states.

        The transition map links a current State to a distribution over
        possible next States after one die roll.

        A state is considered terminal (and skipped) if:
        1. It is the end square (num_squares).
        2. It is the head of a snake or the base of a ladder, as the player
           immediately moves from there to the destination.

        Returns:
            A dictionary where keys are non-terminal State objects and values
            are Categorical distributions over the possible next State objects.
        """
        transitions: dict[State, FiniteDistribution[State]] = {}

        for current_position in range(1, self.num_squares + 1):
            if (
                current_position in self.snakes
                or current_position in self.ladders
                or current_position == self.num_squares
            ):
                continue

            next_state_reward_map: dict[tuple[State, float], float] = dict()
            current_state = State(position=current_position)

            for direction in range(1, self.num_rolls + 1):
                next_position = current_position + direction

                # the player needs to roll the exact
                # number in order to reach 10
                if next_position > self.num_squares:
                    next_position = current_position

                # travel down the snake or the ladder
                next_position = self.snakes.get(next_position, next_position)
                next_position = self.ladders.get(next_position, next_position)

                # assign a value of 1
                # (the Categorical init method will normalize it)
                next_state_reward_map[(State(position=next_position), 1.0)] = (
                    next_state_reward_map.get((State(position=next_position), 1.0), 0.0)
                    + 1.0
                )

            transitions[current_state] = Categorical(next_state_reward_map)

        return transitions


def generate_connections(
    num_squares: int,
    num_connections: int,
    is_snakes: bool,
) -> Dict[int, int]:
    """
    Generates a dictionary of unique connections (snakes or ladders) for a
    Snakes and Ladders board.

    A valid connection is a pair (source, destination) where:
    1. The source and destination are not the start (1) or end (num_squares)
       of the board.
    2. The source is not already a destination for another connection, and
       the destination is not already a source for another connection.
    3. The source and destination are not the same.
    4. For **snakes**, source > destination (downward movement).
    5. For **ladders**, source < destination (upward movement).

    Args:
        num_squares: The total number of squares on the board (e.g., 100).
        num_connections: The desired number of connections to generate.
        is_snakes: If True, generates snakes (source > dest).
                   If False, generates ladders (source < dest).

    Returns:
        A dictionary mapping the connection source square (int)
        to the destination square (int).
    """

    # Square 1 (start) and num_squares (end) cannot be sources or destinations.
    valid_squares = set(range(2, num_squares))

    connections: Dict[int, int] = {}
    sources: Set[int] = set()
    destinations: Set[int] = set()

    # Determine the connection type for clear logic
    connection_type = "snake" if is_snakes else "ladder"

    # We must have at least 2 squares available to make a connection
    if len(valid_squares) < 2:
        return {}

    # The maximum possible number of non-overlapping connections is half the
    # number of available squares, as each uses two squares.
    max_possible = len(valid_squares) // 2
    num_connections = min(num_connections, max_possible)

    while len(connections) < num_connections:
        # Choose a random source square from those not already used
        available_sources = list(valid_squares - sources - destinations)

        if not available_sources:
            # Cannot find any more valid sources, stop the loop.
            logger.warning(
                f"Warning: Could only generate {len(connections)} "
                f"{connection_type}s out of requested {num_connections} due to square conflicts."
            )
            break

        source = random.choice(available_sources)

        # Determine the valid range for the destination based on the connection type
        if is_snakes:
            # Destination must be in the range [2, source - 1]
            # Must be > 1 (not the start) and less than source.
            dest_range = range(2, source)
        else:
            # Destination must be in the range [source + 1, num_squares - 1]
            # Must be > source and less than the end square.
            dest_range = range(source + 1, num_squares)

        # Filter the destination range to exclude squares already used as a source or destination
        valid_destinations = list(set(dest_range) - sources - destinations)

        if valid_destinations:
            destination = random.choice(valid_destinations)

            # Add the new, valid connection
            connections[source] = destination
            sources.add(source)
            destinations.add(destination)

        # If no valid destination is found for the chosen source, the loop continues
        # to pick a new source.

    return connections


def generate_snake_connections(n_snakes: int, n_squares: int) -> Dict[int, int]:
    """
    Generates a dictionary of unique snake connections for the board.

    Args:
        n_snakes: The desired number of snake connections.
        n_squares: The total number of squares on the board.

    Returns:
        A dictionary mapping the snake's head (source) to its tail (destination).
    """
    return generate_connections(
        num_squares=n_squares,
        num_connections=n_snakes,
        is_snakes=True,
    )


def generate_ladder_connections(n_ladders: int, n_squares: int) -> Dict[int, int]:
    """
    Generates a dictionary of unique ladder connections for the board.

    Args:
        n_ladders: The desired number of ladder connections.
        n_squares: The total number of squares on the board.

    Returns:
        A dictionary mapping the ladder's base (source) to its top (destination).
    """
    return generate_connections(
        num_squares=n_squares,
        num_connections=n_ladders,
        is_snakes=False,
    )


if __name__ == "__main__":
    ROLLS = 6
    SQUARES = 100
    N_LADDERS = 15
    N_SNAKES = 20

    # generate random snake/ladder connections for the game
    snakes = generate_snake_connections(n_snakes=N_SNAKES, n_squares=SQUARES)
    ladders = generate_ladder_connections(n_ladders=N_LADDERS, n_squares=SQUARES)

    game = SnakesAndLadders(
        ladders=ladders,
        snakes=snakes,
        num_rolls=ROLLS,
        num_squares=SQUARES,
    )

    # # print the transition probabilities (for debugging purposes)
    # # set SQUARES to a small number
    # print(game)

    # to find the distribution of "time" (as measured by the number of
    # rolls of the dice) to finish the game, we generate 1000 sampling
    # traces and then count the length of each trace (as it terminates)
    # on a TerminalState. Use this list of times to generate the histogram.
    traces = [
        sum(
            s.reward
            for s in game.simulate_reward(
                start_state_distribution=Constant(NonTerminal(State(position=1)))
            )
        )
        for _ in range(1_000)
    ]

    os.makedirs(Path(__file__).parent / "images", exist_ok=True)

    plt.hist(traces)

    plt.title("Histogram of rolls for finishing Snakes and Ladders")
    plt.xlabel("Number of dice rolls")
    plt.ylabel("Frequency")
    plt.tight_layout()

    plt.savefig(
        Path(__file__).parent / "images" / "time_to_finish_snakes_and_ladders2.png"
    )

    plt.close()

    # in order to find the stationary distribution, use
    # the method developed in FiniteMarkovProcess class
    stationary_distribution = game.compute_stationary_distribution()
    top_k_states = heapq.nlargest(10, stationary_distribution, key=itemgetter(1))

    for state, prob in top_k_states:
        print(f"{state=}: {prob=:.3f}")
