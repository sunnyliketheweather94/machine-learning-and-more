import random
from typing import Dict, Set

from loguru import logger


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
