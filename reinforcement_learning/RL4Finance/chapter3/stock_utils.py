import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class Price:
    price: int


class StockPriceDynamics(ABC):
    @abstractmethod
    def calculate_probability_of_up_move(self, current_price: Price) -> float:
        pass

    def calculate_next_price(self, current_price: Price) -> Price:
        probability = self.calculate_probability_of_up_move(current_price=current_price)

        direction = np.random.choice(
            a=[1, -1],
            p=[probability, 1.0 - probability],
            size=1,
        )[0]

        return Price(price=current_price.price + direction)

    def simulate_trace(self, start_price: int) -> Iterator[Price]:
        state = Price(price=start_price)

        while True:
            yield state
            state = self.calculate_next_price(current_price=state)

    def generate_traces(
        self,
        start_price: int,
        time_steps: int = 100,
        num_traces: int = 4,
    ) -> np.ndarray:
        return np.vstack(
            [
                np.fromiter(
                    (
                        s.price
                        for s in itertools.islice(
                            self.simulate_trace(start_price=start_price),
                            time_steps + 1,
                        )
                    ),
                    float,
                )
                for _ in range(num_traces)
            ]
        )


@dataclass
class Process1(StockPriceDynamics):
    level_param: int  # level to which price mean-reverts
    alpha: float  # strength of mean-reversion (must be non-negative)

    def calculate_probability_of_up_move(self, current_price: Price) -> float:
        return 1.0 / (
            1.0 + np.exp(-self.alpha * (self.level_param - current_price.price))
        )
