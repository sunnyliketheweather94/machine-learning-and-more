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
from typing import Mapping

from scipy.stats import poisson

from rl.distributions import Categorical, FiniteDistribution
from rl.markov_process import FiniteMarkovProcess


@dataclass(frozen=True)
class Inventory:
    on_hand: int  # how much exists in the inventory by 6 pm
    on_order: int  # how much will arrive next day at 6 am

    @property
    def inventory_position(self) -> int:
        return self.on_hand + self.on_order


class BicycleInventoryFiniteMP(FiniteMarkovProcess[Inventory]):
    def __init__(self, capacity: int, poisson_lambda: float):
        self.capacity: int = capacity
        self.poisson_lambda: float = poisson_lambda
        self.poisson_dist = poisson(poisson_lambda)
        super().__init__(self.get_transition_map())

    def get_transition_map(self) -> Mapping[Inventory, FiniteDistribution[Inventory]]:
        dist: dict[Inventory, Categorical[Inventory]] = {}

        for alpha in range(self.capacity + 1):
            for beta in range(self.capacity + 1 - alpha):
                source_inventory = Inventory(on_hand=alpha, on_order=beta)
                total_inventory = source_inventory.inventory_position
                leftover_capacity = self.capacity - total_inventory

                probabilities: Mapping[Inventory, float] = {
                    # destination state
                    Inventory(
                        on_hand=total_inventory - customer_demand,
                        on_order=leftover_capacity,
                    ): (
                        # if there's more possible to order,
                        # then calculate prob of customer demand
                        self.poisson_dist.pmf(customer_demand)
                        if customer_demand < total_inventory
                        # if customers buy all available bicycles
                        # then calculate demand prob accordingly
                        else 1.0 - self.poisson_dist.cdf(total_inventory - 1)
                    )
                    for customer_demand in range(total_inventory + 1)
                }

                dist[source_inventory] = Categorical(probabilities)

        return dist


if __name__ == "__main__":
    mp = BicycleInventoryFiniteMP(capacity=6, poisson_lambda=5.0)

    # print out the transition probabilities
    print(mp)

    print()

    # print out the stationary distribution
    stationary_dist = mp.compute_stationary_distribution()

    for state, prob in stationary_dist:
        print(f"State {state}: probability = {prob:.3f}")
