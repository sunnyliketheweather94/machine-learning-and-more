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
import pandas as pd
import seaborn as sns
from scipy.stats import poisson

from rl.distributions import Categorical, Constant, FiniteDistribution
from rl.markov_process import FiniteMarkovRewardProcess, NonTerminal


@dataclass(frozen=True)
class Inventory:
    on_hand: int
    on_order: int

    @property
    def inventory_position(self) -> int:
        return self.on_hand + self.on_order


class BicycleInventoryMRP(FiniteMarkovRewardProcess[Inventory]):
    def __init__(
        self,
        capacity: int,
        poisson_lambda: float,
        holding_cost: float,
        stockout_cost: float,
    ) -> None:
        self.capacity = capacity
        self.poisson_lambda = poisson_lambda
        self.poisson_dist = poisson(self.poisson_lambda)
        self.holding_cost = -holding_cost
        self.stockout_cost = -stockout_cost

        super().__init__(transition_reward_map=self.generate_transition_reward_map())

    def generate_transition_reward_map(
        self,
    ) -> dict[Inventory, FiniteDistribution[tuple[Inventory, float]]]:
        distribution: dict[Inventory, FiniteDistribution[tuple[Inventory, float]]] = (
            dict()
        )

        for alpha in range(self.capacity + 1):
            for beta in range(self.capacity + 1 - alpha):
                current_state = Inventory(on_hand=alpha, on_order=beta)
                total_inventory = current_state.inventory_position
                empty_units = self.capacity - total_inventory
                base_reward = self.holding_cost * current_state.on_hand

                next_state_reward_map: dict[tuple[Inventory, float], float] = {
                    (
                        Inventory(
                            on_hand=total_inventory - demand,
                            on_order=empty_units,
                        ),
                        base_reward,
                    ): self.poisson_dist.pmf(demand).item()
                    for demand in range(total_inventory)
                }

                prob = 1.0 - self.poisson_dist.cdf(total_inventory - 1)

                reward = (
                    base_reward
                    + self.stockout_cost
                    * (
                        self.poisson_lambda
                        - total_inventory
                        * (1 - self.poisson_dist.pmf(total_inventory) / prob)
                    ).item()
                )

                next_state_reward_map[
                    (
                        Inventory(on_hand=0, on_order=empty_units),
                        reward,
                    )
                ] = prob.item()

                distribution[current_state] = Categorical(next_state_reward_map)

        return distribution


if __name__ == "__main__":
    mrp = BicycleInventoryMRP(
        capacity=10,
        poisson_lambda=1.5,
        holding_cost=70.0,
        stockout_cost=200.0,
    )

    # # print out the transition map
    # print(mrp)
    # print()
    # # print out the stationary distribution
    # print(mrp.compute_stationary_distribution())
    # print()

    # # print out the transition reward map nicely
    # for current_state, next_step_map in mrp.transition_reward_map.items():
    #     print(f"Yesterday's {current_state.state}:")
    #     for (next_state, reward), prob in next_step_map.table().items():
    #         print(
    #             f"\tToday's EOD {next_state.state} and Reward {reward:.2f} "
    #             f"--> Probability = {prob:.3f}"
    #         )
    #     print()

    ##### Generate traces and plot heatmap of the average costs per starting state #####
    state = NonTerminal(Inventory(on_hand=10, on_order=0))

    traces: list[list[tuple[Inventory, Inventory, float]]] = [
        [
            (s.state.state, s.next_state.state, s.reward)
            for s in itertools.islice(
                mrp.simulate_reward(start_state_distribution=Constant(value=state)),
                10_000,
            )
        ]
        for _ in range(100)
    ]

    stats = pd.DataFrame(
        {
            "On Hand": [s.on_hand for trace in traces for s, _, _ in trace],
            "On Order": [s.on_order for trace in traces for s, _, _ in trace],
            "Costs": [abs(r) for trace in traces for _, _, r in trace],
        }
    )

    stats = pd.pivot_table(
        data=stats,
        index="On Hand",
        columns="On Order",
        values="Costs",
        aggfunc="mean",
        dropna=False,
    ).sort_index(ascending=False)

    plt.subplots(figsize=(7, 6))
    sns.heatmap(data=stats, cmap="viridis_r")
    plt.title('Total Costs for each "On Hand"/"On Order" state')

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "images" / "bicycle_mrp.png")
    plt.close()
