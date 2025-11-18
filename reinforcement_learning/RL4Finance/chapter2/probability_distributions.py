from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable, Generic, Mapping, Sequence, TypeVar

import numpy as np

A = TypeVar("A")
B = TypeVar("B")


class Distribution(ABC, Generic[A]):
    @abstractmethod
    def sample(self) -> A:
        pass

    def sample_n(self, n: int) -> Sequence[A]:
        return [self.sample() for _ in range(n)]

    @abstractmethod
    def expectation(self, f: Callable[[A], float]) -> float:
        pass

    def map(self, f: Callable[[A], B]) -> Distribution[B]:
        return SampledDistribution(sampler=lambda: f(self.sample()))

    def apply(self, f: Callable[[A], Distribution[B]]) -> Distribution[B]:
        def _sample():
            a = self.sample()
            b_dist = f(a)  # f takes a value but returns a distribution
            return b_dist.sample()

        return SampledDistribution(_sample)


class SampledDistribution(Distribution[A]):
    sampler: Callable[[], A]
    expectation_samples: int

    def __init__(self, sampler: Callable[[], A], expectation_samples: int = 10_000):
        self.sampler = sampler
        self.expectation_samples = expectation_samples

    def sample(self) -> A:
        return self.sampler()

    def expectation(self, f: Callable[[A], float]) -> float:
        return (
            sum(f(self.sample()) for _ in range(self.expectation_samples))
            / self.expectation_samples
        )


class Dice(SampledDistribution[int]):
    sides: int

    def __init__(self, sides: int, expectation_samples: int = 10_000):
        self.sides = sides

        super().__init__(
            sampler=lambda: np.random.randint(low=1, high=self.sides + 1),
            expectation_samples=expectation_samples,
        )


class Uniform(SampledDistribution[float]):
    def __init__(self, expectation_samples: int = 10_000):
        super().__init__(
            sampler=lambda: np.random.uniform(low=0, high=1),
            expectation_samples=expectation_samples,
        )


class Poisson(SampledDistribution[int]):
    lam: float

    def __init__(self, lam: float, expectation_samples: int = 10_000):
        self.lam = lam

        super().__init__(
            sampler=lambda: np.random.poisson(lam=self.lam),
            expectation_samples=expectation_samples,
        )


class Gaussian(SampledDistribution[float]):
    mu: float
    sigma: float

    def __init__(self, mu: float, sigma: float, expectation_samples: int = 10_000):
        self.mu = mu
        self.sigma = sigma

        super().__init__(
            sampler=lambda: np.random.normal(loc=self.mu, scale=self.sigma),
            expectation_samples=expectation_samples,
        )


class Gamma(SampledDistribution[float]):
    alpha: float
    beta: float

    def __init__(self, alpha: float, beta: float, expectation_samples: int = 10_000):
        self.alpha = alpha
        self.beta = beta

        super().__init__(
            sampler=lambda: np.random.gamma(shape=self.alpha, scale=1 / self.beta),
            expectation_samples=expectation_samples,
        )


class Beta(SampledDistribution[float]):
    alpha: float
    beta: float

    def __init__(self, alpha: float, beta: float, expectation_samples: int = 10_000):
        self.alpha = alpha
        self.beta = beta

        super().__init__(
            sampler=lambda: np.random.beta(a=self.alpha, b=self.beta),
            expectation_samples=expectation_samples,
        )


class FiniteDistribution(Distribution[A], ABC):
    @abstractmethod
    def table(self) -> Mapping[A, float]:
        pass

    def probability(self, outcome: A) -> float:
        return self.table().get(outcome, 0)

    def map(self, f: Callable[[A], B]) -> FiniteDistribution[B]:
        result: dict[B, float] = defaultdict(float)

        for outcome, probability in self.table:
            result[f(outcome)] += probability

        return Categorical(result)

    def sample(self) -> A:
        outcomes = list(self.table().keys())
        weights = list(self.table().values())

        return np.random.choice(outcomes, weights=weights)[0]

    def expectation(self, f: Callable[[A], float]) -> float:
        return sum(prob * f(outcome) for outcome, prob in self.table())
