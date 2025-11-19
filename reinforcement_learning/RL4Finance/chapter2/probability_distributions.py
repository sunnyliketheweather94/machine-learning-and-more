from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Generic, Iterable, Iterator, Mapping, Sequence, TypeVar

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
        return self.table().get(outcome, 0.0)

    def map(self, f: Callable[[A], B]) -> FiniteDistribution[B]:
        result: dict[B, float] = defaultdict(float)

        for outcome, probability in self:
            result[f(outcome)] += probability

        return Categorical(result)

    def sample(self) -> A:
        outcomes = list(self.table().keys())
        weights = list(self.table().values())

        return np.random.choice(outcomes, weights=weights)[0]

    def expectation(self, f: Callable[[A], float]) -> float:
        return sum(prob * f(outcome) for outcome, prob in self)

    def __iter__(self) -> Iterator[tuple[A, float]]:
        return iter(self.table().items())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FiniteDistribution):
            return self.table() == other.table()

        return False

    def __repr__(self) -> str:
        return repr(self.table())


@dataclass(frozen=True)
class Constant(FiniteDistribution[A]):
    value: A

    def sample(self) -> A:
        return self.value

    def table(self) -> Mapping[A, float]:
        return {self.value: 1}

    def probability(self, outcome: A) -> float:
        return 1.0 if outcome == self.value else 0.0


@dataclass(frozen=True)
class Bernoulli(FiniteDistribution[bool]):
    p: float

    def sample(self) -> bool:
        return np.random.uniform(0, 1) <= self.p

    def table(self) -> Mapping[bool, float]:
        return {True: self.p, False: 1 - self.p}

    def probability(self, outcome: bool) -> float:
        return self.p if outcome else 1 - self.p


@dataclass(frozen=True)
class Range(FiniteDistribution[int]):
    low: int
    high: int

    def __init__(self, a: int, b: int | None) -> None:
        if b is None:
            a, b = 0, a

        assert b > a

        self.low = a
        self.high = b

    def sample(self) -> int:
        return np.random.randint(low=self.low, high=self.high - 1)

    def table(self) -> Mapping[int, float]:
        length = self.high - self.low
        return {x: 1.0 / length for x in range(self.low, self.high)}


class Choose(FiniteDistribution[A]):
    options: Sequence[A]
    _table: Mapping[A, float] | None = None

    def __init__(self, options: Iterable[A]):
        self.options = list(options)

    def sample(self) -> A:
        return np.random.choice(self.options)

    def table(self) -> Mapping[A, float]:
        if self._table is None:
            counter = Counter(self.options)
            length = len(self.options)
            self._table = {x: counter[x] / length for x in counter}

        return self._table

    def probability(self, outcome: A) -> float:
        return self.table().get(outcome, 0.0)


class Categorical(FiniteDistribution[A]):
    probabilities: Mapping[A, float]

    def __init__(self, distribution: Mapping[A, float]):
        total = sum(distribution.values())

        self.probabilities = {
            outcome: probability / total
            for outcome, probability in distribution.items()
        }

    def table(self) -> Mapping[A, float]:
        return self.probabilities

    def probability(self, outcome: A) -> float:
        return self.probabilities.get(outcome, 0.0)
