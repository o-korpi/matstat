from __future__ import annotations

import math
from math import factorial, exp, sqrt
from statistics import NormalDist
from typing import Protocol
import matplotlib.pyplot as plt



def binomial_c(n, k):
    """Binomial coefficient"""
    return factorial(n) / (factorial(k) * factorial(n - k))


class DiscreteDistribution(Protocol):
    def pmf(self, k: int) -> float:
        ...

    def plot(self) -> None:
        ...


class ContinuousDistribution(Protocol):
    def pdf(self, k: float) -> float:
        ...

    def cdf(self, k: float) -> float:
        ...

    def plot(self, n: int = 25, sample_rate: int = 5) -> None:
        x_axis = range(0, n * sample_rate)
        data = [self.pdf(k / sample_rate) for k in x_axis]
        plt.plot([x / sample_rate for x in x_axis], data)
        plt.show()


class FFG(DiscreteDistribution):

    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def px(self, k: int) -> float:
        return pow(1 - self.p, k - 1) * self.p

    def plot(self, n: int = 25):
        x_axis = range(1, n)
        data = [self.px(x) for x in range(1, n)]
        plt.bar(x_axis, data)
        plt.show()


class GeometricDistribution(DiscreteDistribution):
    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def px(self, k: int) -> float:
        return pow(1 - self.p, k) * self.p

    def plot(self, n: int = 25):
        x_axis = range(0, n)
        data = [self.px(x) for x in range(0, n)]
        plt.bar(x_axis, data)
        plt.show()


class Binomial(DiscreteDistribution):
    """Binomial distribution

    0 <= p <= 1

    """
    def __init__(self, p: float, n: float) -> None:
        super().__init__()

        if (p < 0) or (p > 1):
            raise ValueError("p must be between 0 and 1")

        self.p = p
        self.n = n

    def pmf(self, k: int) -> float:
        if k > self.n:
            return 0
        else:
            return (binomial_c(self.n, k)) * pow(self.p, k) * pow(1-self.p, self.n-k)

    def plot(self, n: int = 25):
        x_axis = range(0, n)
        data = [self.pmf(x) for x in range(0, n)]
        plt.bar(x_axis, data)
        plt.show()

    def expected_value(self):
        return self.n * self.p

    def variance(self):
        return self.n * self.p * (1 - self.p)


class Hypergeometric(DiscreteDistribution):
    """Hypergeometric distribution

    N: Total population size
    A: Number of items of interest
    n: Sample size
    """

    def __init__(self, large_n, large_a, n) -> None:
        super().__init__()

        self.large_n = large_n
        self.large_a = large_a
        self.n = n

    def pmf(self, k: int) -> float:
        if k > self.n or (self.n - k) > (self.large_n - self.large_a) or k > self.large_a:
            return 0
        else:
            return binomial_c(self.large_a, k) * binomial_c((self.large_n - self.large_a), (self.n - k)) / binomial_c(self.large_n, self.n)

    def plot(self, n: int = 25):
        x_axis = range(0, n)
        data = [self.pmf(k) for k in x_axis]
        plt.bar(x_axis, data)
        plt.show()

    def expected_value(self):
        return self.n * self.large_a / self.large_n

    def variance(self):
        return (self.n * self.large_a / self.large_n) * ((self.large_n - self.large_a) / self.large_n) * ((self.large_n - self.n) / (self.large_n - 1))


class Poisson(DiscreteDistribution):
    """Poisson distribution

    Requirements:
    - The rate of which events occur must be constant
    - Events must be independent

    """
    def __init__(self, mu: float) -> None:
        super().__init__()

        if mu <= 0:
            raise ValueError("mu must be positive")

        self.mu = mu

    def pmf(self, k: int) -> float:
        return (pow(self.mu, k) / factorial(k)) * exp(-1 * self.mu)

    def plot(self, n: int = 25):
        x_axis = range(0, n)
        data = [self.pmf(x) for x in range(0, n)]
        plt.bar(x_axis, data)
        plt.show()

    def expected_value(self) -> float:
        return self.mu

    def variance(self) -> float:
        return self.mu

    def to_exponential(self) -> ExponentialDistribution:
        return ExponentialDistribution(1 / self.mu)


class ExponentialDistribution(ContinuousDistribution):
    """Exponential distribution

    Time between events in a Poisson process (inverse of Poisson)

    Requirements:
    - The rate of which events occur must be constant
    - Events must be independent
    - Lambda (l) > 0


    """
    def __init__(self, l: float) -> None:
        super().__init__()

        if l <= 0:
            raise ValueError("Lambda (l) must be positive")

        self.l = l

    def pdf(self, k: float) -> float:
        if k < 0:
            return 0
        else:
            return self.l * exp((-1 * self.l) * k)

    def cdf(self, k: float) -> float:
        if k < 0:
            return 0
        else:
            return 1 - exp((-1 * self.l) * k)

    def to_poisson(self) -> Poisson:
        return Poisson(1 / self.l)


class NormalDistribution(ContinuousDistribution):
    """Normal distribution

    """

    def __init__(self, mu: float, sigma: float) -> None:
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def pdf(self, k: float) -> float:
        return (1 / sqrt(2 * math.pi * self.sigma ** 2)) * exp(-1 * ((k - self.mu) ** 2 / (2 * self.sigma ** 2)))

    def cdf(self, k: float) -> float:
        return NormalDist(mu=self.mu, sigma=self.sigma).cdf(k)

    def z(self, x: float) -> float:
        return (x - self.mu) / self.sigma

    @staticmethod
    def standard_normal_distribution() -> NormalDistribution:
        """Create a standard normal distribution"""
        return NormalDistribution(0, 1)

    def plot(self, n: int = 3, sample_rate: int = 15) -> None:
        x_axis = range(-n * sample_rate, n * sample_rate)
        data = [self.pdf(k / sample_rate) for k in x_axis]
        plt.plot([x / sample_rate for x in x_axis], data)
        plt.show()

    def expected_value(self) -> float:
        return self.mu

    def variance(self) -> float:
        return self.sigma ** 2
