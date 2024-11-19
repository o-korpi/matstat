from math import factorial, exp
from typing import Protocol
import matplotlib.pyplot as plt


def binomial(k, n):
    return factorial(n) / (factorial(k) * factorial(n - k))


class DiscreteDistribution(Protocol):
    def plot(self) -> None:
        ...


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
    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def px(self, k: int, n: int) -> float:
        return (factorial(n) / (factorial(k) * factorial(n - k))) * pow(self.p, k) * pow(1-self.p, n-k)

    def plot(self, n: int = 25):
        x_axis = range(0, n)
        data = [self.px(x, n) for x in range(0, n)]
        plt.bar(x_axis, data)
        plt.show()


# vet inte hur jag gör hypergeometrisk lol


class Poisson(DiscreteDistribution):
    def __init__(self, mu: float) -> None:
        super().__init__()

        if mu <= 0:
            raise ValueError("mu must be positive")

        self.mu = mu

    def px(self, k: int, mu: float) -> float:
        return (pow(mu, k) / factorial(k)) * exp(-1 * mu)

    def plot(self, n: int = 25):
        x_axis = range(0, n)
        data = [self.px(x, self.mu) for x in range(0, n)]
        plt.bar(x_axis, data)
        plt.show()
