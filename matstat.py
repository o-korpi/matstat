#!/usr/bin/env python


from dataclasses import dataclass
from typing import List
import distributions


@dataclass
class Event:
    probability: float
    value: int


@dataclass
class StochVar:
    """
        Discrete, endim stochvar
    """

    events: List[Event]


    # väntevärde
    def expected_value(self) -> float:
        out = 0
        for event in self.events:
            out += event.probability * event.value
        return out


def even_distribution(size: int) -> StochVar:
    events = []
    for value in range(0, size):
        events.append(Event(1 / size, value))

    return StochVar(events)


if __name__ == "__main__":
    ffg = distributions.ExponentialDistribution(0.5)
    ffg.plot(10, 1)

    ffg.to_poisson().plot(10)

