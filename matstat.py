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


if __name__ == "__main__":


    x = StochVar([
        Event(0.2, 1),
        Event(0.3, 2),
        Event(0.3, 3)
    ])

    y = x.expected_value()
    print(y)

    ffg = distributions.GeometricDistribution(1/6)
    ffg.plot(20)

