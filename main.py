#!/usr/bin/env python
from distributions import Poisson

if __name__ == '__main__':
    import sys

    args = sys.argv[1:]

    distribution = args[0]

    # todo: use cli lib
    match distribution:
        case "poisson":
            try:
                Poisson(float(args[1])).plot(int(args[2]))
            except IndexError or ValueError:
                print("Use: poisson [mu] [x range]")
        case _:
            print(f"Unknown distribution '{distribution}'")
