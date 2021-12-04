import itertools
import operator
from pathlib import Path


def load_data(path: Path):
    with path.open("r") as in_file:
        data = [(row.split()[0], int(row.split()[1])) for row in in_file]
    return data


def main():
    data = load_data(Path(__file__).parent / "input.txt")
    print("--- Task 1 ---")
    task1(data)
    print("--- Task 2 ---")
    task2(data)


def task1(data):
    groups = [
        (k, list(g))
        for k, g in itertools.groupby(sorted(data), key=operator.itemgetter(0))
    ]
    sums = {k: sum(gs[1] for gs in g) for k, g in groups}
    print(f"Final horizontal position: {sums['forward']}")
    print(f"Final depth: {sums['down'] - sums['up']}")
    print(f'Product: {sums["forward"] * (sums["down"] - sums["up"])}')


def task2(data):
    aim = 0
    depth = 0
    horizontal = 0
    for row in data:
        match row:
            case ('down', y):
                aim += y
            case ('up', y):
                aim -= y
            case ('forward', x):
                horizontal += x
                depth += aim * x
    print(f"Final horizontal position: {horizontal}")
    print(f"Final depth: {depth}")
    print(f'Product: {horizontal * depth}')


if __name__ == "__main__":
    main()
