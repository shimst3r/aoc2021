from pathlib import Path

import numpy as np
from tqdm import tqdm


def load_data():
    root = Path(__file__).parent

    with open(root / "input.txt") as in_file:
        data = np.fromstring(in_file.read(), sep=",", dtype=int)
    return data


def proliferate(initial_state, days):
    state = np.bincount(initial_state, minlength=9)
    for _ in tqdm(range(days)):
        state = np.roll(state, -1)
        state[6] += state[8]
    return state


def task1():
    initial_state = load_data()
    print(np.sum(proliferate(initial_state, 80)))


def task2():
    initial_state = load_data()
    print(np.sum(proliferate(initial_state, 256)))


def main():
    print("----- Task 1 -----")
    task1()
    print("----- Task 2 -----")
    task2()


if __name__ == "__main__":
    main()
