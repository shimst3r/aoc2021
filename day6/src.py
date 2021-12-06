from pathlib import Path

import numpy as np
from tqdm import tqdm


def load_data():
    root = Path(__file__).parent

    with open(root / "input.txt") as in_file:
        data = np.fromstring(in_file.read(), sep=",", dtype=int)
    return data


def proliferate(initial_state, days):
    state = np.array(initial_state, copy=True)
    for _ in tqdm(range(days)):
        state = state - 1
        state = np.append(
            state,
            np.array([8 for _ in range(len(state[state < 0]))]),
        )
        state[state < 0] = 6
    return state


def task1():
    initial_state = load_data()
    print(len(proliferate(initial_state, 80)))


def task2():
    initial_state = load_data()
    print(len(proliferate(initial_state, 256)))


def main():
    print("----- Task 1 -----")
    task1()
    print("----- Task 2 -----")
    task2()


if __name__ == "__main__":
    main()
