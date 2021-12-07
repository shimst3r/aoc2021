from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar


def align(input):
    arr = np.fromstring(input, sep=",")

    def min_fun(x):
        return np.sum(np.abs(arr - x))

    return min_fun


def align2(input):
    arr = np.fromstring(input, sep=",")

    def min_fun(x):
        return np.sum([(xs * (xs + 1)) / 2.0 for xs in np.abs(arr - x)])

    return min_fun


def load_data():
    root = Path(__file__).parent
    with open(root / "input.txt") as in_file:
        data = in_file.read()
    return data


def optimize(input, align_fn):
    min_fun = align_fn(input)
    res = minimize_scalar(min_fun)
    return round(res.x), min_fun(round(res.x))


def task1():
    data = load_data()
    optimal_x, optimal_fuel = optimize(data, align)
    print(f"Optimal horizontal position: {optimal_x}")
    print(f"Minimal fuel consumption: {optimal_fuel}")


def task2():
    data = load_data()
    optimal_x, optimal_fuel = optimize(data, align2)
    print(f"Optimal horizontal position: {optimal_x}")
    print(f"Minimal fuel consumption: {optimal_fuel}")


if __name__ == "__main__":
    print("----- Task 1 -----")
    task1()
    print("----- Task 2 -----")
    task2()
