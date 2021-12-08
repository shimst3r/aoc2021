from pathlib import Path

import numpy as np


def count(data):
    outputs = []
    for line in data.split("\n"):
        _, output = line.split(" | ")
        output = [len(out) for out in output.split()]
        outputs.append(output)
    flat_output = np.array(outputs).flatten()
    counts = np.bincount(flat_output)
    return np.sum(counts[[2, 3, 4, 7]])


def load_data():
    root = Path(__file__).parent

    with open(root / "input.txt") as in_file:
        data = in_file.read()
    return data


def task1():
    data = load_data()
    print(count(data))


if __name__ == "__main__":
    task1()
