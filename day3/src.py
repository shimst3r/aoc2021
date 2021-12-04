from pathlib import Path

import numpy as np


def compute_bits(input, arg_fn):
    arr = np.array([[int(num) for num in row] for row in input.split()])
    bits = [arg_fn(row) for row in arr.T]
    binary_representation = "".join(str(bit) for bit in bits)
    return int("".join(binary_representation), 2)


def compute_co2_scrubber_rating(input):
    arr = np.array([[int(num) for num in row] for row in input.split()])
    position = 0
    while len(arr) > 1:
        bincount = np.bincount(arr[:, position])
        if bincount[0] == bincount[1]:
            argmin = 0
        else:
            argmin = bincount.argmin()
        arr = arr[arr[:, position] == argmin]
        position += 1
    binary_representation = "".join(str(bit) for bit in arr[0])
    return int("".join(binary_representation), 2)


def compute_epsilon(input):
    def arg_fn(x):
        return np.bincount(x).argmin()

    return compute_bits(input, arg_fn)


def compute_gamma(input):
    def arg_fn(x):
        return np.bincount(x).argmax()

    return compute_bits(input, arg_fn)


def compute_oxygen_generator_rating(input):
    arr = np.array([[int(num) for num in row] for row in input.split()])
    position = 0
    while len(arr) > 1:
        bincount = np.bincount(arr[:, position])
        if len(bincount) > 1 and bincount[0] == bincount[1]:
            argmax = 1
        else:
            argmax = bincount.argmax()
        arr = arr[arr[:, position] == argmax]
        position += 1
    binary_representation = "".join(str(bit) for bit in arr[0])
    return int("".join(binary_representation), 2)


def load_data():
    root_path = Path(__file__).parent
    with open(root_path / "input.txt") as in_file:
        data = in_file.read()
    return data


def task1():
    data = load_data()
    epsilon = compute_epsilon(data)
    gamma = compute_gamma(data)
    print(epsilon * gamma)


def task2():
    data = load_data()
    oxygen_generator_rating = compute_oxygen_generator_rating(data)
    co2_scrubber_rating = compute_co2_scrubber_rating(data)
    print(oxygen_generator_rating * co2_scrubber_rating)


def main():
    task1()
    task2()


if __name__ == "__main__":
    main()
