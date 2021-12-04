from itertools import pairwise
from pathlib import Path

root_dir = Path(__file__).parent


def task1():
    data = load_data()
    x, y = data[:-1], data[1:]
    result = sum(1 for xs, ys in zip(x, y) if ys > xs)
    print(result)


def task2():
    data = load_data()
    x, y = triplewise(data[:-1]), triplewise(data[1:])
    result = sum(1 for xs, ys in zip(x, y) if sum(ys) > sum(xs))
    print(result)


def load_data():
    with open(root_dir/'input', 'r', encoding='utf8') as input_file:
        data = [int(row) for row in input_file]

    return data


def triplewise(iterable):
    "Return overlapping triplets from an iterable"
    # triplewise('ABCDEFG') -> ABC BCD CDE DEF EFG
    for (a, _), (b, c) in pairwise(pairwise(iterable)):
        yield a, b, c


if __name__ == '__main__':
    task1()
    task2()
