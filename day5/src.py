from pathlib import Path

import numpy as np


def main():
    print("----- Task 1 -----")
    task1()


def task1():
    data = _load_data()
    diagram = Diagram.from_input(data)
    print(f"Number of overlapping points: {diagram.count_overlapping_points()}")


class Diagram:
    def __init__(self, coordinates: np.ndarray):
        self.coordinates = coordinates

    def __str__(self):
        return "\n".join(
            [
                "".join([str(int(num)) if num > 0 else "." for num in row])
                for row in self.coordinates
            ]
        )

    @classmethod
    def from_input(cls, input):
        directions = _parse_input(input)
        coordinates = _compute_coordinates(directions)
        return cls(coordinates)

    @property
    def overlapping_points(self):
        return _overlapping_points(self.coordinates)

    def count_overlapping_points(self):
        return np.sum(self.overlapping_points)


def _compute_coordinates(directions):
    x_max, y_max = _max_directions(directions)
    coordinates = np.zeros((x_max, y_max))
    for [[x_s, y_s], [x_e, y_e]] in directions:
        if not (x_s == x_e or y_s == y_e):
            continue
        x_s, y_s, x_e, y_e = _swap_indices(x_s, y_s, x_e, y_e)
        coordinates[y_s : (y_e + 1), x_s : (x_e + 1)] += 1
    return coordinates


def _load_data():
    root = Path(__file__).parent

    with open(root / "input.txt") as in_file:
        data = in_file.read()
    return data


def _max_directions(directions):
    x_max = directions[:, :, 0].max() + 1
    y_max = directions[:, :, 1].max() + 1

    return x_max, y_max


def _overlapping_points(coordinates: np.ndarray):
    return coordinates >= 2


def _parse_input(input):
    directions = []
    # skip the last line as it's empty
    for line in input.split("\n")[:-1]:
        direction = _parse_line(line)
        directions.append(direction)
    return np.array(directions)


def _parse_line(line):
    start, end = line.split(" -> ")
    x_start, y_start = start.split(",")
    x_end, y_end = end.split(",")
    return np.array([[x_start, y_start], [x_end, y_end]], dtype=int)


def _swap_indices(x_s, y_s, x_e, y_e):
    if x_s > x_e:
        x_s, x_e = x_e, x_s
    if y_s > y_e:
        y_s, y_e = y_e, y_s
    return x_s, y_s, x_e, y_e


if __name__ == "__main__":
    main()
