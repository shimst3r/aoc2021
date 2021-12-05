import pytest

from .src import Diagram


@pytest.fixture
def input():
    return """0,9 -> 5,9
8,0 -> 0,8
9,4 -> 3,4
2,2 -> 2,1
7,0 -> 7,4
6,4 -> 2,0
0,9 -> 2,9
3,4 -> 1,4
0,0 -> 8,8
5,5 -> 8,2
"""


@pytest.fixture
def diagram():
    return """.......1..
..1....1..
..1....1..
.......1..
.112111211
..........
..........
..........
..........
222111...."""


def test_diagram_from_input(input, diagram):
    actual_diagram = Diagram.from_input(input)

    assert str(actual_diagram) == diagram


def test_no_diagonal_overlapping(input):
    diagram = Diagram.from_input(input)

    assert diagram.count_overlapping_points() == 5
