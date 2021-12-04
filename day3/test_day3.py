import pytest

from .src import (
    compute_co2_scrubber_rating,
    compute_epsilon,
    compute_gamma,
    compute_oxygen_generator_rating,
)


@pytest.fixture
def input():
    return """
    00100
    11110
    10110
    10111
    10101
    01111
    00111
    11100
    10000
    11001
    00010
    01010
    """


def test_compute_co2_scrubber_rating(input):
    rating = compute_co2_scrubber_rating(input)

    assert rating == 10


def test_compute_epsilon(input):
    epsilon = compute_epsilon(input)

    assert epsilon == 9


def test_compute_gamma(input):
    gamma = compute_gamma(input)

    assert gamma == 22


def test_compute_oxygen_generator_rating(input):
    rating = compute_oxygen_generator_rating(input)

    assert rating == 23
