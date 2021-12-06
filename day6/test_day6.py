import numpy as np
import pytest

from .src import proliferate


@pytest.fixture
def initial_state():
    return np.fromstring("3,4,3,1,2", sep=",", dtype=int)


def test_number_of_fishes_after_18_days(initial_state):
    actual_number = np.sum(proliferate(initial_state, days=18))

    assert actual_number == 26


def test_number_of_fishes_after_80_days(initial_state):
    actual_number = np.sum(proliferate(initial_state, days=80))

    assert actual_number == 5934


def test_number_of_fishes_after_256_days(initial_state):
    actual_number = len(proliferate(initial_state, days=256))

    assert actual_number == 26984457539
