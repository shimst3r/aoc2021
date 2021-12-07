import pytest

from .src import optimize


@pytest.fixture
def input():
    return "16,1,2,0,4,2,7,1,2,14"


def test_align(input):
    actual_consumption = optimize(input)

    assert actual_consumption == 37
