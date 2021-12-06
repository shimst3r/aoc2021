from functools import partial
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import jit as jjit
from numba import jit as njit


def load_data():
    root = Path(__file__).parent

    with open(root / "input.txt") as in_file:
        data = np.fromstring(in_file.read(), sep=",", dtype=int)
    return data


def proliferate(initial_state, days):
    state = np.bincount(initial_state, minlength=9)
    for _ in range(days):
        state = np.roll(state, -1)
        state[6] += state[8]
    return state


numba_proliferate = njit(proliferate)


@partial(jjit, static_argnums=1)
def jax_proliferate(initial_state, days):
    arr = jnp.asarray(initial_state, dtype=jnp.int64)
    state = jnp.bincount(arr, length=9)
    for _ in jnp.arange(0, days):
        state = jnp.roll(state, -1)
        state = state.at[6].set(state[6] + state[8])
    return state


def task1():
    initial_state = load_data()
    print(np.sum(proliferate(initial_state, 80)))


def task2():
    initial_state = load_data()
    print(np.sum(proliferate(initial_state, 256)))


def main():
    print("----- Task 1 -----")
    task1()
    print("----- Task 2 -----")
    task2()


if __name__ == "__main__":
    main()
