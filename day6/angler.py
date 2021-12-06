import jax.numpy as jnp
from jax import jit as jjit
from numba import jit as njit

inp = """2,5,2,3,5,3,5,5,4,2,1,5,5,5,5,1,2,5,1,1,1,1,1,5,5,1,5,4,3,3,1,2,4,2,4,5,4,5,5,5,4,4,1,3,5,1,2,2,4,2,1,1,2,1,1,4,2,1,2,1,2,1,3,3,3,5,1,1,1,3,4,4,1,3,1,5,5,1,5,3,1,5,2,2,2,2,1,1,1,1,3,3,3,1,4,3,5,3,5,5,1,4,4,2,5,1,5,5,4,5,5,1,5,4,4,1,3,4,1,2,3,2,5,1,3,1,5,5,2,2,2,1,3,3,1,1,1,4,2,5,1,2,4,4,2,5,1,1,3,5,4,2,1,2,5,4,1,5,5,2,4,3,5,2,4,1,4,3,5,5,3,1,5,1,3,5,1,1,1,4,2,4,4,1,1,1,1,1,3,4,5,2,3,4,5,1,4,1,2,3,4,2,1,4,4,2,1,5,3,4,1,1,2,2,1,5,5,2,5,1,4,4,2,1,3,1,5,5,1,4,2,2,1,1,1,5,1,3,4,1,3,3,5,3,5,5,3,1,4,4,1,1,1,3,3,2,3,1,1,1,5,4,2,5,3,5,4,4,5,2,3,2,5,2,1,1,1,2,1,5,3,5,1,4,1,2,1,5,3,5,2,1,3,1,2,4,5,3,4,3"""


def main():
    d = [0] * 9
    ord0 = ord("0")
    for c in inp:
        if c != ",":
            d[ord(c) - ord0] += 1
    index_offset = 0
    for day in range(256):
        temp = d[(index_offset + 0) % 9]  # read number of reproducing fish
        index_offset += 1  # roll, ie move all one index lower
        d[(index_offset + 6) % 9] += temp  # reset 0 back to 6
        d[(index_offset + 8) % 9] = temp  # reproduce


@jjit
def jax_main():
    d = jnp.zeros(9)
    ord0 = ord("0")
    for c in inp:
        if c != ",":
            d = d.at[ord(c) - ord0].set(d[ord(c) - ord0] + 1)
    index_offset = 0
    for _ in jnp.arange(0, 256):
        temp = d[(index_offset + 0) % 9]
        index_offset += 1
        d = d.at[(index_offset + 6) % 9].set(d[(index_offset + 6) % 9] + temp)
        d = d.at[(index_offset + 8) % 9].set(temp)


numba_main = njit(main)
