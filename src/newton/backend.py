from enum import Enum, auto

import jax.numpy as jnp
import numpy

np = numpy


class Backend(Enum):
    JAX = auto()
    NUMPY = auto()


def set_backend(name: Backend):
    global np
    if name == Backend.JAX:
        np = jnp
    elif name == Backend.NUMPY:
        np = numpy
    else:
        raise ValueError(f"Unknown backend: {name}")
