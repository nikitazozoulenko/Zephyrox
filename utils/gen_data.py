import numpy as np
import jax
import jax.numpy as jnp


def gen_BM(N:int, T:int, D:int, seed=0):
    """
    Generate N Brownian Motions of length T with D dimensions
    """
    key = jax.random.PRNGKey(seed)
    normal = jax.random.normal(key, (N, T, D))
    BM = jnp.cumsum(normal, axis=1)
    return BM