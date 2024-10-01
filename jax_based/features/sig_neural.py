from typing import Tuple, List, Union, Any, Optional, Dict, Set, Literal, Callable
from abc import ABC, abstractmethod

import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
from jaxtyping import Array, Float, Int, PRNGKeyArray

from features.base import TimeseriesFeatureTransformer


@jax.jit
def scanbody_randomized_signature(
        Z: Float[Array, "n_features"],
        diff: Float[Array, "D"],
        A: Float[Array, "n_features  n_features  D"],
        b: Float[Array, "n_features  D"],
        activation:Callable = jnp.tanh,
        delta_scale: Float = 1.0,
):
    """
    Inner loop of the randomized signature kernel / neural sig kernel.

    Args:
        Z (Float[Array, "n_features"]): Time t value of the randomized signature.
        diff (Float[Array, "D"]): Difference of the time series.
        A (Float[Array, "n_features  n_features  D"]): Random matrix.
        b (Float[Array, "n_features  D"]): Random bias.
        activation (Callable): Activation function.
        delta_scale (Float): Scaling factor for Euler discretization.
    """
    Z_0 = Z
    Z = jnp.dot(Z, A) + b
    Z = activation(Z) * diff[None, :]
    Z = Z.mean(axis=-1)
    return Z_0 + delta_scale*Z, None


def randomized_signature(
        X: Float[Array, "T  D"],
        A: Float[Array, "n_features  n_features  D"],
        b: Float[Array, "n_features  D"],
        Z_0: Float[Array, "n_features"],
    ) -> Float[Array, "n_features"]:
    """
    Randomized signature of a single time series X, with tanh
    activation function.

    Args:
        X (Float[Array, "T  D"]): Input tensor of shape (T, d).
        A (Float[Array, "n_features  n_features  D"]): Random matrix.
        b (Float[Array, "n_features  D"]): Random bias.
        Z_0 (Float[Array, "n_features"]): Initial value of the randomized signature.
    """
    T = X.shape[0]
    diffs = jnp.diff(X, axis=0) # shape (T-1, D)
    carry, _ = lax.scan(
        lambda carry, diff: scanbody_randomized_signature(carry, diff, A, b),
        Z_0,
        diffs
    )
    return carry


class RandomizedSignature(TimeseriesFeatureTransformer):
    def __init__(
            self,
            seed: PRNGKeyArray,
            n_features: int = 512,
            max_batch: int = 512,
        ):
        """Randomized signature / neural signature kernel with tanh
        activation.

        Args:
            seed (PRNGKeyArray): Random seed for matrices, biases, initial value.
            n_features (int): Dimension of feature vector.
            max_batch (int): Max batch size for computations.
        """
        super().__init__(max_batch)
        self.n_features = n_features
        self.seed = seed


    def fit(self, X: Float[Array, "N  T  D"], y=None):
        """
        Initializes the random matrices and biases used in the 
        randomized signature kernel / neural sig kernel.

        Args:
            X (Float[Array, "N  T  D"]): Batched time series input.
        """
        # Get shape, dtype
        N, T, D = X.shape
        dtype = X.dtype
        seed_A, seed_b, seed_Z0 = jax.random.split(self.seed, 3)
        
        # Initialize the random matrices
        self.A = jax.random.normal(
            seed_A, 
            (self.n_features, self.n_features, D),
            dtype=dtype
        )
        self.A /= np.sqrt(self.n_features)

        # Initialize the random biases
        self.b = jax.random.normal(
            seed_b, 
            (self.n_features, D),
            dtype=dtype
        )

        # Initialize the initial value of the randomized signature
        self.Z0 = jax.random.normal(
            seed_Z0, 
            (self.n_features,),
            dtype=dtype
        )

        #vmap the transform
        self.vmapped_transform = jax.vmap(
            lambda x: randomized_signature(x, self.A, self.b, self.Z0),
        )

        return self


    def _batched_transform(
            self,
            X: Float[Array, "N  T  D"],
        ) -> Float[Array, "N  n_features"]:
        return self.vmapped_transform(X)
    




##############################
##### time in-homogenous #####
##############################




@jax.jit
def time_inhomogenous_randomized_signature(
        X: Float[Array, "T  D"],
        A: Float[Array, "n_features  n_features  D"],
        b: Float[Array, "n_features  D"],
        Z_0: Float[Array, "n_features"],
    ) -> Float[Array, "n_features"]:
    """
    Randomized signature of a single time series X, with tanh
    activation function.

    Args:
        X (Float[Array, "T  D"]): Input tensor of shape (T, d).
        A (Float[Array, "n_features  n_features  D"]): Random matrix.
        b (Float[Array, "n_features  D"]): Random bias.
        Z_0 (Float[Array, "n_features"]): Initial value of the randomized signature.
    """
    T = X.shape[0]
    diffs = jnp.diff(X, axis=0) # shape (T-1, D)

    def scan_body(carry, x):
        Tdiff, TA, Tb = x
        return scanbody_randomized_signature(carry, Tdiff, TA, Tb)
    
    carry, _ = lax.scan(
        scan_body,
        Z_0,
        xs = (diffs, A, b)
    )
    return carry


class TimeInhomogenousRandomizedSignature(TimeseriesFeatureTransformer):
    def __init__(
            self,
            seed: PRNGKeyArray,
            n_features: int = 512,
            max_batch: int = 512,
        ):
        """Randomized signature / neural signature kernel with tanh
        activation.

        Args:
            seed (PRNGKeyArray): Random seed for matrices, biases, initial value.
            n_features (int): Dimension of feature vector.
            max_batch (int): Max batch size for computations.
        """
        super().__init__(max_batch)
        self.n_features = n_features
        self.seed = seed


    def fit(self, X: Float[Array, "N  T  D"], y=None):
        """
        Initializes the random matrices and biases used in the 
        randomized signature kernel / neural sig kernel.

        Args:
            X (Float[Array, "N  T  D"]): Batched time series input.
        """
        # Get shape, dtype
        N, T, D = X.shape
        dtype = X.dtype
        seed_A, seed_b, seed_Z0 = jax.random.split(self.seed, 3)
        
        # Initialize the random matrices
        self.A = jax.random.normal(
            seed_A, 
            (T-1, self.n_features, self.n_features, D),
            dtype=dtype
        )
        self.A /= np.sqrt(self.n_features)

        # Initialize the random biases
        self.b = jax.random.normal(
            seed_b, 
            (T-1, self.n_features, D),
            dtype=dtype
        )

        # Initialize the initial value of the randomized signature
        self.Z0 = jax.random.normal(
            seed_Z0, 
            (self.n_features,),
            dtype=dtype
        )

        #vmap the transform
        self.vmapped_transform = jax.vmap(
            lambda x: time_inhomogenous_randomized_signature(x, self.A, self.b, self.Z0),
        )

        return self


    def _batched_transform(
            self,
            X: Float[Array, "N  T  D"],
        ) -> Float[Array, "N  n_features"]:
        return self.vmapped_transform(X)