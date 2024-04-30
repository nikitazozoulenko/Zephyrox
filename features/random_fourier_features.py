from typing import Tuple, List, Union, Any, Optional, Dict, Literal

import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
from jaxtyping import Array, Float, Int, PRNGKeyArray

from features.base import TimeseriesFeatureTransformer


@jax.jit
def random_fourier_features(
        X: Float[Array, "N  D"],
        random_matrix: Float[Array, "D  n_features//2"],
    ) -> Float[Array, "N  n_features"]:
    """
    Computes the random Fourier features for the RBF kernel.
    Assumes the array 'random_matrix' has already been multiplied 
    by 1/sigma.

    Args:
        X (Float[Array, "N  D"]): Batched input array.
        random_matrix (Float[Array, "D  n_features//2"]): Random matrix.
    
    Returns:
        Float[Array, "N  n_features"]: Random Fourier features.
    """
    Y = jnp.dot(X, random_matrix)
    return jnp.concatenate([jnp.cos(Y), jnp.sin(Y)], axis=-1)
    
        

class RandomFourierFeatures(TimeseriesFeatureTransformer):

    def __init__(
            self, 
            seed: PRNGKeyArray,
            n_features: int = 500,
            sigma: float = 1.0,
            max_batch: int = 1000,
        ):
        """
        Random Fourier Features (RFF) for the RBF kernel. The RBF kernel is defined 
        on R^d via k(x, y) = exp(-||x-y||^2 / (2 * sigma^2)), and the RFF map is 
        z(x) =  [cos(w_1^T x), ... cos(w_D^T x), sin(w_1^T x), ..., sin(w_D^T x)],
        where w_i are drawn iid N(0, 1/sigma^2).

        Args:
            sigma (float): Scales the variance in the RBF kernel.
            n_features (int): Dimension of RFF map.
            max_batch (int): Maximum batch size for computations.
        """
        super().__init__(max_batch)
        self.seed = seed
        self.n_features = n_features
        self.sigma = sigma
    

    def fit(
            self, 
            X: Float[Array, "N  D"]
        ):
        """
        Initializes the random weights and biases for the RFF map. 
        The weights are drawn from a N(0, 1/(n_features*sigma)^2) 
        distribution.

        Args:
            X (Array): Input array of shape (N, d)
        """
        # Initialize the random matrix
        D = X.shape[-1]
        self.weights = jax.random.normal(self.seed, (D, self.n_features//2)) / self.sigma


    def _batched_transform(
            self, 
            X: Float[Array, "N  D"]
        ) -> Float[Array, "N  n_features"]:
        """
        Computes the RFF features for the given batched input array.

        Args:
            X (Float[Array, "N  D"]): Input array.
        
        Returns:
            Random Fourier features of shape (N, n_features).
        """
        return random_fourier_features(X, self.weights) / np.sqrt(2 * (self.n_features//2) )
