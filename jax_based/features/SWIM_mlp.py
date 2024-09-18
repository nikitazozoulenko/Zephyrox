from typing import Tuple, List, Union, Any, Optional, Dict, Set, Literal, Callable
from abc import ABC, abstractmethod
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
from jaxtyping import Array, Float, Int, PRNGKeyArray

from features.base import TimeseriesFeatureTransformer



def init_single_SWIM_layer(
        X: Float[Array, "N  d"],
        y: Float[Array, "N  D"],
        n_features: int,
        seed: PRNGKeyArray,
    ) -> Tuple[Float[Array, "d  n_features"], Float[Array, "n_features"]]:
    """
    Fits the weights for a single layer of the SWIM model.

    Args:
        X (Float[Array, "N  d"]): Previous layer's output.
        n_features (int): Next hidden layer size.
        seed (PRNGKeyArray): Random seed for the weights and biases.
    Returns:
        Weights (d, n_features) and biases (1, n_features) for the next layer.
    """
    seed_idxs, seed_sample = jax.random.split(seed, 2)
    N, d = X.shape
    EPS = 1e-06

    #obtain pair indices
    n = 3*N
    idx1 = jnp.arange(0, n) % N
    delta = jax.random.randint(seed_idxs, shape=(n,), minval=1, maxval=N)
    idx2 = (idx1 + delta) % N

    #calculate 'gradients'
    dx = X[idx2] - X[idx1]
    dy = y[idx2] - y[idx1]
    dists = jnp.maximum(EPS, jnp.linalg.norm(dx, axis=1, keepdims=True) )
    gradients = (jnp.linalg.norm(dy, axis=1, keepdims=True) / dists ).reshape(-1)
    #gradients = (np.max(np.abs(dy), axis=1, keepdims=True) / dists ).reshape(-1) #NOTE paper uses this instead

    #sample pairs, weighted by gradients     
    selected_idx = jax.random.choice(
        seed_sample, 
        n,
        shape=(n_features,), 
        replace=True, # NOTE should this be a parameter?
        p=gradients/gradients.sum()
        )
    idx1 = idx1[selected_idx]
    dx = dx[selected_idx]
    dists = dists[selected_idx]
    
    #define weights and biases
    weights = (dx / dists**2).T
    biases = -jnp.sum(weights * X[idx1].T, axis=0, keepdims=True) - 0.5  # NOTE experiment with this. also +-0.5 ?
    return weights, biases



def forward_SWIM(
        X: Float[Array, "N  d"],
        weights: Float[Array, "d  n_features"],
        biases: Float[Array, "1  n_features"],
        add_residual: bool,
        activation = lambda x : jnp.maximum(0,x+0.5), # jnp.tanh,
    ) -> Float[Array, "N  n_features"]:
    """
    Forward pass for a single layer of the SWIM model.

    Args:
        X (Float[Array, "N  d"]): Input to the layer.
        weights (Float[Array, "d  n_features"]): Weights for the layer.
        biases (Float[Array, "1  n_features"]): Biases for the layer.
        add_residual (bool): Whether to use residual connections.
        activation (Callable): Activation function for the layer.
    Returns:
        Output of the layer of shape (N, n_features).
    """
    d, D = weights.shape
    X1 = activation(X @ weights + biases)
    if add_residual and d==D:
        return X1 + X
    else:
        return X1



def SWIM_all_layers(
        X0: Float[Array, "N  d"],
        y: Float[Array, "N  D"],
        n_features: int,
        n_layers: int,
        add_residual: bool,
        seed: PRNGKeyArray,
    ):
    """
    Fits the weights for the SWIM model, iteratively layer by layer

    Args:
        X0 (Float[Array, "N  d"]): First layer input.
        y (Float[Array, "N  p"]): Target training data.
        n_features (int): Hidden layer size.
        n_layers (int): Number of layers in the network.
        add_residual (bool): Whether to use residual connections.
        seed (PRNGKeyArray): Random seed for the weights and biases.
    Returns:
        Weights (d, n_features) and biases (1, n_features) for the next layer.
    """

    def scan_body(carry, seed): # (carry, x) -> (carry, y)
        X, y = carry
        w, b = init_single_SWIM_layer(X, y, n_features, seed)
        return (forward_SWIM(X, w, b, add_residual), y), (w, b)

    init_carry = (X0, y)
    # carry is (X, seed)
    carry, WaB = lax.scan(
        scan_body,
        init_carry,
        xs=jax.random.split(seed, n_layers),
    )
    return WaB



def all_forward(
        X: Float[Array, "N  d"], 
        w1: Float[Array, "d  D"],
        b1: Float[Array, "1  D"], 
        weights: Float[Array, "n_layers-1  d  D"],
        biases: Float[Array, "n_layers-1  1  D"], 
        n_layers:int,
        add_residual: bool,
        activation = lambda x : jnp.maximum(0,x+0.5), # jnp.tanh,
        #activation = jnp.tanh,
    ):
    """
    Forward pass for the SWIM model.

    Args:
        X (Float[Array, "N  d"]): Input to the model.
        w1 (Float[Array, "d  D"]): Weights for the first layer.
        b1 (Float[Array, "1  D"]): Biases for the first layer.
        weights (Float[Array, "n_layers-1  d  D"]): Weights for the remaining layers.
        biases (Float[Array, "n_layers-1  1  D"]): Biases for the remaining layers.
        n_layers (int): Number of layers in the network.
        add_residual (bool): Whether to use residual connections
        activation (Callable): Activation function for the network.
    Returns:
        Output of the model of shape (N, D).
    """
    #First hidden layer
    X = forward_SWIM(X, w1, b1, add_residual, activation)
    if n_layers == 1:
        return X
    #subsequent layers in a scan loop
    else:
        def scan_body(carry, t):
            X = carry
            w, b = weights[t], biases[t]
            return forward_SWIM(X, w, b, add_residual, activation), None

        X, _ = lax.scan(scan_body, X, xs=jnp.arange(n_layers-1))
        return X



class SWIM_MLP(TimeseriesFeatureTransformer):
    def __init__(
            self,
            seed: PRNGKeyArray,
            n_features: int = 512,
            n_layers: int = 3,
            add_residual: bool = False,
            max_batch: int = 512,
        ):
        """Implementation of the original paper's SWIM model
        https://gitlab.com/felix.dietrich/swimnetworks-paper/,
        but with support for residual connections.

        Args:
            seed (PRNGKeyArray): Random seed for matrices, biases, initial value.
            n_features (int): Hidden layer dimension.
            n_layers (int): Number of layers in the network.
            add_residual (bool): Whether to use residual connections.
            max_batch (int): Max batch size for computations.
        """
        super().__init__(max_batch)
        self.n_features = n_features
        self.n_layers = n_layers
        self.seed = seed
        self.add_residual = add_residual
        self.w1 = None
        self.b1 = None
        self.weights = None
        self.biases = None


    def fit(
            self, 
            X: Float[Array, "N  D"], 
            y: Float[Array, "N  d"]
        ):
        """
        Initializes MLP weights and biases, using SWIM algorithm.

        Args:
            X (Float[Array, "N  D"]): Input training data.
            y (Float[Array, "N  d"]): Target training data.
        """
        # Get shape, dtype
        N, D = X.shape
        seed1, seedrest = jax.random.split(self.seed, 2)

        #first do first layer, which cannot always be done in a scan loop
        self.w1, self.b1 = init_single_SWIM_layer(
            X, y, self.n_features, seed1
            )
        X = forward_SWIM(X, self.w1, self.b1, self.add_residual)
        
        #rest of the layers
        if self.n_layers > 1:
            self.weights, self.biases = SWIM_all_layers(
                X, y, self.n_features, self.n_layers-1, self.add_residual, seedrest
                )

        return self


    def _batched_transform(
            self,
            X: Float[Array, "N  T  D"],
        ) -> Float[Array, "N  n_features"]:

        return all_forward(
            X, self.w1, self.b1, self.weights, self.biases, self.n_layers, self.add_residual
            )