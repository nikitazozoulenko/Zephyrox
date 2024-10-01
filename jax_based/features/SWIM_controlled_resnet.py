
from typing import Tuple, List, Union, Any, Optional, Dict, Set, Literal, Callable
from abc import ABC, abstractmethod
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
from jaxtyping import Array, Float, Int, PRNGKeyArray

from features.base import TimeseriesFeatureTransformer
from features.SWIM_mlp import init_single_SWIM_layer


def step_forward_controlled_resnet(
        Z: Float[Array, "N  d"],
        XTdiff: Float[Array, "N  D"],
        W: Float[Array, "d  d  D"],
        b: Float[Array, "d  D"],
        activation: Callable = jnp.tanh,
        delta_scale: Float = 1.0,
    ):
    """
    Single T=1 step forward of a Controlled ResNet.
    """
    Z0 = Z
    Z = jnp.dot(Z, W) + b
    Z = activation(Z) * XTdiff[:, None, :]
    Z = Z.sum(axis=-1)
    return Z0 + delta_scale * Z



def forward(
        Z0: Float[Array, "N  d"],
        X: Float[Array, "N  T  D"], 
        weights: Float[Array, "T  d  d  D"],
        biases: Float[Array, "T  1  d  D"], 
        activation = lambda x : jnp.maximum(0,x+0.5), # jnp.tanh,
    ):
    """
    Forward of a Controlled ResNet.
    """
    T, d, d, D = weights.shape
    Xdiff = jnp.diff(X, axis=1)

    def scan_body(carry, x):
        Z = carry
        (W,b), XTdiff = x
        return step_forward_controlled_resnet(
            Z, XTdiff, W, b, activation, 1/T
            ), None

    Z, _ = lax.scan(
        scan_body, 
        Z0, 
        xs=( (weights, biases), Xdiff.transpose(1,0,2) )
        )
    return Z



def get_init_state(
        X: Float[Array, "N  T  D"],
        w: Float[Array, "T*D  d"],
        b: Float[Array, "1  d"],
        activation = jnp.tanh, # lambda x : jnp.maximum(0,x+0.5),
    ):
    """
    Initializes Z0 by randomly projecting the flattened time series.
    """
    N, T, D = X.shape
    return activation(X.reshape(N, T*D) @ w + b)



def init_single_CRN_layer(
        seed: PRNGKeyArray,
        Z: Float[Array, "N  d"],
        XTdiff: Float[Array, "N  D"],
        y: Float[Array, "N  p"],
    ) -> Tuple[Float[Array, "d  d  D"], Float[Array, "1  d  D"]]:
    """
    Initializes one 'layer' of the Controlled ResNet model.
    """
    N, D = XTdiff.shape
    N, d = Z.shape
    w, b = init_single_SWIM_layer(Z, y, D*d, seed)
    w = w.reshape(d, d, D) #i think this reshape is fine
    b = b.reshape(1, d, D)
    return w,b



class SampledControlledResNet(TimeseriesFeatureTransformer):
    def __init__(
            self,
            seed: PRNGKeyArray,
            n_features: int,
            activation: Callable = jnp.tanh,
            max_batch: int = 512,
            transform_label_to_onehot: bool = False,
        ):
        """Controlled ResNet (or randomized signature) initialized with 
        gradient-based sampling (SWIM). NOTE Initial state Z_0 is hard-coded
        to be a (fixed) random projection of whole time series.

        Args:
            seed (PRNGKeyArray): Random seed for matrices, biases, initial value.
            n_features (int): Number of features of the path development Z.
            max_batch (int): Max batch size for computations.
            transform_label_to_onehot (bool): Whether to transform labels to one-hot,
                which is required for classification tasks.
        """
        super().__init__(max_batch)
        self.seed = seed
        self.n_features = n_features
        self.activation = activation
        self.transform_label_to_onehot = transform_label_to_onehot
        self.weights = None
        self.biases = None


    def fit(
            self, 
            X: Float[Array, "N  T  D"], 
            y: Float[Array, "N  p"]
        ):
        """
        Initializes the weights and biases, using SWIM algorithm in the 
        Controlled ResNet setting. 

        Args:
            X (Float[Array, "N  T  D"]): Input training data.
            y (Float[Array, "N  T  p"]): Target training data.
        """
        # Get shape, dtype
        N, T, D = X.shape
        seedZ0w, seedZ0b, seedRes = jax.random.split(self.seed, 3)

        # Transform to one-hot if needed for classification fitting.
        if self.transform_label_to_onehot and (y.ndim == 1 or y.shape[-1] == 1):
            y = jax.nn.one_hot(y, num_classes=len(jnp.unique(y))) # TODO i think this should work

        #obtain random projection matrix for Z0 initialization #NOTE i think this is kaiming initialization
        self.proj_w = jax.random.normal(seedZ0w, (D*T, self.n_features)) / np.sqrt(D*T/2)
        self.proj_b = jax.random.normal(seedZ0b, (1, self.n_features)) * np.sqrt(2)
        Z0 = get_init_state(X, self.proj_w, self.proj_b)
        Xdiff = jnp.diff(X, axis=1)

        #the controlled resnet part
        def scan_body(carry, x):
            Z = carry
            XTdiff, seed = x
            w, b = init_single_CRN_layer(seed, Z, XTdiff, y)
            return step_forward_controlled_resnet(Z, XTdiff, w, b, self.activation, 1/(T-1)), (w,b)

        carry = Z0
        Z, (self.weights, self.biases) = lax.scan(
            scan_body, 
            carry, 
            xs=(Xdiff.transpose(1,0,2), jax.random.split(seedRes, T-1))
            )
        return self


    def _batched_transform(
            self,
            X: Float[Array, "N  T  D"],
        ) -> Float[Array, "N  n_features"]:

        return forward(
            get_init_state(X, self.proj_w, self.proj_b),
            X,
            self.weights,
            self.biases,
            self.activation
            )
    




