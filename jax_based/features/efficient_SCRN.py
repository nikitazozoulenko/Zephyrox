
from typing import Tuple, List, Union, Any, Optional, Dict, Set, Literal, Callable
from abc import ABC, abstractmethod
from functools import partial
import time

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



def get_init_state_randproj(
        X: Float[Array, "N  T  D"],
        w: Float[Array, "T*D  d"],
        b: Float[Array, "1  d"],
        activation, # lambda x : jnp.maximum(0,x+0.5),
    ):
    """
    Initializes Z0 by randomly projecting the flattened time series.
    """
    N, T, D = X.shape
    return activation(X.reshape(N, T*D) @ w + b)



def memory_efficient_SCRN(
        X_train: Float[Array, "N  T  D"],
        y_train: Float[Array, "N  p"],
        X_test: Float[Array, "N1  T  D"],
        seed: PRNGKeyArray,
        n_features: int,
        activation: Callable = lambda x: jnp.maximum(0,x+0.5), #jnp.tanh,
        without_dx: bool = False,
    ):
    """Memory efficient fit and transform of the SampledControlledResNet.
    Does not save any model weights, throws away the previous timestep's
    values and weights after each iteration.
    """
    N1, T, D = X_train.shape
    N2, T, D = X_test.shape
    seedZ0w, seedZ0b, seedRes = jax.random.split(seed, 3)
    t0 = time.time()

    # Transform to one-hot if needed for classification fitting.
    if y_train.ndim == 1:
        y_train = jax.nn.one_hot(y_train, num_classes=len(jnp.unique(y_train)))
    
    #obtain random projection matrix for Z0 initialization
    proj_w = jax.random.normal(seedZ0w, (D*T, n_features)) / np.sqrt(D*T)
    proj_b = jax.random.normal(seedZ0b, (1, n_features))
    Z0_train = get_init_state_randproj(X_train, proj_w, proj_b, activation)
    Z0_test = get_init_state_randproj(X_test, proj_w, proj_b, activation)

    #the controlled resnet part
    Xdiff_train = jnp.diff(X_train, axis=1)
    Xdiff_test = jnp.diff(X_test, axis=1)

    def scan_body(carry, x):
        Z_train, Z_test = carry
        XTdiff_train, XTdiff_test, seed = x
        w, b = init_single_CRN_layer(seed, Z_train, XTdiff_train, y_train)
        Z_train = step_forward_controlled_resnet(Z_train, XTdiff_train, w, b, activation)
        Z_test = step_forward_controlled_resnet(Z_test, XTdiff_test, w, b, activation)
        return (Z_train, Z_test), None

    if without_dx:
        Xdiff_test = jnp.ones_like(Xdiff_test) / (T-1)
        Xdiff_train = jnp.ones_like(Xdiff_train) / (T-1)

    (Z_train, Z_test), _ = lax.scan(
        scan_body, 
        (Z0_train, Z0_test), 
        xs=(Xdiff_train.transpose(1,0,2), Xdiff_test.transpose(1,0,2), jax.random.split(seedRes, T-1))
        )
    t1 = time.time()

    return Z_train, Z_test, t1-t0, t1-t0