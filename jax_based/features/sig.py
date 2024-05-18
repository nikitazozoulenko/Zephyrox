from typing import Tuple, List, Union, Any, Optional, Dict, Literal
from abc import ABC, abstractmethod

import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
from jaxtyping import Array, Float, Int
from signax import signature, logsignature

from features.base import TimeseriesFeatureTransformer


class SigTransform(TimeseriesFeatureTransformer):
    def __init__(
            self,
            trunc_level: int = 3, #signature truncation level
            max_batch: int = 512,
        ):
        """
        Signature feature class.

        Args:
            trunc_level (int): Signature truncation level.
            max_batch (int): Maximum chunk size for computations.
        """
        super().__init__(max_batch)
        self.trunc_level = trunc_level
        self.max_batch = max_batch


    def fit(self, X, y=None):
        return self


    def _batched_transform(self, X: Array) -> Array:
        return signature(X, self.trunc_level)


class LogSigTransform(TimeseriesFeatureTransformer):
    def __init__(
            self,
            trunc_level: int = 3, #signature truncation level
            max_batch: int = 512,
        ):
        """
        Log-signature feature class.

        Args:
            trunc_level (int): Signature truncation level.
            max_batch (int): Maximum chunk size for computations.
        """
        super().__init__(max_batch)
        self.trunc_level = trunc_level
        self.max_batch = max_batch


    def fit(self, X, y=None):
        return self


    def _batched_transform(self, X: Array) -> Array:
        return logsignature(X, self.trunc_level)