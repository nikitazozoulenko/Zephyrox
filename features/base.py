from typing import Tuple, List, Union, Any, Optional, Dict, Literal
from abc import ABC, abstractmethod

import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
from jaxtyping import Array, Float, Int
from sklearn.base import TransformerMixin, BaseEstimator
from netket.jax import apply_chunked


###########################################  |
########### Abstract Base Class ###########  |
########################################### \|/

class TimeseriesFeatureTransformer(ABC, TransformerMixin, BaseEstimator):
    def __init__(self, max_batch: int = 512):
        self.max_batch = max_batch # Maximum batch size for computations. 


    @abstractmethod
    def fit(self, X: Float[Array, "N  T  D"]):
        """
        Fit the transformer to the training data.

        Args:
            X (Float[Array, "N T D"]): Batched time series data.
        """
        pass


    @abstractmethod
    def _batched_transform(
        self, 
        X: Float[Array, "N  T  D"],
    ) -> Float[Array, "N  ..."]:
        """
        Transform the input time series data into features.

        Args:
            X (Float[Array, "N  T  D"]): Batched time series.

        Returns:
            (Float[Array, "N  ..."]): The time series features.
        """
        pass

    
    def transform(self, X: Float[Array, "N  T  D"]) -> Float[Array, "N  ..."]:
        """
        Transform the input time series data into features. Splits the
        data into sub-batches if necessary based on 'max_batch'.

        Args:
            X (Float[Array, "N  T  D"]): Batched time series data.

        Returns:
            (Float[Array, "N  ..."]): The batched time series features.
        """
        trans_chunked = apply_chunked(
            self._batched_transform,
            chunk_size=self.max_batch
        )
        return trans_chunked(X)