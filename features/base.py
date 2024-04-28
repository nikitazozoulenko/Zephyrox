from typing import Tuple, List, Union, Any, Optional, Dict, Literal
from abc import ABC, abstractmethod

import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
from jaxtyping import Array, Float, Int
from sklearn.base import TransformerMixin, BaseEstimator

#######################################  |
########### Helper function ###########  |
####################################### \|/

def split_maxbatch(
        X: Float[Array, "N  ..."], 
        max_batch: int
    ) -> Tuple[List[Array], Array]:
    """
    Splits the input array into batches of size max_batch,
    and a rest array with the remaining data.

    Args:
        X (Float[Array, "N  ..."]): Input array.
        max_batch (int): Maximum batch size.
    
    Returns:
        Tuple[List[Array], Array]: List of batches and the remaining data.
    """
    N = X.shape[0]
    n_batches = N // max_batch
    batched = X[:n_batches*max_batch].reshape(n_batches, max_batch, *X.shape[1:])
    rest = X[n_batches*max_batch:]
    return batched, rest

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
        batch_list, rest = split_maxbatch(X, self.max_batch)
        if batch_list.size == 0:
            batch_list = jnp.array([rest])
            rest = jnp.array([])

        carry, features = lax.scan(
            lambda c, x: (c, self._batched_transform(x)),
            None,
            batch_list
        )
        features = jnp.concatenate(features, axis=0)

        if rest.size > 0:
            rest_features = self._batched_transform(rest)
            features = jnp.concatenate([features, rest_features], axis=0)
        
        return features