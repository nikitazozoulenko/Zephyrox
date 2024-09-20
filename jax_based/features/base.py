from typing import Tuple, List, Union, Any, Optional, Dict, Literal
from abc import ABC, abstractmethod

import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
from jaxtyping import Array, Float, Int, PRNGKeyArray
from sklearn.base import TransformerMixin, BaseEstimator
from netket.jax import apply_chunked


###########################################  |
########### Abstract Base Class ###########  |
########################################### \|/

class TimeseriesFeatureTransformer(ABC, TransformerMixin, BaseEstimator):
    def __init__(self, max_batch: int = 512):
        """Abstract base class for time series transformers 
        (feature extractors). Classes should implement 'fit'
        and '_batched_transform' methods.

        Args:
            max_batch (int): Maximum chunk size for computations.
        """
        self.max_batch = max_batch


    @abstractmethod
    def fit(self, X: Float[Array, "N  T  D"], y=None):
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
    

########################################  |
########### Tabular Features ###########  |
######################################## \|/


class TabularTimeseriesFeatures(TimeseriesFeatureTransformer):
    def __init__(
            self,
            max_batch: int = 1000000,
            epsilon: float = 0.00001,
        ):
        """
        Flattens time series to a big vector in R^Td.

        Args:
            max_batch (int): Maximum batch size for computations.
            epsilon (float): Small value to avoid division by zero.
        """
        super().__init__(max_batch)
        self.epsilon = epsilon


    def fit(self, X: Float[Array, "N  T  D"], y=None):
        self.mean = X.mean(axis=0, keepdims=True)
        self.std = X.std(axis=0, keepdims=True)


    def _batched_transform(
        self, 
        X: Float[Array, "N  T  D"],
    ) -> Float[Array, "N  T*D"]:
        N, T, D = X.shape
        X = (X - self.mean) / (self.std + self.epsilon)
        return X.reshape(N, -1)
    

###########################################################  |
########### Random Guess (Uninformed Transfomer) ##########  |
#####################################3##################### \|/


class RandomGuesser(TimeseriesFeatureTransformer):
    def __init__(
            self,
            seed : PRNGKeyArray = jax.random.PRNGKey(0),
            n_features: int = 512,
            max_batch: int = 1000000,
        ):
        """
        Class that generates random normal features 
        independent of input data, containing no meaningful 
        information.

        Args:
            seed (PRNGKeyArray): Random seed for random features.
            n_features (int): Number of random features.
            max_batch (int): Maximum batch size for computations.
        """
        super().__init__(max_batch)
        self.seed = seed
        self.n_features = n_features


    def fit(self, X, y=None):
        pass


    def _batched_transform(
        self, 
        X: Float[Array, "N  T  D"],
    ) -> Float[Array, "N  n_features"]:
        N, T, D = X.shape
        return jnp.array(np.random.randn(N, self.n_features))