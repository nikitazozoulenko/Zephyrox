from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.functional as F
from torch import Tensor
from sklearn.base import TransformerMixin, BaseEstimator

###########################################  |
########### Abstract Base Class ###########  |
########################################### \|/

class TimeseriesFeatureExtractor(ABC, TransformerMixin, BaseEstimator):
    def __init__(self, max_batch: int = 512):
        """Abstract base class for time series feature extractors 
        Classes should implement 'fit' and '_batched_transform' methods.

        Args:
            max_batch (int): Maximum chunk size for computations.
        """
        self.max_batch = max_batch


    @abstractmethod
    def fit(self, X: Tensor, y=None):
        """Fit the transformer to tabular training data.

        Args:
            X (Tensor): Batched time series data of shape (N, T, D).
        """
        pass


    @abstractmethod
    def _batched_transform(self, X: Tensor) -> Tensor:
        """Transforms the input time series data into features.

        Args:
            X (Tensor): Batched time series tensor of shape (N,T,D)

        Returns:
            (Tensor): The time series features.
        """
        pass

    
    def transform(self, X: Tensor) -> Tensor:
        """Transform the input time series data into features. Splits the
        data into sub-batches if necessary based on 'self.max_batch'.

        Args:
            X (Tensor): Batched time series tensor of shape (N,T,D)

        Returns:
            (Tensor): Feature vectors of shape (N, ...)
        """
        split_X = torch.split(X, self.max_batch, dim=0)
        return torch.cat(
            [self._batched_transform(x) for x in split_X],
            axis=0
            )



########################################  |
########### Tabular Features ###########  |
######################################## \|/


class TabularTimeseriesFeatures(TimeseriesFeatureExtractor):
    def __init__(
            self,
            max_batch: int = 100000,
        ):
        """
        Flattens time series to a big vector in R^TD.

        Args:
            max_batch (int): Maximum batch size for computations.
        """
        super().__init__(max_batch)


    def fit(self, X: Tensor, y=None):
        return self


    def _batched_transform(self, X: Tensor) -> Tensor:
        N = X.shape[0]
        return X.reshape(N, -1)


###################################  |
########### Random Guess ##########  |
################################### \|/


class RandomGuesser(TimeseriesFeatureExtractor):
    def __init__(
            self,
            seed: int = 0,
            n_features: int = 2,
            max_batch: int = 1000000,
        ):
        """
        Class that generates random normal features independent 
        of input data, containing no meaningful information.

        Args:
            seed (int): Random seed for random features.
            n_features (int): Number of random features.
            max_batch (int): Maximum batch size for computations.
        """
        super().__init__(max_batch)
        self.n_features = n_features
        

    def fit(self, X, y=None):
        return self


    def _batched_transform(self, X: Tensor) -> Tensor:
        N = X.shape[0]
        return torch.randn(N, 
                        self.n_features,
                        device=X.device,
                        dtype=X.dtype,
                        )
