
from typing import Tuple, List, Union, Any, Optional, Dict, Set, Literal, Callable

import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
from jaxtyping import Array, Float, Int, PRNGKeyArray

from features.base import TimeseriesFeatureTransformer
from aeon.transformations.collection.convolution_based import Rocket, MultiRocket, MiniRocket

##########################################
####        ROCKET wrappers           ####
##########################################

class AbstractRocketWrapper(TimeseriesFeatureTransformer):
    def __init__(
            self,
            rocket,
            max_batch: int,
        ):
        super().__init__(max_batch)
        self.rocket = rocket


    def fit(self, X: Float[Array, "N  T  D"], y=None): #shape (N, T, D)
        self.rocket.fit(np.array(X).transpose(0,2,1))
        return self


    def _batched_transform(
            self,
            X: Float[Array, "N  T  D"],
        ) -> Float[Array, "N  n_features"]:
        X_np = np.array(X).transpose(0,2,1)
        features = self.rocket.transform(X_np)
        return jnp.array(features)
    


class RocketWrapper(AbstractRocketWrapper):
    def __init__(
            self,
            n_features: int = 3000,
            max_batch: int = 1000000,
        ):
        self.n_features = n_features
        super().__init__(
            Rocket(max(1, n_features//2)),
            max_batch
            )



class MiniRocketWrapper(AbstractRocketWrapper):
    def __init__(
            self,
            n_features: int = 3000,
            max_batch: int = 1000000,
        ):
        self.n_features = n_features
        super().__init__(
            MiniRocket(max(4, n_features)), 
            max_batch
            )



class MultiRocketWrapper(AbstractRocketWrapper):
    def __init__(
            self,
            n_features: int = 3000,
            max_batch: int = 1000000,
        ):
        """
        Wrapper for the MultiRocketTransform from the aeon library.
        Original paper: https://link.springer.com/article/10.1007/s10618-022-00844-1

        Args:
            max_batch (int): Maximum batch size for computations.
            n_features (int):  Number of random features.
        """
        self.n_features = n_features
        super().__init__(
            MultiRocket(max(84, n_features//8)), 
            max_batch
            )