from typing import Tuple, List, Union, Any, Optional, Dict, Set, Literal, Callable
from abc import ABC, abstractmethod

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
    Fits the weights for the SWIM model, iteratively layer by layer

    Args:
        X (Float[Array, "N  d"]): Previous layer's output.
        n_features (int): Next hidden layer size.
        seed (PRNGKeyArray): Random seed for the weights and biases.
    Returns:
        Weights (d, n_features) and biases (1, n_features) for the next layer.
    """
    seed_idxs, seed_sample = jax.random.split(seed, 2)
    N, d = X.shape

    #obtain pair indices
    n_pairs_pre = jnp.minimum(N * (N - 1), n_features)
    idx1 = jnp.arange(0, n_pairs_pre)
    delta = jax.random.randint(seed_idxs, shape=(n_pairs_pre,), minval=1, maxval=N)
    idx2 = (idx1 + delta) % N

    #calculate 'gradients'
    pairs_diff = X[idx2] - X[idx1]



    def sample_parameters(self, x, y, rng):
        """
        Sample directions from points to other points in the given dataset (x, y).
        """

        # n_repetitions repeats the sampling procedure to find better directions.
        # If we require more samples than data points, the repetitions will cause more pairs to be drawn.
        n_repetitions = max(1, int(np.ceil(self.layer_width / x.shape[0]))) * self.repetition_scaler

        # This guarantees that:
        # (a) we draw from all the N(N-1)/2 - N possible pairs (minus the exact idx_from=idx_to case)
        # (b) no indices appear twice at the same position (never idx0[k]==idx1[k] for all k)
        candidates_idx_from = rng.integers(low=0, high=x.shape[0], size=x.shape[0]*n_repetitions)
        delta = rng.integers(low=1, high=x.shape[0]-1, size=candidates_idx_from.shape[0])
        candidates_idx_to = (candidates_idx_from + delta) % x.shape[0]
        
        directions = x[candidates_idx_to, ...] - x[candidates_idx_from, ...]
        dists = np.linalg.norm(directions, axis=1, keepdims=True)
        dists = np.clip(dists, a_min=self.dist_min, a_max=None)
        directions = directions / dists

        dy = y[candidates_idx_to, :] - y[candidates_idx_from, :]
        if self.is_classifier:
            dy[np.abs(dy) > 0] = 1

        # We always sample with replacement to avoid forcing to sample low densities
        probabilities = self.weight_probabilities(dy, dists)
        selected_idx = rng.choice(dists.shape[0],
                                  size=self.layer_width,
                                  replace=True,
                                  p=probabilities)
        directions = directions[selected_idx]
        dists = dists[selected_idx]
        idx_from = candidates_idx_from[selected_idx]
        idx_to = candidates_idx_to[selected_idx]
        
        return directions, dists, idx_from, idx_to
    

    # diffs = jnp.diff(X, axis=0) # shape (T-1, D)
    # carry, _ = lax.scan(
    #     lambda carry, diff: innerscan_randomized_signature(carry, diff, A, b),
    #     Z_0,
    #     diffs
    # )
    # return carry





class SWIM_MLP(TimeseriesFeatureTransformer):
    def __init__(
            self,
            seed: PRNGKeyArray,
            n_features: int = 512,
            n_layers: int = 2,
            max_batch: int = 512,
        ):
        """Implementation of the original paper's SWIM model.
        https://gitlab.com/felix.dietrich/swimnetworks-paper/

        Args:
            seed (PRNGKeyArray): Random seed for matrices, biases, initial value.
            n_features (int): Hidden layer dimension.
            n_layers (int): Number of layers in the network.
            max_batch (int): Max batch size for computations.
        """
        super().__init__(max_batch)
        self.n_features = n_features
        self.n_layers = n_layers
        self.seed = seed


    def fit(self, X: Float[Array, "N  T  D"], y=None):
        """
        Initializes the random matrices and biases used in the 
        randomized signature kernel / neural sig kernel.

        Args:
            X (Float[Array, "N  T  D"]): Batched time series input.
        """
        # Get shape, dtype
        N, T, D = X.shape
        dtype = X.dtype
        seed_A, seed_b = jax.random.split(self.seed, 2)
        
        # Initialize the random matrices
        self.A = jax.random.normal(
            seed_A, 
            (self.n_features, self.n_features, D),
            dtype=dtype
        )
        self.A /= np.sqrt(1)

        # Initialize the random biases
        self.b = jax.random.normal(
            seed_b, 
            (self.n_features, D),
            dtype=dtype
        )

        #vmap the transform
        self.vmapped_transform = jax.vmap(
            lambda x: randomized_signature(x, self.A, self.b, self.Z0),
        )

        return self


    def _batched_transform(
            self,
            X: Float[Array, "N  T  D"],
        ) -> Float[Array, "N  n_features"]:
        return self.vmapped_transform(X)
























from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Tuple, Union
import numpy as np
from sklearn.base import BaseEstimator

@dataclass
class BaseSWIM(BaseEstimator, ABC):
    is_classifier: bool = False
    layer_width: int = None

    activation: Union[Callable[[np.ndarray], np.ndarray], str] = "none"
    weights: np.ndarray = None
    biases: np.ndarray = None
    n_parameters: int = 0

    input_shape: Tuple[int, ...] = None
    output_shape: Tuple[int, ...] = None

    @staticmethod
    def identity_activation(x):
        return x

    @staticmethod
    def relu_activation(x):
        return np.maximum(x, 0)

    @staticmethod
    def tanh_activation(x):
        return np.tanh(x)

    def __post_init__(self):
        self._classes = None

        if not isinstance(self.activation, Callable):
            if self.activation == "none" or self.activation is None:
                self.activation = BaseSWIM.identity_activation
            elif self.activation == "relu":
                self.activation = BaseSWIM.relu_activation
            elif self.activation == "tanh":
                self.activation = BaseSWIM.tanh_activation
            else:
                raise ValueError(f"Unknown activation {self.activation}.")

    @abstractmethod
    def fit(self, x, y=None):
        pass

    def transform(self, x, y=None):
        if self.layer_width is None:
            raise ValueError("The fit method did not set the number of outputs, i.e. layer_width.")
        
        x = self.prepare_x(x)
        result = self.activation(x @ self.weights + self.biases)
        return result

    def fit_transform(self, x, y=None):
        self.fit(x, y)
        return self.transform(x, y)

    def predict(self, x):
        return self.transform(x)
    
    def prepare_x(self, x):
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
        return x
    
    def prepare_y(self, y):
        """Prepares labels for the sampling.

        For the classification problem, applies one-hot encoding for the labels.
        For the regression problem, adds a dimension to the labels if neccesary.
        """
        if len(y.shape) < 2:
            y = y.reshape(-1, 1)

        if not self.is_classifier:
            return y, y 
        
        self._classes = np.unique(y)  
        n_classes = len(self._classes)
        y_encoded_index = np.argmax(y == self._classes, axis=1)
        y_encoded_onehot = np.eye(n_classes)[y_encoded_index]
        return y_encoded_onehot, y_encoded_index.reshape(-1, 1)  


    def prepare_y_inverse(self, y):
        """Inverse to prepare_y(self, y).

        For the classification problem, restores labels from the one-hot predictions.
        For the regression problem, has no effect on the labels.
        """
        if not self.is_classifier:
            return y 
        
        probability_max = np.argmax(y, axis=1)
        predictions = self._classes[probability_max].reshape(-1, 1)
        return predictions
    
    def clean_inputs(self, x, y):
        x = self.prepare_x(x)
        y, _ = self.prepare_y(y)
        return x, y
    

#######################  |
##### Dense Layer #####  |
####################### \|/

@dataclass
class DenseSWIM(BaseSWIM):
    parameter_sampler: Union[Callable, str] = "relu"
    sample_uniformly: bool = False
    random_seed: int = 1
    dist_min: np.float64 = 1e-10
    repetition_scaler: int = 1

    idx_from: np.ndarray = None
    idx_to: np.ndarray = None

    def __post_init__(self):
        super().__post_init__()
        
        if not isinstance(self.parameter_sampler, Callable):
            if self.parameter_sampler == "relu":
                self.parameter_sampler = self.sample_parameters_relu
            elif self.parameter_sampler == "tanh":
                self.parameter_sampler = self.sample_parameters_tanh
            elif self.parameter_sampler == "random":
                self.parameter_sampler = self.sample_parameters_randomly
            else:
                raise ValueError(f"Unknown parameter sampler {self.parameter_sampler}.")

    def fit(self, x, y=None):
        if self.layer_width is None:
            raise ValueError("layer_width must be set.")
        
        x, y = self.clean_inputs(x, y)
        rng = np.random.default_rng(self.random_seed)

        weights, biases, idx_from, idx_to = self.parameter_sampler(x, y, rng)

        self.idx_from = idx_from
        self.idx_to = idx_to
        self.weights = weights
        self.biases = biases

        self.n_parameters = np.prod(weights.shape) + np.prod(biases.shape)
        return self

    def sample_parameters_tanh(self, x, y, rng):
        scale = 0.5 * (np.log(1 + 1/2) - np.log(1 - 1/2))

        directions, dists, idx_from, idx_to = self.sample_parameters(x, y, rng)
        weights = (2 * scale * directions / dists).T
        biases = -np.sum(x[idx_from, :] * weights.T, axis=-1).reshape(1, -1) - scale

        return weights, biases, idx_from, idx_to

    def sample_parameters_relu(self, x, y, rng):
        scale = 1.0

        directions, dists, idx_from, idx_to = self.sample_parameters(x, y, rng)
        weights = (scale / dists.reshape(-1, 1) * directions).T
        biases = -np.sum(x[idx_from, :] * weights.T, axis=-1).reshape(1, -1)

        return weights, biases, idx_from, idx_to
    
    def sample_parameters_randomly(self, x, _, rng):
        weights = rng.normal(loc=0, scale=1, size=(self.layer_width, x.shape[1])).T
        biases = rng.uniform(low=-np.pi, high=np.pi, size=(self.layer_width, 1)).T
        idx0 = None
        idx1 = None
        return weights, biases, idx0, idx1
    
    def sample_parameters(self, x, y, rng):
        """
        Sample directions from points to other points in the given dataset (x, y).
        """

        # n_repetitions repeats the sampling procedure to find better directions.
        # If we require more samples than data points, the repetitions will cause more pairs to be drawn.
        n_repetitions = max(1, int(np.ceil(self.layer_width / x.shape[0]))) * self.repetition_scaler

        # This guarantees that:
        # (a) we draw from all the N(N-1)/2 - N possible pairs (minus the exact idx_from=idx_to case)
        # (b) no indices appear twice at the same position (never idx0[k]==idx1[k] for all k)
        candidates_idx_from = rng.integers(low=0, high=x.shape[0], size=x.shape[0]*n_repetitions)
        delta = rng.integers(low=1, high=x.shape[0]-1, size=candidates_idx_from.shape[0])
        candidates_idx_to = (candidates_idx_from + delta) % x.shape[0]
        
        directions = x[candidates_idx_to, ...] - x[candidates_idx_from, ...]
        dists = np.linalg.norm(directions, axis=1, keepdims=True)
        dists = np.clip(dists, a_min=self.dist_min, a_max=None)
        directions = directions / dists

        dy = y[candidates_idx_to, :] - y[candidates_idx_from, :]
        if self.is_classifier:
            dy[np.abs(dy) > 0] = 1

        # We always sample with replacement to avoid forcing to sample low densities
        probabilities = self.weight_probabilities(dy, dists)
        selected_idx = rng.choice(dists.shape[0],
                                  size=self.layer_width,
                                  replace=True,
                                  p=probabilities)
        directions = directions[selected_idx]
        dists = dists[selected_idx]
        idx_from = candidates_idx_from[selected_idx]
        idx_to = candidates_idx_to[selected_idx]
        
        return directions, dists, idx_from, idx_to
    

    def weight_probabilities(self, dy, dists):
        """Compute probability that a certain weight should be chosen as part of the network.
        This method computes all probabilities at once, without removing the new weights one by one.

        Args:
            dy: function difference
            dists: distance between the base points
            rng: random number generator

        Returns:
            probabilities: probabilities for the weights.
        """
        # compute the maximum over all changes in all y directions to sample good gradients for all outputs
        gradients = (np.max(np.abs(dy), axis=1, keepdims=True) / dists).ravel()

        if self.sample_uniformly or np.sum(gradients) < self.dist_min:
            # When all gradients are small, avoind dividing by a small number
            # and default to uniform distribution.
            probabilities = np.ones_like(gradients) / len(gradients)
        else:
            probabilities = gradients / np.sum(gradients)

        return probabilities
    


