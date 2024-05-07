from typing import Tuple, List, Union, Any, Optional, Dict, Literal

import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
from jaxtyping import Array, Float, Int, PRNGKeyArray

from features.base import TimeseriesFeatureTransformer
from features.random_fourier_features import RandomFourierFeatures

################################################################  |
################# For the vanilla signature ####################  |
################################################################ \|/

def inner_sig_TRP_linear(
        carry_V: Float[Array, "n_features  T-1"],
        Pm_dot_Xdiff: Float[Array, "n_features  T-1"],
    ):
    """Carry function to be used in lax.scan for the linear signature kernel"""
    # (Xdiff @ Pm) * cumsum_shift1(V)
    carry_V = Pm_dot_Xdiff * (jnp.cumsum(carry_V, axis=1) - carry_V) 
    return carry_V, jnp.sum(carry_V, axis=1) #carry, output



@jax.jit
def linear_tensorised_random_projection_features(
        X: Float[Array, "T  D"],
        P: Float[Array, "trunc_level  n_features  D"],
    ) -> Float[Array, "trunc_level  n_features"]:
    """
    Uses tensorized random projections to compute a randomized
    feature vector of the (vanilla/linear) truncated signature kernel.
    Signature truncation level depends on the shape of P.

    Args:
        X (Float[Array, "T  D"]): 
            A single time series.
        P (Float[Array, "trunc_level  n_features  D"]): 
            Random matrices with i.i.d. standard Gaussian entries,
            for each truncation level of the signature.

    Returns:
        Float[Array, "trunc_level  n_features"]: 
            Feature vector of the time series.
    """
    D = P.shape[-1]
    Xdiff = jnp.diff(X, axis=0) / D**0.5 # shape (T-1, D)
    P_dot_Xdiff = jnp.dot(P, Xdiff.T) # shape (trunc_level, n_features, T-1)
    carry, V = lax.scan(inner_sig_TRP_linear, P_dot_Xdiff[0], P_dot_Xdiff[1:])

    #concatenate first level with the rest
    return jnp.concatenate(
        [jnp.sum(P_dot_Xdiff[0:1], axis=-1), V],
        axis=0
        )
         


class SigVanillaTensorizedRandProj(TimeseriesFeatureTransformer):
    def __init__(
            self,
            seed: PRNGKeyArray,
            n_features: int = 512,
            trunc_level: int = 3, #signature truncation level
            max_batch: int = 128,
            concat_levels: bool = True,
        ):
        """
        Transformer class for randomized vanilla truncated signature 
        features via tensorized random projections.

        Args:
            seed (PRNGKeyArray): Random seed for random matrices.
            n_features (int): Size of random projection.
            trunc_level (int): Signature truncation level.
            max_batch (int): Maximum batch size for computations.
            concat_levels (bool): Whether to concatenate the features 
                of each truncation level.
        """
        super().__init__(max_batch)
        self.n_features = n_features
        self.trunc_level = trunc_level
        self.seed = seed
        self.concat_levels = concat_levels


    def fit(self, X: Float[Array, "N  T  D"]):
        """
        Initializes the Tensorized Random Projections of the for 
        the vanilla signature (corresponding to the linear kernel). 
        This is a (trunc_level, n_features, D) i.i.d. standard 
        Gaussians random matrix.

        Args:
            X (Float[Array, "N  T  D"]): Example batched time series data.
        """
        # Get shape, dtype and device info.
        d = X.shape[-1]
        dtype = X.dtype
        
        #initialize the tensorized projection matrix for each truncation level
        self.P = jax.random.normal(
            self.seed, 
            (self.trunc_level, self.n_features, d),
            dtype=dtype
        )
        # self.P /= np.sqrt(self.n_features)

        #vmap the transform
        if self.concat_levels:
            self.vmapped_transform = jax.vmap(
                lambda x: jnp.concat(linear_tensorised_random_projection_features(x, self.P)[1:], axis=-1),
            )
        else:
            self.vmapped_transform = jax.vmap(
                lambda x: linear_tensorised_random_projection_features(x, self.P)[-1]
        )

        return self


    def _batched_transform(
            self,
            X: Float[Array, "N  T  D"],
        ) -> Float[Array, "N  n_features"]:
        """
        Computes the TRP features for the given batched input array.

        Args:
            X (Float[Array, "N  T  D"]): A single time series.
        
        Returns:
            Time series features of shape (N, n_features).
        """
        return self.vmapped_transform(X)
    
# ###################################################################  |
# ################# For the RBF-lifted signature ####################  |
# ################################################################### \|/

class SigRBFTensorizedRandProj(TimeseriesFeatureTransformer):
    def __init__(
            self,
            seed: PRNGKeyArray,
            n_features: int = 512,
            trunc_level: int = 3, #signature truncation level
            rbf_dimension: int = 512,
            sigma :float = 1.0,
            max_batch: int = 128,
            rff_max_batch: int = 2000,
            concat_levels: bool = True,
        ):
        """
        Transformer class for randomized RBF truncated signature 
        features via tensorized random projections.

        Args:
            seed (PRNGKeyArray): Random seed.
            n_features (int): Size of random projection.
            trunc_level (int): Signature truncation level.
            rbf_dimension (int): Dimension of Random Fourier Features (RFF) map.
            sigma (float): Sigma parameter of the RBF kernel.
            max_batch (int): Maximum batch size for computations.
            concat_levels (bool): Whether to concatenate the features 
                of each truncation level.
        """
        super().__init__(max_batch)
        self.n_features = n_features
        self.trunc_level = trunc_level
        self.rbf_dimension = rbf_dimension
        self.sigma = sigma
        self.seed = seed
        self.max_batch = max_batch
        self.rff_max_batch = rff_max_batch
        self.concat_levels = concat_levels

        trp_seed, rff_seed = jax.random.split(seed)
        self.linear_trp = SigVanillaTensorizedRandProj(
            trp_seed,
            n_features,
            trunc_level,
            max_batch,
            concat_levels,
        )
        self.rff = RandomFourierFeatures(
            rff_seed,
            rbf_dimension,
            sigma,
            rff_max_batch,
        )


    def fit(self, X: Float[Array, "N  T  D"]):
        """
        Initializes the Tensorized Random Projections of the for 
        the vanilla signature (corresponding to the linear kernel),
        and the Random Fourier Features (RFF) for the RBF kernel.

        Args:
            X (Float[Array, "N  T  D"]): Example batched time series data.
        """
        # bad, i dont like having to follow the sci-kit learn API.
        # Fix is to instead use equinox modules, but i would need to 
        # specify input dimensions at class init.
        self.rff.fit(X[0])
        test_rff = self.rff.transform(X[0:1, 0, :])
        self.linear_trp.fit(test_rff)
        return self


    def _batched_transform(
            self,
            X: Float[Array, "N  T  D"],
        ) -> Float[Array, "N  n_features"]:
        """
        Computes the RBF TRP features for the given batched input array.

        Args:
            X (Float[Array, "N  T  D"]): A single time series.
        
        Returns:
            Time series features of shape (N, n_features).
        """
        rff = self.rff.transform(X)
        return self.linear_trp.transform(rff)