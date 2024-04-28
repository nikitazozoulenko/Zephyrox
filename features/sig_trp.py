from typing import Tuple, List, Union, Any, Optional, Dict, Literal

import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
from jaxtyping import Array, Float, Int

from features.base import TimeseriesFeatureTransformer

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
            n_features: int = 512,
            trunc_level: int = 3, #signature truncation level
            max_batch: int = 128,
            seed: int = 0,
        ):
        """
        Transformer class for randomized vanilla truncated signature 
        features via tensorized random projections.

        Args:
            n_features (int): Size of random projection.
            trunc_level (int): Signature truncation level.
            max_batch (int): Maximum batch size for computations.
            seed (int): Random seed for random matrices.
        """
        super().__init__(max_batch)
        self.n_features = n_features
        self.trunc_level = trunc_level
        self.seed = seed


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
            jax.random.PRNGKey(self.seed), 
            (self.trunc_level, self.n_features, d),
            dtype=dtype
        )
        self.P /= np.sqrt(self.n_features)

        #vmap the transform
        self.vmapped_transform = jax.vmap(
            lambda x: linear_tensorised_random_projection_features(x, self.P),
            in_axes = 0,
            out_axes = 0
        )

        return self


    def _batched_transform(
            self,
            X: Float[Array, "N  T  D"],
        ) -> Float[Array, "N  trunc_level  n_features"]:
        """
        Computes the TRP features for the given batched input array.

        Args:
            X (Float[Array, "N  T  D"]): A single time series.
        
        Returns:
            Time series features of shape (N, trunc_level, n_features).
        """
        return self.vmapped_transform(X)
    
# ###################################################################  |
# ################# For the RBF-lifted signature ####################  |
# ################################################################### \|/

# @torch.jit.script
# def calc_P_RFF(
#         X: Tensor,
#         rff_weights_m : Tensor,
#         P_m: Tensor,
#         D: int,
#     ):
#     """
#     Intermediate step in the calculation of the TRP-RFSF features.
#     See Algo 3 in https://arxiv.org/pdf/2311.12214.pdf.

#     Args:
#         X (Tensor): Tensor of shape (..., T, d) of time series.
#         rff_weights_m (Tensor): Tensor of shape (d, D) of RFF weights.
#         P_m (Tensor): Shape (2D, D) with i.i.d. standard Gaussians.
#         D (int): RFF dimension.
#     """
#     matmul = X @ rff_weights_m #shape (..., T, D)
#     rff = torch.cat([torch.cos(matmul), 
#                      torch.sin(matmul)], 
#                      dim=-1) / D**0.5 #shape (..., T, 2D)
#     U = rff.diff(dim=-2) @ P_m #shape (..., T-1, D)
#     return U



# @torch.jit.script
# def rff_tensorised_random_projection_features(
#         X: Tensor,
#         trunc_level: int,
#         rff_weights: Tensor,
#         P: Tensor,
#     ):
#     """
#     Calculates the TRP-RFSF features for the given input tensor,
#     when the underlying kernel is the RBF kernel. See Algo 3 in
#     https://arxiv.org/pdf/2311.12214.pdf.

#     Args:
#         X (Tensor): Tensor of shape (..., T, d) of time series.
#         trunc_level (int): Truncation level of the signature transform.
#         rff_weights (Tensor): Tensor of shape (trunc_level, d, D) with
#             independent RFF weights for each truncation level.
#         P (Tensor): Shape (trunc_level, 2D, D) with i.i.d. standard 
#             Gaussians.

#     Returns:
#         Tensor: Tensor of shape (trunc_level, ..., D) of TRP-RFSF features
#             for each truncation level.
#     """
#     #first level
#     D = P.shape[-1]
#     V = calc_P_RFF(X, rff_weights[0], P[0], D) / D**0.5  #shape (..., T-1, D)

#     #subsequent levels
#     for m in range(1, trunc_level):
#         U = calc_P_RFF(X, rff_weights[m], P[m], D) #shape (..., T-1, D)
#         V = cumsum_shift1(V, dim=-2) * U           #shape (..., T-1, D)
    
#     return V.sum(dim=-2)



# class SigRBFTensorizedRandProj():
#     def __init__(
#             self,
#             trunc_level: int, #signature truncation level
#             n_features: int, #TRP dimension and RBF RFF dimension/2
#             sigma: float, #RBF parameter
#         ):
#         self.trunc_level = trunc_level
#         self.n_features = n_features
#         self.sigma = sigma


#     def fit(self, X: Tensor, y=None):
#         """
#          Initializes the random weights for the TRP-RFSF map for the 
#         RBF kernel. This is 'trunc_level' independent RFF weights, 
#         and (trunc_level, 2D, D) matrix of i.i.d. standard Gaussians 
#         for the tensorized projection.

#         Args:
#             X (Tensor): Example input tensor of shape (N, T, d).
#         """
#         # Get shape, dtype and device info.
#         d = X.shape[-1]
#         device = X.device
#         dtype = X.dtype
        
#         #initialize the RFF weights for each truncation level
#         self.rff_weights = torch.randn(
#                     self.trunc_level,
#                     d,
#                     self.n_features, 
#                     device=device,
#                     dtype=dtype
#                     ) / self.sigma
        
#         #initialize the tensorized projection matrix for each truncation level
#         self.P = torch.randn(self.trunc_level,
#                              2*self.n_features, 
#                              self.n_features,
#                              device=device,
#                              dtype=dtype,)
#         return self

            
#     def transform(
#             self,
#             X:Tensor,
#         ):
#         """
#         Computes the RBF TRP-RFSF features for the given input tensor,
#         mapping time series from (T,d) to (n_features).

#         Args:
#             X (Tensor): Tensor of shape (N, T, d).
        
#         Returns:
#             Tensor: Tensor of shape (N, n_features).
#         """
#         features = rff_tensorised_random_projection_features(
#             X, self.trunc_level, self.rff_weights, self.P
#             )
#         return features
        
