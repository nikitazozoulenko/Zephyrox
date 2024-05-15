from typing import List, Dict, Set, Any, Optional, Tuple, Literal, Callable
import torch
from torch import Tensor
from torch.nn.functional import relu
from torch.nn.functional import tanh
import numpy as np

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from kernels.abstract_base import TimeSeriesKernel, StaticKernel
from kernels.static_kernels import RBFKernel


@torch.jit.script
def randomized_sig_tanh(
        X:Tensor,
        A:Tensor,
        b:Tensor,
        Y_0:Tensor,
    ):
    """
    Randomized signature of a (batched) time series X, with tanh
    activation function.

    Args:
        X (Tensor): Input tensor of shape (N, T, d).
        A (Tensor): Tensor of shape (M, M, d). Random matrix.
        b (Tensor): Tensor of shape (M, d). Random bias.
        Y_0 (Tensor): Initial value of the randomized signature.
            Tensor of shape (M).
    """
    N, T, d = X.shape
    diff = X.diff(dim=1) # shape (N, T-1, d)
    Y_0 = torch.tile(Y_0, (N, 1)) # shape (N, M)

    #iterate y[t+1] = y[t] + ...
    Z = torch.tensordot(tanh(Y_0), A, dims=1) + b[None] # shape (N, M, d)
    Y = Y_0 + (Z * diff[:, 0:1, :]).sum(dim=-1) # shape (N, M)
    for t in range(1, T-1):
        Z = torch.tensordot(tanh(Y), A, dims=1) + b[None]
        Y = Y + (Z * diff[:, t:t+1, :]).sum(dim=-1)
    return Y


@torch.jit.script
def randomized_sig_linear(
        X:Tensor,
        A:Tensor,
        b:Tensor,
        Y_0:Tensor,
    ):
    """
    Randomized signature of a (batched) time series X

    Args:
        X (Tensor): Input tensor of shape (N, T, d).
        A (Tensor): Tensor of shape (M, M, d). Random matrix.
        b (Tensor): Tensor of shape (M, d). Random bias.
        Y_0 (Tensor): Initial value of the randomized signature.
            Tensor of shape (M).
    """
    N, T, d = X.shape
    diff = X.diff(dim=1) # shape (N, T-1, d)
    Y_0 = torch.tile(Y_0, (N, 1)) # shape (N, M)

    #iterate y[t+1] = y[t] + ...
    Z = torch.tensordot(Y_0, A, dims=1) + b[None] # shape (N, M, d)
    Y = Y_0 + (Z * diff[:, 0:1, :]).sum(dim=-1) # shape (N, M)
    for t in range(1, T-1):
        Z = torch.tensordot(Y, A, dims=1) + b[None]
        Y = Y + (Z * diff[:, t:t+1, :]).sum(dim=-1)
    return Y



class RandomizedSignature():
    def __init__(
            self,
            n_features: int, #TRP dimension and RBF RFF dimension/2
            activation:Literal["tanh", "linear"] = "linear",
            seed:Optional[int] = None,
        ):
        self.n_features = n_features
        self.activation = activation
        self.seed = seed


    def fit(self, X: Tensor, y=None):
        """
        Initializes the random matrices and biases used in the 
        randomized signature kernel.

        Args:
            X (Tensor): Example input tensor of shape (N, T, d) of 
                timeseries.
        """
        # Get shape, dtype and device info.
        N, T, d = X.shape
        device = X.device
        dtype = X.dtype
        
        # Create a generator and set the seed
        gen = torch.Generator(device=device)
        if self.seed:
            gen.manual_seed(self.seed)
        
        # Initialize the random matrices and biases
        self.A = torch.randn(self.n_features, 
                             self.n_features, 
                             d, 
                             device=device,
                             dtype=dtype,
                             generator=gen
                             ) / np.sqrt(self.n_features * d * T)
        self.b = torch.randn(self.n_features,
                             d,
                             device=device,
                             dtype=dtype,
                             generator=gen
                             )

        self.Y_0 = torch.randn(self.n_features,
                               device=device,
                               dtype=dtype,
                               generator=gen)
        return self

            
    def transform(
            self,
            X:Tensor,
        ):
        """
        Computes the RBF TRP-RFSF features for the given input tensor,
        mapping time series from (T,d) to (n_features).

        Args:
            X (Tensor): Tensor of shape (N, T, d).
        
        Returns:
            Tensor: Tensor of shape (N, n_features).
        """
        if self.activation == "tanh":
            features = randomized_sig_tanh(X, self.A, self.b, self.Y_0)
        else:
            features = randomized_sig_linear(X, self.A, self.b, self.Y_0)
        return features