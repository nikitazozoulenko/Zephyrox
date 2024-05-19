from typing import Tuple, List, Union, Any, Optional, Dict, Literal, Callable
import os
import sys
# sys.path.append(os.path.dirname(os.getcwd()))
# sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
from torch import Tensor

from ridge_loocv import fit_ridge_LOOCV


def apply_chunked(fn: Callable, X: Tensor, chunk_size: int=1000):
    """Applies a function to a tensor in chunks.

    Args:
        fn (Callable): Function to apply.
        X (Tensor): Input tensor of shape (N, ...).
        chunk_size (int): Chunk size. Defaults to 1000.

    Returns:
        Tensor: Output tensor.
    """
    batches = torch.split(X, chunk_size)
    output = [fn(batch) for batch in batches]
    return torch.cat(output, dim=0)



class RocketFeatures(nn.Module):
    def __init__(self, D, T, n_kernels, kernel_length=9, seed=0):
        super(RocketFeatures, self).__init__()

        max_exponent = np.floor(np.log2((T - 1) / (kernel_length- 1))).astype(np.int64)
        dilations = 2**np.arange(max_exponent + 1)
        n_kernels_per_dilation = n_kernels // dilations[-1] *2 #// 4


        self.n_kernels = n_kernels_per_dilation * len(dilations)
        self.convs = nn.ModuleList(
            [nn.Conv1d(
                in_channels=D, 
                out_channels=n_kernels_per_dilation, 
                kernel_size=kernel_length,
                dilation = dilation,
                padding = "same",
                bias=True) 
             for dilation in dilations]
        )

    
    def init_biases(self, X: Tensor, chunk_size: int=1000):
        """Initializes the biases of the convolutional layers,
        using the quantiles of the data. Assumes the data to
        be shuffled.

        WARNING: Slow even for 10 points, 10 000 kernels, T=113. (1.3s elapsed)

        Args:
            X (Tensor): Shape (N, D, T).
            chunk_size (int): Batch size for computations
        """
        with torch.no_grad():
            # first set the biases to zero
            for conv in self.convs:
                conv.bias.data.zero_()

            #obtain output
            out_per_conv = [apply_chunked(conv, X, chunk_size) for conv in self.convs]

            #initalize bias using random quantiles
            for out, conv in zip(out_per_conv, self.convs):
                #out: (N, n_kernels_per_dilation, T)
                n_ker_per_dil = out.shape[1]
                quantiles = 0.8 * torch.rand(n_ker_per_dil) + 0.1
                q = torch.quantile(out.permute(0,2,1).reshape(-1, n_ker_per_dil), quantiles, dim=0)
                conv.bias.data = torch.diag(q)

        return self

    
    def forward(self, x): # can be made more memory efficient, as we don't need to store all the intermediate results
        # x: (N, D, T)
        x = [conv(x) for conv in self.convs]
        x = torch.cat(x, dim=1)
        x = torch.mean((x>0), dim=-1, dtype=x.dtype)
        #TODO add multirocket features. Currently only have ROCKET features.
        return x



class Rocket(nn.Module):
    def __init__(self, D, T, n_kernels, n_out, kernel_length=9, seed=0):
        super(Rocket, self).__init__()
        self.rocket_features = RocketFeatures(D, T, n_kernels, kernel_length, seed)
        self.linear = nn.Linear(self.rocket_features.n_kernels, n_out)

    def init_biases(self, X: Tensor, chunk_size: int=1000):
        self.rocket_features.init_biases(X, chunk_size)
        return self
    
    def fit_ridge_LOOCV(self, X: Tensor, y: Tensor, alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000], chunk_size=1000):
        X = apply_chunked(self.rocket_features, X, chunk_size)
        optimal_beta, intercept, best_alpha= fit_ridge_LOOCV(X, y, alphas)
        self.linear.weight.data = optimal_beta
        self.linear.bias.data = intercept
        return self
    
    def forward(self, x):
        x = self.rocket_features(x)
        x = self.linear(x)
        return x