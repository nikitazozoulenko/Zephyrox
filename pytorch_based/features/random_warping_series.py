from typing import List, Dict, Set, Any, Optional, Tuple, Literal, Callable
import numpy as np
import torch
from torch import Tensor

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from kernels.abstract_base import StaticKernel
from kernels.static_kernels import LinearKernel


############################################  |
######### Dynamic Time Warping #############  |
############################################ \|/


@torch.jit.script
def dtw_update_antidiag(
        M:Tensor,
        s:int,
        t:int,
    ):
    """
    Used in 'DP_dynamic_time_warping' to update the antidiagonals
    asynchroneously. Note that s,t>0"""
    M00 = M[..., s-1, t-1]
    M01 = M[..., s-1, t  ]
    M10 = M[..., s  , t-1]
    Mmin = torch.minimum(torch.minimum(M00, M01), M10)
    M[..., s, t] += Mmin


# dynamic time warping --- dynamic programming
@torch.jit.script
def DP_dynamic_time_warping(
        M: Tensor,
    ):
    """
    Dynamic programming for the computation of the DTW similarity.

    Args:
        M (Tensor): Tensor of shape (..., T, T2).
    
    Returns:
        Tensor: Tensor of shape (...) of DTW similarities.
    """
    T,  T2 = M.shape[-2:]

    M = M.clone()
    M[..., :, 0] = M[..., :, 0].cumsum(dim=-1)
    M[..., 0, :] = M[..., 0, :].cumsum(dim=-1)
    #iterate over antidiagonals
    for diag in range(2, T+T2-1):
        futures : List[torch.jit.Future[None]] = []
        for s in range(max(1, diag - T2 + 1), min(diag, T)):
            t = diag - s
            futures.append( torch.jit.fork(dtw_update_antidiag, M, s, t) )
        [torch.jit.wait(fut) for fut in futures]
    return M[..., -1, -1]


############################################  |
######### Random Warping Series ############  |
############################################ \|/


class RandomWarpingSeries():
    """
    Random Warping Series (RWS) algorithm 1 from 
    https://proceedings.mlr.press/v84/wu18b/wu18b.pdf.
    The RWS feature map is the Dynamic Time Warping (DTW) 
    distance to 'n_features' random series of random length D.
    """

    def __init__(
            self,
            n_features:int,
            D_min:int = 2,
            D_max:int = 50,
            sigma:float = 1.0,
            local_kernel:StaticKernel = LinearKernel(),
        ):
        """
        The RWS feature map is the Dynamic Time Warping (DTW) 
        distance to 'n_features' random series of random length D.

        Args:
            n_features (int): Number of random series to generate.
                This is the dimension of the feature space.
            D_min (int): Minimum length of random series.
            D_max (int): Maximum length of random series.
            sigma (float): Volatility of the Brownian Motions.
        """
        self.n_features = n_features
        self.D_min = D_min
        self.D_max = D_max
        self.sigma = sigma
        self.has_initialized = False
        self.local_kernel = local_kernel
    
    
    def _init_series(
            self, 
            X:Tensor,
        ):
        """
        Generates 'n_features' random series of length D, where D is 
        drawn uniformly at random from [D_min, D_max]. See Section 3.1
        https://proceedings.mlr.press/v84/wu18b/wu18b.pdf. We let our 
        paths be Brownian motions, as is hinted in the paper.

        The random series is a Tensor with shape (R, D, d). Time series 
        are constant after timestep D_r, where D_r is the length of the 
        r'th series.

        Args:
            X (Tensor): Example input tensor of shape (..., T, d) of 
                timeseries.
        """
        # Get shape, dtype and device info.
        d = X.shape[-1]
        device = X.device
        dtype = X.dtype
        
        # initialize random series
        D = torch.randint(self.D_min, self.D_max, (self.n_features,), 
                          device=device)
        series = torch.zeros((self.n_features, D.max(), d),
                             dtype=dtype, device=device)
        for r in range(self.n_features):
            series[r, :D[r], :] = torch.randn(D[r], d,
                            dtype=dtype, device=device) / self.sigma
        self.series = series.cumsum(dim=1) # shape (n_features, D, d)



    def __call__(
            self,
            X: Tensor,
        ):
        """
        Returns the Random Warping Series feature map of the input.
        O(N * T * n_features * D * d) time complexity,
        O(N * T * n_features * D)     space complexity.

        Args:
            X (Tensor): Input tensor of shape (N, T, d).
        
        Returns:
            Tensor: Tensor of shape (N, n_features) of RWS similarities.
        """
        # init
        if not self.has_initialized:
            self._init_series(X)
            self.has_initialized = True

        # calculate distance of pairs |X_s^i - S_t^j|^2
        dists = torch.sqrt(self.local_kernel.time_square_dist(X, self.series))
        return DP_dynamic_time_warping(dists) / np.sqrt(self.n_features)