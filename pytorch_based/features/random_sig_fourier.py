from typing import List, Dict, Set, Any, Optional, Tuple, Literal, Callable
import torch
from torch import Tensor
from kernels.sig_trunc import cumsum_shift1

###################################################################  |
################# For the RBF-lifted signature ####################  |
################################################################### \|/

@torch.jit.script
def calc_P_RFF(
        X: Tensor,
        rff_weights_m : Tensor,
        P_m: Tensor,
        D: int,
    ):
    """
    Intermediate step in the calculation of the TRP-RFSF features.
    See Algo 3 in https://arxiv.org/pdf/2311.12214.pdf.

    Args:
        X (Tensor): Tensor of shape (..., T, d) of time series.
        rff_weights_m (Tensor): Tensor of shape (d, D) of RFF weights.
        P_m (Tensor): Shape (2D, D) with i.i.d. standard Gaussians.
        D (int): RFF dimension.
    """
    matmul = X @ rff_weights_m #shape (..., T, D)
    rff = torch.cat([torch.cos(matmul), 
                     torch.sin(matmul)], 
                     dim=-1) / D**0.5 #shape (..., T, 2D)
    U = rff.diff(dim=-2) @ P_m #shape (..., T-1, D)
    return U



@torch.jit.script
def tensorised_random_projection_features(
        X: Tensor,
        trunc_level: int,
        rff_weights: Tensor,
        P: Tensor,
    ):
    """
    Calculates the TRP-RFSF features for the given input tensor,
    when the underlying kernel is the RBF kernel. See Algo 3 in
    https://arxiv.org/pdf/2311.12214.pdf.

    Args:
        X (Tensor): Tensor of shape (..., T, d) of time series.
        trunc_level (int): Truncation level of the signature transform.
        rff_weights (Tensor): Tensor of shape (trunc_level, d, D) with
            independent RFF weights for each truncation level.
        P (Tensor): Shape (trunc_level, 2D, D) with i.i.d. standard 
            Gaussians.

    Returns:
        Tensor: Tensor of shape (trunc_level, ..., D) of TRP-RFSF features
            for each truncation level.
    """
    #first level
    D = P.shape[-1]
    V = calc_P_RFF(X, rff_weights[0], P[0], D) / D**0.5  #shape (..., T-1, D)
    levels = [V.sum(dim=-2)] #sum has shape (..., D)

    #subsequent levels
    for m in range(1, trunc_level):
        U = calc_P_RFF(X, rff_weights[m], P[m], D) #shape (..., T-1, D)
        V = cumsum_shift1(V, dim=-2) * U #shape (..., T-1, D)
        levels.append(V.sum(dim=-2)) # sum has shape (..., D)
    
    return torch.stack(levels, dim=0) #shape (trunc_level, ..., D)



class TRP_RFSF_Gaussian():
    def __init__(
            self,
            trunc_level: int, #signature truncation level
            n_features: int, #TRP dimension and RBF RFF dimension/2
            sigma: float, #RBF parameter
            only_last : bool = False, #whether to use only the last level
        ):
        self.trunc_level = trunc_level
        self.n_features = n_features
        self.sigma = sigma
        self.only_last = only_last
        self.has_initialized = False


    def _init_given_input(
            self, 
            X: Tensor
        ):
        """
        Initializes the random weights for the TRP-RFSF map for the 
        RBF kernel. This is 'trunc_level' independent RFF weights, 
        and (trunc_level, 2D, D) matrix of i.i.d. standard Gaussians 
        for the tensorized projection.

        Args:
            X (Tensor): Example input tensor of shape (..., T, d) of 
                timeseries.
        """
        # Get shape, dtype and device info.
        d = X.shape[-1]
        device = X.device
        dtype = X.dtype
        
        #initialize the RFF weights for each truncation level
        self.rff_weights = torch.randn(
                    self.trunc_level,
                    d,
                    self.n_features, 
                    device=device,
                    dtype=dtype
                    ) / self.sigma
        
        #initialize the tensorized projection matrix for each truncation level
        self.P = torch.randn(self.trunc_level,
                             2*self.n_features, 
                             self.n_features,
                             device=device,
                             dtype=dtype,)

            
    def __call__(
            self,
            X:Tensor,
        ):
        """
        Computes the TRP-RFSF features for the given input tensor,
        mapping time series from (T,d) to (n_features)

        Args:
            X (Tensor): Tensor of shape (..., T, d).
        
        Returns:
            Tensor: Tensor of shape (trunc_level, ..., n_features) or
                (..., n_features) if self.only_last=True.
        """
        if not self.has_initialized:
            self._init_given_input(X)
            self.has_initialized = True
        
        features = tensorised_random_projection_features(
            X, self.trunc_level, self.rff_weights, self.P
            )

        if self.only_last:
            return features[-1]
        else:
            return features
        

################################################################  |
################# For the vanilla signature ####################  |
################################################################ \|/
        
@torch.jit.script
def linear_tensorised_random_projection_features(
        X: Tensor,
        trunc_level: int,
        P: Tensor,
    ):
    """
    Calculates the TRP-RFSF features for the given input tensor,
    when the underlying kernel is the linear kernel. See Algo 3 in
    https://arxiv.org/pdf/2311.12214.pdf for details when the kernel
    is a translation invariant kernel (not applicable to the linear 
    kernel).

    Args:
        X (Tensor): Tensor of shape (..., T, d) of time series.
        trunc_level (int): Truncation level of the signature transform.
        P (Tensor): Shape (trunc_level, d, D) with i.i.d. standard 
            Gaussians.

    Returns:
        Tensor: Tensor of shape (trunc_level, ..., D) of TRP-RFSF features
            for each truncation level.
    """
    #first level
    D = P.shape[-1]
    V = X.diff(dim=-2) @ P[0] / D**0.5  #shape (..., T-1, D)
    levels = [V.sum(dim=-2)] #sum has shape (..., D)

    #subsequent levels
    for m in range(1, trunc_level):
        U = X.diff(dim=-2) @ P[m] #shape (..., T-1, D)
        V = cumsum_shift1(V, dim=-2) * U #shape (..., T-1, D)
        levels.append(V.sum(dim=-2)) # sum has shape (..., D)
    
    return torch.stack(levels, dim=0) #shape (trunc_level, ..., D)



class TRP_RFSF_Linear():
    def __init__(
            self,
            trunc_level: int, #signature truncation level
            n_features: int, #TRP dimension
            only_last : bool = False, #whether to use only the last level
        ):
        self.trunc_level = trunc_level
        self.n_features = n_features
        self.only_last = only_last
        self.has_initialized = False


    def _init_given_input(
            self, 
            X: Tensor
        ):
        """
        Initializes the Tensorized Random Projections of the 
        TRP-RFSF map for the vanilla signature (correpsonding 
        to the linear kernel). This is a (d, n_features) i.i.d.
        standard Gaussians random matrix for each truncation level.

        Args:
            X (Tensor): Example input tensor of shape (..., T, d) of 
                timeseries.
        """
        # Get shape, dtype and device info.
        d = X.shape[-1]
        device = X.device
        dtype = X.dtype
        
        #initialize the tensorized projection matrix for each truncation level
        self.P = torch.randn(self.trunc_level,
                             d, 
                             self.n_features,
                             device=device,
                             dtype=dtype,)


    def __call__(
            self,
            X:Tensor,
        ):
        """
        Computes the TRP-RFSF features for the given input tensor,
        mapping time series from (T,d) to (n_features)

        Args:
            X (Tensor): Tensor of shape (..., T, d).
        
        Returns:
            Tensor: Tensor of shape (trunc_level, ..., n_features) or
                (..., n_features) if self.only_last=True.
        """
        if not self.has_initialized:
            self._init_given_input(X)
            self.has_initialized = True
        
        features = linear_tensorised_random_projection_features(
            X, self.trunc_level, self.P
            )
        #TODO stack with 1
        if self.only_last:
            return features[-1]
        else:
            return features