from typing import List, Dict, Set, Any, Optional, Tuple, Literal, Callable
import numpy as np
import torch
from torch import Tensor

class RBF_RandomFourierFeatures:
    def __init__(
            self, 
            sigma : float = 1.0,
            n_features : int = 1000,
            seed : Optional[int] = None,
            method : Literal["cos(x)sin(x)", "cos(x + b)"] = "cos(x)sin(x)"
        ):
        """
        Random Fourier Features (RFF) for the RBF kernel. The RBF kernel is defined 
        on R^d via k(x, y) = exp(-||x-y||^2 / (2 * sigma^2)), and the RFF map is 
        given by either z(x) = [cos(w_1^T x + b_1), ... cos(w_D^T x + b_D)], or
        z(x) =  [cos(w_1^T x), ... cos(w_D^T x), sin(w_1^T x), ..., sin(w_D^T x)],
        where w_i are drawn iid from a N(0, 1/sigma^2) vector, and b_i are drawn
        iid uniformly from (0, 2*pi).

        Args:
            sigma (float, optional): Bandwidth of the RBF kernel. Defaults to 1.0.
            n_features (int, optional): Number of features. Defaults to 1000.
            seed (int, optional): Seed for random matrix initialization.
            method (Literal["cos(x)sin(x)", "cos(x + b)"], optional): Method for 
                generating the RFF map. Defaults to "cos(x)sin(x)".
        """
        self.n_features = n_features
        self.seed = seed
        self.sigma = sigma
        self.method = method
        self.has_initialized = False
    

    def _init_given_input(
            self, 
            X: Tensor
        ):
        """
        Initializes the random weights and biases for the RFF map. 
        The weights are drawn from a N(0, 1/sigma^2) distribution.

        Args:
            X (Tensor): Input tensor of shape (..., d)
        """
        d = X.shape[-1]
        device = X.device
        dtype = X.dtype

        gen = torch.Generator(device=device)
        if self.seed is not None:
            gen.manual_seed(self.seed)
        else:
            gen.seed()
        self.weights = torch.randn(d,
                                   self.n_features, 
                                   generator=gen,
                                   device=device,
                                   dtype=dtype
                                   ) / self.sigma
        if self.method == "cos(x + b)":
            self.biases = 2 * np.pi * torch.rand(self.n_features, 
                                                device=device, 
                                                dtype=dtype)


    def __call__(
            self, 
            X: Tensor
        ) -> Tensor:
        """
        Maps the input bbR^d to the feature space R^(2*n_features) 
        or R^(n_features), depending on self.method. The random weights 
        will be initialized the first time __call__ is called.

        Args:
            X (Tensor): Input tensor of shape (..., d)

        Returns:
            Tensor: Transformed tensor of shape (..., D)
        """
        if not self.has_initialized:
            self._init_given_input(X)
            self.has_initialized = True
        # X: (..., d)
        # weights: (d, D)
        # biases: (D,)
        # X @ weights: (..., D)
        matmul = X @ self.weights
        if self.method == "cos(x + b)":
            return torch.cos(matmul + self.biases) / np.sqrt(self.n_features/2)
        else:
            return torch.cat([torch.cos(matmul), 
                              torch.sin(matmul)], dim=-1
                              ) / np.sqrt(self.n_features)