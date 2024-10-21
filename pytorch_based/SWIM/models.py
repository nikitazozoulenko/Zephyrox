from typing import Tuple, List, Union, Any, Optional, Dict, Literal, Callable, Type
import abc

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from sklearn.linear_model import RidgeCV, RidgeClassifierCV


############################################################################
##### Base classes                                                     #####
##### - FittableModule: A nn.Module with .fit(X, y) support            #####
##### - Sequential: chaining together multiple FittableModules         #####
##### - make_fittable: turns type nn.Module into FittableModule        #####
############################################################################


class FittableModule(nn.Module):
    def __init__(self):
        super(FittableModule, self).__init__()
    

    @abc.abstractmethod
    def fit(self, 
            X: Optional[Tensor] = None, 
            y: Optional[Tensor] = None,
        ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Given neurons of the previous layer, and target labels, fit the 
        module. Returns the forwarded activations and labels [f(X), y].

        Args:
            X (Optional[Tensor]): Forward-propagated activations of training data, shape (N, d).
            y (Optional[Tensor]): Training labels, shape (N, p).
        
        Returns:
            Forwarded activations and labels [f(X), y].
        """
        raise NotImplementedError("Method fit must be implemented in subclass.")
        #return self(X), y


class Sequential(FittableModule):
    def __init__(self, *layers: FittableModule):
        """
        Args:
            *layers (FittableModule): Variable length argument list of FittableModules to chain together.
        """
        super(Sequential, self).__init__()
        self.layers = nn.ModuleList(layers)


    def fit(self, X: Tensor, y: Tensor):
        for layer in self.layers:
            X, y = layer.fit(X, y)
        return X, y


    def forward(self, X: Tensor) -> Tensor:
        for layer in self.layers:
            X = layer(X)
        return X
    


def make_fittable(module_class: Type[nn.Module]) -> Type[FittableModule]:
    class FittableModuleWrapper(FittableModule, module_class):
        def __init__(self, *args, **kwargs):
            FittableModule.__init__(self)
            module_class.__init__(self, *args, **kwargs)
        
        def fit(self, 
                X: Optional[Tensor] = None, 
                y: Optional[Tensor] = None,
            ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
            return self(X), y
    
    return FittableModuleWrapper


Tanh = make_fittable(nn.Tanh)
ReLU = make_fittable(nn.ReLU)
Identity = make_fittable(nn.Identity)


############################################################################
##### Layers                                                           #####
##### - Dense: Fully connected layer                                  #####
##### - SWIMLayer                                                     #####
##### - RidgeCV (TODO currently just an sklearn wrapper)
############################################################################

class Dense(FittableModule):
    def __init__(self,
                 generator: torch.Generator,
                 in_dim: int,
                 out_dim: int,
                 activation: Optional[nn.Module] = None,
                 ):
        """Dense MLP layer with LeCun weight initialization,
        Gaussan bias initialization."""
        super(Dense, self).__init__()
        self.generator = generator
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dense = nn.Linear(in_dim, out_dim)
        self.activation = activation
    
    def fit(self, X:Tensor, y:Tensor):
        with torch.no_grad():
            nn.init.normal_(self.dense.weight, mean=0, std=self.in_dim**-0.5, generator=self.generator)
            nn.init.normal_(self.dense.bias, mean=0, std=self.in_dim**-0.25, generator=self.generator)
            return self(X), y
    
    def forward(self, X):
        X = self.dense(X)
        if self.activation is not None:
            X = self.activation(X)
        return X
    


class SWIMLayer(FittableModule):
    def __init__(self,
                 generator: torch.Generator,
                 in_dim: int, 
                 out_dim: int,
                 activation: Optional[nn.Module] = None,
                 sampling_method: Literal['uniform', 'gradient'] = 'gradient'
                 ):
        """Dense MLP layer with pair sampled weights (uniform or gradient-weighted).

        Args:
            generator (torch.Generator): PRNG object.
            in_dim (int): Input dimension.
            out_dim (int): Output dimension.
            activation (nn.Module): Activation function.
            sampling_method (str): Pair sampling method. Uniform or gradient-weighted.
        """
        super(SWIMLayer, self).__init__()
        self.generator = generator
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dense = nn.Linear(in_dim, out_dim)
        self.sampling_method = sampling_method
        self.activation = activation


    def fit(self, 
            X: Tensor, 
            y: Tensor,
        ) -> Tuple[Tensor, Tensor]:
        """Given forward-propagated training data X at the previous 
        hidden layer, and supervised target labels y, fit the weights
        iteratively by letting rows of the weight matrix be given by
        pairs of samples from X. See paper for more details.

        Args:
            X (Tensor): Forward-propagated activations of training data, shape (N, d).
            y (Tensor): Training labels, shape (N, p).
        
        Returns:
            Forwarded activations and labels [f(X), y].
        """
        with torch.no_grad():
            N, d = X.shape
            dtype = X.dtype
            device = X.device
            EPS = torch.tensor(0.1, dtype=dtype, device=device)

            #obtain pair indices
            n = 5*N
            idx1 = torch.arange(0, n, dtype=torch.int32, device=device) % N
            delta = torch.randint(1, N, size=(n,), dtype=torch.int32, device=device, generator=self.generator)
            idx2 = (idx1 + delta) % N
            dx = X[idx2] - X[idx1]
            dists = torch.linalg.norm(dx, axis=1, keepdims=True)
            dists = torch.maximum(dists, EPS)
            
            if self.sampling_method=="gradient":
                #calculate 'gradients'
                dy = y[idx2] - y[idx1]
                y_norm = torch.linalg.norm(dy, axis=1, keepdims=True) #NOTE 2023 paper uses ord=inf instead of ord=2
                grad = (y_norm / dists).reshape(-1) 
                p = grad/grad.sum()
            elif self.sampling_method=="uniform":
                p = torch.ones(n, dtype=dtype, device=device) / n
            else:
                raise ValueError(f"sampling_method must be 'uniform' or 'gradient'. Given: {self.sampling_method}")

            #sample pairs
            selected_idx = torch.multinomial(
                p,
                self.out_dim,
                replacement=True,
                generator=self.generator
                )
            idx1 = idx1[selected_idx]
            dx = dx[selected_idx]
            dists = dists[selected_idx]

            #define weights and biases
            weights = dx / (dists**2)
            biases = -torch.einsum('ij,ij->i', weights, X[idx1]) - 0.5
            self.dense.weight.data = weights
            self.dense.bias.data = biases
            return self(X), y
    

    def forward(self, X):
        X = self.dense(X)
        if self.activation is not None:
            X = self.activation(X)
        return X



class RidgeCVModule(FittableModule):
    def __init__(self, alphas=np.logspace(-1, 3, 10)):
        """RidgeCV layer using sklearn's RidgeCV. TODO dont use sklearn"""
        super(RidgeCVModule, self).__init__()
        self.ridge = RidgeCV(alphas=alphas)

    def fit(self, X: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Fit the RidgeCV model. TODO dont use sklearn"""
        X_np = X.detach().cpu().numpy().astype(np.float64)
        y_np = y.detach().cpu().squeeze().numpy().astype(np.float64)
        self.ridge.fit(X_np, y_np)
        return self(X), y

    def forward(self, X: Tensor) -> Tensor:
        """Forward pass through the RidgeCV model. TODO dont use sklearn"""
        X_np = X.detach().cpu().numpy().astype(np.float64)
        y_pred_np = self.ridge.predict(X_np)
        return torch.tensor(y_pred_np, dtype=X.dtype, device=X.device).unsqueeze(1) #TODO unsqueeze???


class RidgeClassifierCVModule(FittableModule):
    def __init__(self, alphas=np.logspace(-1, 3, 10)):
        """RidgeClassifierCV layer using sklearn's RidgeClassifierCV. TODO dont use sklearn"""
        super(RidgeClassifierCVModule, self).__init__()
        self.ridge = RidgeClassifierCV(alphas=alphas)

    def fit(self, X: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Fit the sklearn ridge model."""
        # Make y categorical from one_hot NOTE assumees y one-hot
        y_cat = torch.argmax(y, dim=1)
        X_np = X.detach().cpu().numpy().astype(np.float64)
        y_np = y_cat.detach().cpu().squeeze().numpy().astype(np.float64)
        self.ridge.fit(X_np, y_np)
        return self(X), y

    def forward(self, X: Tensor) -> Tensor:
        X_np = X.detach().cpu().numpy().astype(np.float64)
        y_pred_np = self.ridge.predict(X_np)
        return torch.tensor(y_pred_np, dtype=X.dtype, device=X.device)


######################################
#####       Residual Block       #####
######################################


def create_layer(generator: torch.Generator,
                layer_name:str, 
                in_dim:int, 
                out_dim:int,
                activation: Optional[nn.Module],
                sampling_method: str,
                ):
    if layer_name == "dense":
        return Dense(generator, in_dim, out_dim, activation)
    elif layer_name == "SWIM":
        return SWIMLayer(generator, in_dim, out_dim, activation, sampling_method)
    elif layer_name == "identity":
        return Identity()
    else:
        raise ValueError(f"layer_name must be one of ['dense', 'SWIM', 'identity']. Given: {layer_name}")


class ResidualBlock(FittableModule):
    def __init__(self, 
                 generator: torch.Generator,
                 in_dim: int,
                 bottleneck_dim: int,
                 layer1: str,
                 layer2: str,
                 activation: nn.Module = nn.Tanh(),
                 residual_scale: float = 1.0,
                 sampling_method: Literal['uniform', 'gradient'] = 'gradient',
                 ):
        """Residual block with 2 layers and a skip connection.
        
        Args:
            generator (torch.Generator): PRNG object.
            in_dim (int): Input dimension.
            bottleneck_dim (int): Dimension of the bottleneck layer.
            layer1 (str): First layer in the block. One of ["dense", "swim", "identity"].
            layer2 (str): See layer1.
            activation (nn.Module): Activation function.
            residual_scale (float): Scale of the residual connection.
            sampling_method (str): Pair sampling method for SWIM. One of ['uniform', 'gradient'].
        """
        super(ResidualBlock, self).__init__()
        self.residual_scale = residual_scale
        self.first = create_layer(generator, layer1, in_dim, bottleneck_dim, None, sampling_method)
        self.activation = activation
        self.second = create_layer(generator, layer2, bottleneck_dim, in_dim, None, sampling_method)


    def fit(self, X: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            X0 = X
            X, y = self.first.fit(X,y)
            X = self.activation(X)
            X, y = self.second.fit(X,y)
        return X0 + X * self.residual_scale, y


    def forward(self, X: Tensor) -> Tensor:
        X0 = X
        X = self.first(X)
        X = self.activation(X)
        X = self.second(X)
        return X0 + X * self.residual_scale
    

#####################################
##### Residual Networks         #####
##### - ResNet                  #####
##### - NeuralEulerODE          #####
#####################################

class ResNet(Sequential):
    def __init__(self, 
                 generator: torch.Generator,
                 in_dim: int,
                 hidden_size: int,
                 bottleneck_dim: int,
                 n_blocks: int,
                 upsample_layer: Literal['dense', 'SWIM', 'identity'] = 'SWIM',
                 upsample_activation: nn.Module = nn.Tanh(),
                 res_layer1: str = "SWIM",
                 res_layer2: str = "dense",
                 res_activation: nn.Module = nn.Tanh(),
                 residual_scale: float = 1.0,
                 sampling_method: Literal['uniform', 'gradient'] = 'gradient',
                 output_layer: Literal['ridge', 'dense', 'identity'] = 'ridge',
                 ):
        """Residual network with multiple residual blocks.
        
        Args:
            generator (torch.Generator): PRNG object.
            in_dim (int): Input dimension.
            hidden_size (int): Dimension of the hidden layers.
            bottleneck_dim (int): Dimension of the bottleneck layer.
            n_blocks (int): Number of residual blocks.
            upsample_layer (str): Layer before any residual connections. One of ['dense', 'SWIM', 'identity'].
            upsample_activation (nn.Module): Activation function for the upsample layer.
            res_layer1 (str): First layer in the block. One of ["dense", "swim", "identity"].
            res_layer2 (str): See layer1.
            res_activation (nn.Module): Activation function for the residual blocks.
            residual_scale (float): Scale of the residual connection.
            sampling_method (str): Pair sampling method for SWIM. One of ['uniform', 'gradient'].
            output_layer (str): Output layer. One of ['ridge', 'ridge classifier', 'dense', 'identity'].
        """
        upsample = create_layer(generator, 
                                     upsample_layer, 
                                     in_dim, 
                                     hidden_size, 
                                     upsample_activation, 
                                     sampling_method)
        residual_blocks = [
            ResidualBlock(generator, 
                          hidden_size, 
                          bottleneck_dim, 
                          res_layer1, 
                          res_layer2, 
                          res_activation, 
                          residual_scale, 
                          sampling_method)
            for _ in range(n_blocks)
        ]
        if output_layer == 'dense':
            out = Dense(generator, hidden_size, 1, None)
        elif output_layer == 'ridge':
            out = RidgeCVModule()
        elif output_layer == 'ridge classifier':
            out = RidgeClassifierCVModule()
        elif output_layer == 'identity':
            out = Identity()
        else:
            raise ValueError(f"output_layer must be one of ['ridge', 'ridge classifier', 'dense', 'identity']. Given: {output_layer}")
        
        super(ResNet, self).__init__(upsample, *residual_blocks, out)



class NeuralEulerODE(ResNet):
    def __init__(self, 
                 generator: torch.Generator,
                 in_dim: int,
                 hidden_size: int,
                 n_layers: int,
                 upsample_layer: Literal['dense', 'SWIM', 'identity'] = 'SWIM',
                 upsample_activation: nn.Module = nn.Tanh(),
                 res_layer: str = "SWIM",
                 res_activation: nn.Module = nn.Tanh(),
                 residual_scale: float = 1.0,
                 sampling_method: Literal['uniform', 'gradient'] = 'gradient',
                 output_layer: Literal['ridge', 'dense'] = 'dense',
                 ):
        """Euler discretization of Neural ODE."""
        super(NeuralEulerODE, self).__init__(generator, in_dim, hidden_size, None,
                                             n_layers, upsample_layer, upsample_activation,
                                             res_layer, "identity", res_activation,
                                             residual_scale, sampling_method, output_layer)