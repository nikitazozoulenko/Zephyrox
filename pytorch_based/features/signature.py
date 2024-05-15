from typing import List, Dict, Set, Any, Optional, Tuple, Literal, Callable
import torch
from torch import Tensor
import signatory

from sklearn.base import TransformerMixin, BaseEstimator



def sig(
    X: Tensor,
    trunc_level: int,
):
    """
    Computes the truncated signature of time series of
    shape (T,d) with optional batch support.
    
    Args:
        X (Tensor): Tensor of shape (N, T, d) or (T, d) 
            of time series.
        trunc_level (int): Signature truncation level.
    
    Returns:
        Tensor: Tensor of shape (N, D) or (D) where 
            D = 1 + d + d^2 + ... + d^trunc_level.
    """
    if len(X.shape) == 2:
        X = X.unsqueeze(0)
    return signatory.signature(X, trunc_level).squeeze(0)



def logsig(
    X: Tensor,
    trunc_level: int,
):
    """
    Computes the truncated log-signature of time series of
    shape (T,d) with optional batch support.
    
    Args:
        X (Tensor): Tensor of shape (N, T, d) or (T, d) 
            of time series.
        trunc_level (int): Signature truncation level.
    
    Returns:
        Tensor: Tensor of shape (N, D) or (D) where D is 
            O(d^trunc_level) but smaller than the signature.
    """
    if len(X.shape) == 2:
        X = X.unsqueeze(0)
    return signatory.logsignature(X, trunc_level).squeeze(0)



class SigTransform(TransformerMixin, BaseEstimator):
    def __init__(
            self,
            trunc_level: int = 3, #signature truncation level
        ):
        """
        Summary

        Args:
            n_features (int): _description_. Defaults to 500.
            trunc_level (int): _description_. Defaults to 3.
        """
        self.trunc_level = trunc_level


    def fit(self, X: Tensor, y=None):
        return self


    def transform(
            self,
            X:Tensor,
        ):
        return sig(X, self.trunc_level)



class LogSigTransform(TransformerMixin, BaseEstimator):
    def __init__(
            self,
            trunc_level: int = 3, #signature truncation level
        ):
        """
        Summary

        Args:
            n_features (int): _description_. Defaults to 500.
            trunc_level (int): _description_. Defaults to 3.
        """
        self.trunc_level = trunc_level


    def fit(self, X: Tensor, y=None):
        return self


    def transform(
            self,
            X:Tensor,
        ):
        return logsig(X, self.trunc_level)
