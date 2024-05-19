from typing import Tuple, List, Union, Any, Optional, Dict, Literal, Callable

import torch
from torch import Tensor


def _ridge_LOOCV_d_leq_n(
        X: Tensor,
        y: Tensor,
        alphas: List[float]
    ) -> Tuple[Tensor, float]:
    """LOOCV for Ridge regression when D <= N.

    Args:
        X (Tensor): Shape (N, D).
        y (Tensor): Shape (N,).
        alphas (List[float]): Alphas to test.
    
    Returns:
        Tuple[Tensor, float]: Coefficients beta and intercept.
    """
    # Use SVD for decomposition
    U, S, Vt = torch.linalg.svd(X, full_matrices=False)
    Ut_y = U.T @ y
    U_squared = U ** 2

    # Efficient LOOCV
    best_alpha = -1
    best_error = torch.inf
    for alpha in alphas:
        d = S / (S**2 + alpha)
        beta_ridge = Vt.T @ (d * Ut_y)
        y_pred = X @ beta_ridge
        residuals = y - y_pred
        H_diag = torch.sum(U_squared * (S**2 / (S**2 + alpha)), dim=1)
        error = ((residuals / (1 - H_diag)) ** 2).mean()

        if error < best_error:
            best_error = error
            best_alpha = alpha

    # Compute optimal beta and intercept
    d_optimal = S / (S**2 + best_alpha)
    optimal_beta = Vt.T @ (d_optimal * Ut_y)
    intercept = y.mean() - torch.dot(X.mean(dim=0), optimal_beta)
    return optimal_beta, intercept, best_alpha



def _ridge_LOOCV_n_leq_d(
        X: Tensor,
        y: Tensor,
        alphas: List[float]
    ) -> Tuple[Tensor, float]:
    """LOOCV for Ridge regression when N < D.

    Args:
        X (Tensor): Shape (N, D).
        y (Tensor): Shape (N,).
        alphas (List[float]): Alphas to test.
    
    Returns:
        Tuple[Tensor, float]: Coefficients beta and intercept.
    """
    # Use eigendecomposition of Gram matrix
    K = X @ X.T
    eigvals, Q = torch.linalg.eigh(K)
    QT_y = Q.T @ y

    # Efficient LOOCV
    best_alpha = -1
    best_error = torch.inf
    for alpha in alphas:
        w = 1.0 / (eigvals + alpha)
        c = Q @ (w * QT_y)
        y_pred = K @ c
        residuals = y - y_pred
        H_diag = torch.sum( (Q ** 2) * (eigvals / (eigvals + alpha)), dim=1)
        error = ((residuals / (1 - H_diag)) ** 2).mean()

        if error < best_error:
            best_error = error
            best_alpha = alpha

    # Compute optimal beta and intercept
    w_opt = 1.0 / (eigvals + best_alpha)
    c_opt = Q @ (w_opt * QT_y)
    optimal_beta = X.T @ c_opt
    intercept = y.mean() - torch.dot(X.mean(dim=0), optimal_beta)
    return optimal_beta, intercept, best_alpha



def fit_ridge_LOOCV(
        X: Tensor, 
        y: Tensor, 
        alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    ) -> Tuple[Tensor, float]:
    """Find the optimal Ridge fit using efficient Leave-One-Out 
    Cross-Validation with SVD and eigendecomposition.

    Args:
        X (Tensor): Input data of shape (N, D).
        y (Tensor): Target data of shape (N,).
        alphas (List): List of alphas to test.
    
    Returns:
        Tuple[Tensor, float]: Coefficients beta and intercept.
    """
    N, D = X.shape
    if N > D:
        return _ridge_LOOCV_d_leq_n(X, y, alphas)
    else:
        return _ridge_LOOCV_n_leq_d(X, y, alphas)