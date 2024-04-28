import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
from jaxtyping import Array, Float, Int


def normalize_mean_std(
        X: Float[Array, "N ..."],
        epsilon: float = 0.00001,
    ):
    """Normalize 'X' across axis=0 using mean and std.

    Args:
        X (Float[Array, "N ..."]): Data to normalize.
        epsilon (float): Small value to avoid division by zero.

    Returns:
        'X' normalized by the mean and std.
    """
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    return (X - mean) / (std+epsilon)



def normalize_mean_std_traindata(
        train: Float[Array, "N ..."],
        test: Float[Array, "N ..."],
        epsilon: float = 0.00001,
    ):
    """Normalize 'train' and 'test' across axis=0 using mean and std
    of 'train' only.

    Args:
        train (Float[Array, "N ..."]): Tabular train set.
        test (Float[Array, "N ..."]): Tabular test set.
        epsilon (float): Small value to avoid division by zero.
    
    Returns:
        'train' and 'test' normalized by the mean and std of 'train'.
    """
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    train = (train - mean) / (std+epsilon)
    test = (test - mean) / (std+epsilon)
    return train, test



def avg_pool_time(
        X: Float[Array, "... T D"],
        max_T: int,
    ):
    """
    Reduces the time dimension of X by performing average pooling.
    
    Args:
        X (Float[Array, "... T D"]): Input array of time series.
        max_T (int): Maximum time length.
    
    Returns: 
        Pooled array of shape (..., new_T, D), where new_T may be less than max_T.
    """

    # reshape to 3D
    original_shape = X.shape
    X = X.reshape(-1, X.shape[-2], X.shape[-1])
    N, T, D = X.shape
    pool_size = 1 + (T-1) // max_T

    # pool time dimension
    new_T = T // pool_size
    X_grouped = X[:, :new_T*pool_size, :].reshape(N, new_T, pool_size, D)
    pooled = X_grouped.mean(axis=2)
    if new_T*pool_size < T:
        rest = X[:, new_T*pool_size:, :].mean(axis=1, keepdims=True)
        pooled = jnp.concatenate([pooled, rest], axis=1)

    # reshape back to original shape
    return pooled.reshape(original_shape[:-2] + (-1, D))



def augment_time(
        X: Float[Array, "... T D"],
        min_val: float = 0.0,
        max_val: float = 1.0,
    ):
    """
    Add time channel to 'X' with values uniformly between 
    'min_val' and 'max_val'.

    Args:
        X (Float[Array, "... T D"]): Input array of time series.
        min_val (float): Minimum value of time.
        max_val (float): Maximum value of time.
    
    Returns: 
        Array of shape (..., T, D+1).
    """
    # reshape to 3D
    original_shape = X.shape
    X = X.reshape(-1, X.shape[-2], X.shape[-1])
    N, T, d = X.shape
    dtype = X.dtype

    # concat time dimension. NOTE torch.repeat works like np.tile
    time = jnp.linspace(min_val, max_val, T, dtype=dtype)
    time = jnp.tile(time, (N, 1))[:,:, None] #shape (N, T, 1)
    X = jnp.concatenate([X, time], axis=-1)

    # reshape back to original shape
    return X.reshape(original_shape[:-1] + (d+1,))



def add_basepoint_zero(
        X: Float[Array, "... T D"],
        first: bool = True,
    ):
    """
    Add basepoint zero to 'X' in the time dimension.
    
    Args:
        X (Float[Array, "... T D"]): Input array of time series.
        first (bool): If True, add basepoint at the beginning of time.
                      If False, add basepoint at the end of time.
        
    Returns: 
        Tensor with shape (..., T+1, d).
    """
    # reshape to 3D
    original_shape = X.shape
    X = X.reshape(-1, X.shape[-2], X.shape[-1])
    N, T, d = X.shape
    dtype = X.dtype

    # add basepoint
    basepoint = jnp.zeros((N, 1, d), dtype=dtype)
    v = [basepoint, X] if first else [X, basepoint]
    X = jnp.concatenate(v, axis=1)

    # reshape back to original shape
    return X.reshape(original_shape[:-2] + (T+1, d))



def I_visibility(X: Float[Array, "... T D"]):
    """
    Performs the I-visiblity transform on 'X', see page 5 of
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9412642

    Args:
        X (Float[Array, "... T D"]): Input array of time series.
        
    Returns: 
        Tensor with shape (..., T+2, d+1).
    """
    # reshape to 3D
    original_shape = X.shape
    X = X.reshape(-1, X.shape[-2], X.shape[-1])
    N, T, d = X.shape
    dtype = X.dtype

    # (vec(0), 0) (x_1, 0) then (x_1, 1) (x_2, 1) ...
    X = add_basepoint_zero(X, first=True) # start of time
    start = jnp.concatenate([X[:, 0:2, :], 
                               jnp.zeros((N, 2, 1), dtype=dtype)], 
                            axis=-1)
    rest = jnp.concatenate([X[:, 1:, :], 
                              jnp.ones((N, T, 1), dtype=dtype)], 
                            axis=-1)
    X = jnp.concatenate([start, rest], axis=1)

    # reshape back to original shape
    return X.reshape(original_shape[:-2] + (T+2, d+1))



def T_visibility(X: Float[Array, "... T D"]):
    """
    Performs the T-visiblity transform on 'X', see page 5 of
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9412642

    Args:
        X (Float[Array, "... T D"]): Input array of time series.
        
    Returns: 
        Tensor with shape (..., T+2, d+1).
    """
    # reshape to 3D
    original_shape = X.shape
    X = X.reshape(-1, X.shape[-2], X.shape[-1])
    N, T, d = X.shape
    dtype = X.dtype

    # (x_1, 1) (x_2, 1) ... then (x_T, 0) (vec(0), 0)
    X = add_basepoint_zero(X, first=False) # end of time
    rest = jnp.concatenate([X[:, :-1, :], 
                              jnp.ones((N, T, 1), dtype=dtype)], 
                            axis=-1)
    end = jnp.concatenate([X[:, -2:, :], 
                             jnp.zeros((N, 2, 1), dtype=dtype)],
                             axis=-1)
    X = jnp.concatenate([rest, end], axis=1)

    # reshape back to original shape
    return X.reshape(original_shape[:-2] + (T+2, d+1))



# def normalize_streams(train:Tensor, 
#                       test:Tensor,
#                       max_T:int = 100,
#                       ):
#     """Inputs are 3D arrays of shape (N, T, d) where N is the number of time series, 
#     T is the length of each time series, and d is the dimension of each time series.
#     Performs average pooling to reduce the length of the time series to at most max_T,
#     z-score normalization, basepoint addition, and time augmentation.
#     """
#     # Make time series length smaller
#     _, T, d = train.shape
#     if T > max_T:
#         pool_size = 1 + (T-1) // max_T
#         train = avg_pool_time(train, pool_size)
#         test = avg_pool_time(test, pool_size)

#     # Normalize data by training set mean and std
#     train, test = z_score_normalize(train, test)

#     # clip to avoid numerical instability
#     c = 5.0
#     train = train.clip(-c, c)
#     test = test.clip(-c, c)
#     return train, test