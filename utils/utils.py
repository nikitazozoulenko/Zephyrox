from typing import List, Dict, Set, Any, Optional, Tuple, Literal, Callable
import numpy as np
import os
import sys
from tqdm import tqdm
import inspect

#####################################################################
################## Print torch tensors ##############################
#####################################################################


def mod_retrieve_name(var): # TODO make this work for calls of the type print_shape(a+2)
    callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


def print_shape(X):
    """Prints the name and shape of an array."""
    print(X.shape, mod_retrieve_name(X)[-1], X.dtype, "\n")


def print_name(X):
    """Prints the name and shape of an array, then the array itself."""
    if hasattr(X, 'shape'):
        print(X.shape, mod_retrieve_name(X)[-1], X.dtype)
    else:
        print(mod_retrieve_name(X)[-1])
    print(X, "\n")