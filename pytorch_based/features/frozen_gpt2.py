from typing import List, Dict, Set, Any, Optional, Tuple, Literal, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import GPT2Model
import numpy as np

from base import TimeseriesFeatureExtractor
from multirocket import four_multirocket_pooling


#############################
##### ProjGPT nn.Module #####
#############################


class ProjTimeseriesGPT2(nn.Module):
    def __init__(self, D): #D is the dimensionality of the time series.
        super().__init__()
        self.gpt = GPT2Model.from_pretrained("gpt2")
        hidden_size = self.gpt.config.n_embd  # 1600 for gpt2-xl, 768 for gpt2
        self.proj = nn.Linear(D, hidden_size)


    def forward(self, x):
        x = self.proj(x)
        outputs = self.gpt(inputs_embeds=x)
        return outputs.last_hidden_state


#########################################
##### Timeseries Feature Extractors #####
#########################################


class ProjTimeseriesGPT2Last(TimeseriesFeatureExtractor):
    def __init__(self, max_batch=512):
        super().__init__(max_batch)

    def fit(self, X: Tensor, y=None):
        """ creates the ProjTimeseriesGPT2 object and initializes it"""
        N, T, D = X.shape
        self.model = ProjTimeseriesGPT2(D).to(X.device)
        return self

    def _batched_transform(self, X:Tensor,): #X shape (N, T, D)
        hidden_state = self.model(X)
        return hidden_state[:, -1, :]
    


class ProjTimeseriesGPT2Multipooling(TimeseriesFeatureExtractor):
    def __init__(self, max_batch=512):
        super().__init__(max_batch)

    def fit(self, X: Tensor, y=None):
        """ creates the ProjTimeseriesGPT2 object and initializes it"""
        N, T, D = X.shape
        self.model = ProjTimeseriesGPT2(D).to(X.device)
        return self

    def _batched_transform(self, X:Tensor,): #X shape (N, T, D)
        hidden_state = self.model(X)
        return four_multirocket_pooling(hidden_state.permute(0,2,1))



class ProjTimeseriesGPT2MultipoolingAndLast(TimeseriesFeatureExtractor):
    def __init__(self, max_batch=512):
        super().__init__(max_batch)

    def fit(self, X: Tensor, y=None):
        """ creates the ProjTimeseriesGPT2 object and initializes it"""
        N, T, D = X.shape
        self.model = ProjTimeseriesGPT2(D).to(X.device)
        return self

    def _batched_transform(self, X:Tensor,): #X shape (N, T, D)
        hidden_state = self.model(X)
        pooling = four_multirocket_pooling(hidden_state.permute(0,2,1))
        return torch.cat([hidden_state[:, -1, :], pooling], dim=1)