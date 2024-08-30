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
    def __init__(self, D, n_features): #D is the dimensionality of the time series.
        super().__init__()
        self.gpt = GPT2Model.from_pretrained("gpt2").eval()
        hidden_size = self.gpt.config.n_embd  # 1600 for gpt2-xl, 768 for gpt2
        n_times = max(1, n_features // hidden_size)
        self.proj = nn.Linear(D, n_times*hidden_size).eval()
        self.hidden_size = hidden_size

        # Set requires_grad to False for all parameters
        for param in self.proj.parameters():
            param.requires_grad = False
        for param in self.gpt.parameters():
            param.requires_grad = False
        
        # Ensure the model is in eval mode
        self.eval()


    def forward(self, x):
        with torch.no_grad():
            x = self.proj(x)
            outputs = []
            for i, splitted_x in enumerate(torch.split(x, self.hidden_size, dim=-1)):
                print("split",i)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                output = self.gpt(inputs_embeds=splitted_x, output_hidden_states=True)
                outputs.append(output.hidden_states[-1])
        return torch.cat(outputs, dim=-1)


#########################################
##### Timeseries Feature Extractors #####
#########################################


class ProjTimeseriesGPT2Last(TimeseriesFeatureExtractor):
    def __init__(self, n_features, max_batch=512):
        super().__init__(max_batch)
        self.n_features = n_features

    def fit(self, X: Tensor, y=None):
        """ creates the ProjTimeseriesGPT2 object and initializes it"""
        N, T, D = X.shape
        self.model = ProjTimeseriesGPT2(D, self.n_features).to(X.device).eval()
        return self

    def _batched_transform(self, X:Tensor,): #X shape (N, T, D)
        hidden_state = self.model(X)
        return hidden_state[:, -1, :].clone()
    


class ProjTimeseriesGPT2Multipooling(TimeseriesFeatureExtractor):
    def __init__(self, n_features, max_batch=512):
        super().__init__(max_batch)
        self.n_features = max(1, n_features//4)

    def fit(self, X: Tensor, y=None):
        """ creates the ProjTimeseriesGPT2 object and initializes it"""
        N, T, D = X.shape
        self.model = ProjTimeseriesGPT2(D, self.n_features).to(X.device).eval()
        return self

    def _batched_transform(self, X:Tensor,): #X shape (N, T, D)
        hidden_state = self.model(X)
        return four_multirocket_pooling(hidden_state.permute(0,2,1)).clone()



class ProjTimeseriesGPT2MultipoolingAndLast(TimeseriesFeatureExtractor):
    def __init__(self, n_features, max_batch=512):
        super().__init__(max_batch)
        self.n_features = max(1, n_features//5)

    def fit(self, X: Tensor, y=None):
        """ creates the ProjTimeseriesGPT2 object and initializes it"""
        N, T, D = X.shape
        self.model = ProjTimeseriesGPT2(D, self.n_features).to(X.device).eval()
        return self

    def _batched_transform(self, X:Tensor,): #X shape (N, T, D)
        hidden_state = self.model(X)
        pooling = four_multirocket_pooling(hidden_state.permute(0,2,1))
        return torch.cat([hidden_state[:, -1, :], pooling], dim=1).clone()