import os
import numpy as np
import torch
import torch.nn as nn

from core.network import network_architectures
from core.utils import torch_utils

class RawSA:
    def __init__(self, cfg):
        self.output_dim = np.prod(cfg.rep_fn_config['in_dim']) * cfg.action_dim
        self.device = cfg.device

    def __call__(self, state, action):
        assert len(state.shape) == 1
        vec = np.zeros(self.output_dim)
        vec[action*len(state): (action+1)*len(state)] = state
        return vec

    def parameters(self):
        return []

    def state_dict(self):
        return []

    def load_state_dict(self, item):
        return


class IdentityRepresentation(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.output_dim = cfg.rep_fn_config['out_dim']
        self.device = cfg.device

    def forward(self, x):
        return torch_utils.tensor(x, self.device)


class OneHotRepresentation(RawSA):
    def __init__(self, cfg, ranges=None):
        super().__init__(cfg)
        self.ranges = ranges if ranges is not None else cfg.rep_fn_config['range']
        self.output_dim = np.prod(self.ranges)
        self.device = cfg.device

    def __call__(self, x):
        reshaped = False
        if isinstance(x, torch.Tensor): x = torch_utils.to_np(x)
        if len(x.shape) == 1:
            x = x.reshape((1, -1))
            reshaped = True
        r = np.zeros((len(x), self.output_dim))
        idxs = np.zeros(len(x))
        for d in range(len(self.ranges)):
            v = x[:, d]
            idxs += v * np.prod(self.ranges[d+1:])
        r[np.arange(len(x)), idxs.astype(int)] = 1
        if reshaped:
            r = r[0]
        return torch_utils.tensor(r, self.device)
