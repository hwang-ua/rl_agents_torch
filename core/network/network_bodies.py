from functools import reduce
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

from core.network import network_utils

class FCBody(nn.Module):
    def __init__(self, device, input_dim, hidden_units=(64, 64), activation=functional.relu, init_type='xavier'):
        super().__init__()
        self.to(device)
        self.device = device
        dims = (input_dim,) + hidden_units
        self.layers = nn.ModuleList([network_utils.layer_init_xavier(nn.Linear(dim_in, dim_out).to(device)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])

        if init_type == "xavier":
            self.layers = nn.ModuleList([network_utils.layer_init_xavier(nn.Linear(dim_in, dim_out).to(device)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        elif init_type == "uniform":
            self.layers = nn.ModuleList([network_utils.layer_init_uniform(nn.Linear(dim_in, dim_out).to(device)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        else:
            raise ValueError('init_type is not defined: {}'.format(init_type))

        self.activation = activation
        self.feature_dim = dims[-1]

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return x

