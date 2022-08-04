import numpy as np
import torch
import torch.nn as nn

from core.network import network_utils, network_bodies
from core.utils import torch_utils



class FCNetwork(nn.Module):
    def __init__(self, device, input_units, hidden_units, output_units, head_activation=lambda x:x, init_type='xavier', rep=None):
        super().__init__()
        if rep is None:
            self.rep = lambda x:x
        else:
            self.rep = rep()
            input_units = self.rep.output_dim

        body = network_bodies.FCBody(device, input_units, hidden_units=tuple(hidden_units), init_type=init_type)
        self.body = body
        if init_type == "xavier":
            self.fc_head = network_utils.layer_init_xavier(nn.Linear(body.feature_dim, output_units))
        elif init_type == "uniform":
            self.fc_head = network_utils.layer_init_uniform(nn.Linear(body.feature_dim, output_units))
        else:
            raise ValueError('init_type is not defined: {}'.format(init_type))

        self.device = device
        self.head_activation = head_activation
        self.to(device)

    def forward(self, x):
        if not isinstance(x, torch.Tensor): x = torch_utils.tensor(x, self.device)
        if len(x.shape) > 2: x = x.view(x.shape[0], -1)
        x = self.rep(x)
        y = self.body(x)
        y = self.fc_head(y)
        y = self.head_activation(y)
        return y


class DoubleCriticDiscrete(nn.Module):
    def __init__(self, device, input_units, hidden_units, output_units, head_activation=lambda x: x, init_type='xavier', rep=None):
        super().__init__()
        self.device = device
        if rep is None:
            self.rep = lambda x:x
        else:
            self.rep = rep()
            input_units = self.rep.output_dim
        self.q1_net = FCNetwork(device, input_units, hidden_units, output_units, head_activation=head_activation, init_type=init_type)
        self.q2_net = FCNetwork(device, input_units, hidden_units, output_units, head_activation=head_activation, init_type=init_type)
    
    # def forward(self, x, a):
    def forward(self, x):
        if not isinstance(x, torch.Tensor): x = torch_utils.tensor(x, self.device)
        recover_size = False
        if len(x.size()) == 1:
            recover_size = True
            x = x.reshape((1, -1))
            
        x = self.rep(x)
        q1 = self.q1_net(x)
        q2 = self.q2_net(x)

        if recover_size:
            q1 = q1[0]
            q2 = q2[0]
        return q1, q2
