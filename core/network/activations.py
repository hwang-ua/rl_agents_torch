import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as functional

class ActvFactory:

    @classmethod
    def get_activation_fn(cls, cfg):
        # Creates a function for constructing the value value_network
        if cfg.activation_config['name'] == 'None':
            return lambda x:x
        elif cfg.activation_config['name'] == 'ReLU':
            return functional.relu
        else:
            raise NotImplementedError

