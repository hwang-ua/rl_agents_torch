import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical

from core.network import network_utils, network_bodies
from core.utils import torch_utils
import core.network.representation as representation

class AwacMLPDiscrete(nn.Module):
    def __init__(self, device, obs_dim, act_dim, hidden_sizes, rep=None):
        super().__init__()
        self.device = device
        if rep is None:
            self.rep = lambda x: x
        else:
            self.rep = rep()
            obs_dim = self.rep.output_dim
            
        self.body = network_bodies.FCBody(device, obs_dim, hidden_units=tuple(hidden_sizes))
        body_out = obs_dim if hidden_sizes==[] else hidden_sizes[-1]
        self.mu_layer = nn.Linear(body_out, act_dim)
        
        self.log_std_logits = nn.Parameter(torch.zeros(act_dim, requires_grad=True))
        self.min_log_std = -6
        self.max_log_std = 0
        # self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
    
    def forward(self, obs):
        if not isinstance(obs, torch.Tensor): obs = torch_utils.tensor(obs, self.device)
        # print("Using the special policy")
        recover_size = False
        if len(obs.size()) == 1:
            recover_size = True
            obs = obs.reshape((1, -1))
        obs = self.rep(obs)
        net_out = self.body(obs)
        probs = self.mu_layer(net_out)
        probs = F.softmax(probs, dim=1)

        m = Categorical(probs)
        action = m.sample()
        logp = m.log_prob(action)
        if recover_size:
            action, logp = action[0], logp[0]
        return action, logp
    
    def get_logprob(self, obs, actions):
        if not isinstance(obs, torch.Tensor): obs = torch_utils.tensor(obs, self.device)
        if not isinstance(actions, torch.Tensor): actions = torch_utils.tensor(actions, self.device)
        obs = self.rep(obs)
        net_out = self.body(obs)
        probs = self.mu_layer(net_out)
        probs = F.softmax(probs, dim=1)
        m = Categorical(probs)
        logp_pi = m.log_prob(actions)
        return logp_pi
    

class PolicyFactory:
    @classmethod
    def get_policy_fn(cls, cfg):
        if cfg.policy_fn_config['policy_type'] == "policy-discrete":
            return lambda: AwacMLPDiscrete(cfg.device, np.prod(cfg.policy_fn_config['in_dim']),
                                           cfg.action_dim, cfg.policy_fn_config['hidden_units'],
                                           rep=cfg.rep_fn)
        else:
            raise NotImplementedError