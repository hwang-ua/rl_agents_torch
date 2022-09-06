import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical

from core.network import network_utils, network_bodies
from core.utils import torch_utils
import core.network.representation as representation

class GaussianPolicy(nn.Module):
    # def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
    def __init__(self, device, input_units, hidden_units, num_actions, action_space=None):
        super(GaussianPolicy, self).__init__()
        self.LOG_SIG_MAX = 2
        self.LOG_SIG_MIN = -20
        self.epsilon = 1e-8

        body = network_bodies.FCBody(device, input_units, hidden_units=tuple(hidden_units))
        self.mean_linear = network_utils.layer_init_xavier(nn.Linear(body.feature_dim, num_actions))
        self.log_std_linear = network_utils.layer_init_xavier(nn.Linear(body.feature_dim, num_actions))

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.tensor(
                float(action_space["high"] - action_space["low"]) / 2.)
            self.action_bias = torch.tensor(
                float(action_space["high"] + action_space["low"]) / 2.)

        self.device = device
        self.body = body
        self.to(device)

    def forward(self, x):
        if not isinstance(x, torch.Tensor): x = torch_utils.tensor(x, self.device)
        if len(x.shape) > 2: x = x.view(x.shape[0], -1)
        x = self.body(x)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX)
        return mean, log_std

    def sample(self, x):
        if not isinstance(x, torch.Tensor): x = torch_utils.tensor(x, self.device)
        if len(x.shape) > 2: x = x.view(x.shape[0], -1)

        mean, log_std = self.forward(x)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class AwacMLPCont(nn.Module):
    def __init__(self, device, obs_dim, act_dim, hidden_sizes, action_range=1.0, rep=None, init_type='xavier', info=None):
        super().__init__()
        self.device = device
        if rep is None:
            self.rep = lambda x: x
        else:
            self.rep = rep()
            obs_dim = self.rep.output_dim
        body = network_bodies.FCBody(device, obs_dim, hidden_units=tuple(hidden_sizes), init_type=init_type)
        body_out = obs_dim if hidden_sizes==[] else hidden_sizes[-1]
        self.body = body
        if init_type == "xavier":
            self.mu_layer = network_utils.layer_init_xavier(nn.Linear(body_out, act_dim))
        elif init_type == "uniform":
            self.mu_layer = network_utils.layer_init_uniform(nn.Linear(body_out, act_dim))
        elif init_type == "zeros":
            self.mu_layer = network_utils.layer_init_zero(nn.Linear(body_out, act_dim))
        elif init_type == "constant":
            self.mu_layer = network_utils.layer_init_constant(nn.Linear(body_out, act_dim), const=info)
        else:
            raise ValueError('init_type is not defined: {}'.format(init_type))

        self.log_std_logits = nn.Parameter(torch.zeros(act_dim, requires_grad=True))
        self.min_log_std = -6
        self.max_log_std = 0
        # self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.action_range = action_range
    
    def forward(self, obs, deterministic=False):
        if not isinstance(obs, torch.Tensor): obs = torch_utils.tensor(obs, self.device)
        # print("Using the special policy")
        recover_size = False
        if len(obs.size()) == 1:
            recover_size = True
            obs = obs.reshape((1, -1))
        obs = self.rep(obs)
        net_out = self.body(obs)
        mu = self.mu_layer(net_out)
        mu = torch.tanh(mu) * self.action_range
        
        log_std = torch.sigmoid(self.log_std_logits)
        
        log_std = self.min_log_std + log_std * (self.max_log_std - self.min_log_std)
        std = torch.exp(log_std)
        # print("Std: {}".format(std))
        
        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()
        
        # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
        # NOTE: The correction formula is a little bit magic. To get an understanding
        # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
        # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
        # Try deriving it yourself as a (very difficult) exercise. :)
        logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
        # logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        
        if recover_size:
            pi_action, logp_pi = pi_action[0], logp_pi[0]
        return pi_action, logp_pi
    
    def get_logprob(self, obs, actions):
        if not isinstance(obs, torch.Tensor): obs = torch_utils.tensor(obs, self.device)
        if not isinstance(actions, torch.Tensor): actions = torch_utils.tensor(actions, self.device)
        obs = self.rep(obs)
        net_out = self.body(obs)
        mu = self.mu_layer(net_out)
        mu = torch.tanh(mu) * self.action_range
        log_std = torch.sigmoid(self.log_std_logits)
        # log_std = self.log_std_layer(net_out)
        log_std = self.min_log_std + log_std * (
            self.max_log_std - self.min_log_std)
        std = torch.exp(log_std)
        pi_distribution = Normal(mu, std)
        logp_pi = pi_distribution.log_prob(actions).sum(axis=-1)
        return logp_pi
    

class AwacMLPDiscrete(nn.Module):
    def __init__(self, device, obs_dim, act_dim, hidden_sizes, rep=None, init_type='xavier', info=None):
        super().__init__()
        self.device = device
        if rep is None:
            self.rep = lambda x: x
        else:
            self.rep = rep()
            obs_dim = self.rep.output_dim
        body = network_bodies.FCBody(device, obs_dim, hidden_units=tuple(hidden_sizes), init_type=init_type)
        body_out = obs_dim if hidden_sizes==[] else hidden_sizes[-1]
        self.body = body
        if init_type == "xavier":
            self.mu_layer = network_utils.layer_init_xavier(nn.Linear(body_out, act_dim))
        elif init_type == "uniform":
            self.mu_layer = network_utils.layer_init_uniform(nn.Linear(body_out, act_dim))
        elif init_type == "zeros":
            self.mu_layer = network_utils.layer_init_zero(nn.Linear(body_out, act_dim))
        elif init_type == "constant":
            self.mu_layer = network_utils.layer_init_constant(nn.Linear(body_out, act_dim), const=info)
        else:
            raise ValueError('init_type is not defined: {}'.format(init_type))

        # self.body = network_bodies.FCBody(device, obs_dim, hidden_units=tuple(hidden_sizes))
        # self.mu_layer = nn.Linear(body_out, act_dim)
        
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
        # if cfg.policy_fn_config['policy_type'] == "gaussian":
        #     return lambda: GaussianPolicy(cfg.device, np.prod(cfg.policy_fn_config['in_dim']),
        #                                   cfg.policy_fn_config['hidden_units'],
        #                                   cfg.action_dim, action_space=cfg.action_range)
        if cfg.policy_fn_config['policy_type'] == "policy-cont":
            return lambda: AwacMLPCont(cfg.device, np.prod(cfg.policy_fn_config['in_dim']),
                                       cfg.action_dim, cfg.policy_fn_config['hidden_units'],
                                       action_range=cfg.action_range,
                                       rep=cfg.rep_fn,
                                       init_type=cfg.policy_fn_config.get('init_type', 'xavier'),
                                       info=cfg.policy_fn_config.get('info', None),
                                       )
        elif cfg.policy_fn_config['policy_type'] == 'policy-discrete':
            return lambda: AwacMLPDiscrete(cfg.device, np.prod(cfg.policy_fn_config['in_dim']),
                                           cfg.action_dim, cfg.policy_fn_config['hidden_units'],
                                           rep=cfg.rep_fn,
                                           init_type=cfg.policy_fn_config.get('init_type', 'xavier'),
                                           info=cfg.policy_fn_config.get('info', None),
                                           )
        else:
            raise NotImplementedError