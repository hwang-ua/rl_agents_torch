import numpy as np
from collections import namedtuple
from core.agent import base
from core.utils import torch_utils
from core.utils import helpers

import os
import torch
from torch.nn.utils import clip_grad_norm_
# from networks import Critic, Actor, Value

"""
Changed based on https://github.com/BY571/Implicit-Q-Learning/blob/main/discrete_iql/agent.py
"""

class IQLOnline(base.ActorCritic):
    def __init__(self, cfg):
        super(IQLOnline, self).__init__(cfg)

        # self.state_size = cfg.state_dim
        # self.action_size = cfg.action_dim
        # self.device = cfg.device
        
        self.clip_grad_param = cfg.clip_grad_param # 100
        self.temperature = cfg.temperature #3
        self.expectile = cfg.expectile #torch.FloatTensor([0.8]).to(device)
        
        self.value_net = cfg.state_value_fn()
        if 'load_params' in self.cfg.val_fn_config and self.cfg.val_fn_config['load_params']:
            self.load_state_value_fn(cfg.val_fn_config['path'])
        self.value_optimizer = cfg.vs_optimizer_fn(list(self.value_net.parameters()))

        if cfg.agent_name == 'IQLOnline' and cfg.load_offline_data:
            self.fill_offline_data_to_buffer()

    def load_state_value_fn(self, parameters_dir):
        path = os.path.join(self.cfg.data_root, parameters_dir)
        self.value_net.load_state_dict(torch.load(path, map_location=self.cfg.device))
        self.cfg.logger.info("Load state value function from {}".format(path))

    def compute_loss_pi(self, data):
        states, actions = data['obs'], data['act']
        with torch.no_grad():
            v = self.value_net(states)
            q1, q2 = self.ac_targ.q1q2(states)
        q1_pi, q2_pi = q1[np.arange(len(actions)), actions], q2[np.arange(len(actions)), actions]
        min_Q = torch.min(q1_pi, q2_pi)
        
        exp_a = torch.exp((min_Q - v) * self.temperature)
        exp_a = torch.min(exp_a, torch.FloatTensor([100.0]).to(states.device)).squeeze(-1)
        
        log_probs = self.ac.pi.get_logprob(states, actions)
        actor_loss = -(exp_a * log_probs).mean()
        
        return actor_loss, log_probs
    
    def compute_loss_value(self, data):
        states, actions = data['obs'], data['act']
        with torch.no_grad():
            q1, q2 = self.ac_targ.q1q2(states)
        q1_pi, q2_pi = q1[np.arange(len(actions)), actions], q2[np.arange(len(actions)), actions]
        min_Q = torch.min(q1_pi, q2_pi)
        
        value = self.value_net(states)
        value_loss = helpers.expectile_loss(min_Q - value, self.expectile).mean()
        return value_loss
    
    def compute_loss_q(self, data):
        states, actions, rewards, next_states, dones = data['obs'], data['act'], data['reward'], data['obs2'], data['done']
        with torch.no_grad():
            next_v = self.value_net(next_states)
            q_target = rewards + (self.gamma * (1 - dones) * next_v)
        
        q1, q2 = self.ac.q1q2(states)
        q1, q2 = q1[np.arange(len(actions)), actions], q2[np.arange(len(actions)), actions]

        critic1_loss = ((q1 - q_target) ** 2).mean()
        critic2_loss = ((q2 - q_target) ** 2).mean()
        loss_q = critic1_loss + critic2_loss
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())
        return loss_q, q_info
        
    def update(self, data):
        self.value_optimizer.zero_grad()
        loss_vs = self.compute_loss_value(data)
        loss_vs.backward()
        self.value_optimizer.step()
        
        loss_q, q_info = self.compute_loss_q(data)
        self.q_optimizer.zero_grad()
        loss_q.backward()
        clip_grad_norm_(self.ac.q1q2.parameters(), self.clip_grad_param)
        self.q_optimizer.step()

        if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
            self.sync_target()

        loss_pi, _ = self.compute_loss_pi(data)
        # self.actor_optimizer.zero_grad()
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        # self.actor_optimizer.step()
        self.pi_optimizer.step()

        return loss_pi.item(), loss_q.item(), loss_vs.item()

    def save(self, early=False):
        parameters_dir = self.cfg.get_parameters_dir()
        if early:
            path = os.path.join(parameters_dir, "actor_net_earlystop")
        elif self.cfg.checkpoints:
            path = os.path.join(parameters_dir, "actor_net_{}".format(self.total_steps))
        else:
            path = os.path.join(parameters_dir, "actor_net")
        torch.save(self.ac.pi.state_dict(), path)

        if early:
            path = os.path.join(parameters_dir, "critic_net_earlystop")
        else:
            path = os.path.join(parameters_dir, "critic_net")
        torch.save(self.ac.q1q2.state_dict(), path)

        if early:
            path = os.path.join(parameters_dir, "vs_net_earlystop")
        else:
            path = os.path.join(parameters_dir, "vs_net")
        torch.save(self.value_net.state_dict(), path)


class IQLOffline(IQLOnline):
    def __init__(self, cfg):
        super(IQLOffline, self).__init__(cfg)
        self.offline_param_init()

    def get_data(self):
        return self.get_offline_data()
        
    def feed_data(self):
        self.update_stats(0, None)
        return

