import os
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import torch

from core.agent import base
from core.utils import torch_utils


class CQLAgentOffline(base.ValueBased):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.offline_param_init()
        self.alpha = cfg.cql_alpha

    def get_data(self):
        return self.get_offline_data()

    def feed_data(self):
        self.update_stats(0, None)
        return

    def update(self, data):
        in_ = data['obs']
        act = data['act']
        ns = data['obs2']
        t = data['done']
        r = data['reward']
        
        """
        According to https://github.com/BY571/CQL/blob/main/CQL-DQN/agent.py
        def learn(self, experiences):
        """
        q_s = self.val_net(self.rep_net(in_))
        q_s_a = q_s[np.arange(len(in_)), act]
        with torch.no_grad():
            q_tar = r + (self.cfg.discount * (1 - t) * self.targets.val_net(self.targets.rep_net(ns)).max(1)[0])
        loss = self.cql_loss(q_s, q_s_a, q_tar)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.rep_net.parameters()) + list(self.val_net.parameters()), 1)
        self.optimizer.step()

        self.training_loss.append(torch_utils.to_np(loss))
        self.update_stats(0, None)
        if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
            self.sync_target()
        return

    # def epoch_step(self):
    #     train_s, train_a, train_r, train_ns, train_t, train_na, _, _, _ = self.trainset
    #
    #     self.agent_rng.shuffle(self.training_indexs)
    #     ls_epoch = []
    #     for b in range(int(np.ceil(self.training_size / self.cfg.batch_size))):
    #         idxs = self.training_indexs[b * self.cfg.batch_size: (b + 1) * self.cfg.batch_size]
    #         in_ = torch_utils.tensor(self.cfg.state_normalizer(train_s[idxs]), self.cfg.device)
    #         act = train_a[idxs]
    #         r = torch_utils.tensor(train_r[idxs], self.cfg.device)
    #         ns = torch_utils.tensor(self.cfg.state_normalizer(train_ns[idxs]), self.cfg.device)
    #         t = torch_utils.tensor(train_t[idxs], self.cfg.device)
    #         na = train_na[idxs]
    #
    #         """
    #         According to https://github.com/BY571/CQL/blob/main/CQL-DQN/agent.py
    #         def learn(self, experiences):
    #         """
    #         q_s = self.val_net(self.rep_net(in_))
    #         q_s_a = q_s[np.arange(len(in_)), act]
    #         q_pred = q_s_a
    #         with torch.no_grad():
    #             q_tar = r + (self.cfg.discount * (1 - t) * self.targets.val_net(self.targets.rep_net(ns)).max(1)[0])
    #         loss = self.cql_loss(q_s, q_s_a, q_pred, q_tar)
    #
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(list(self.rep_net.parameters()) + list(self.val_net.parameters()), 1)
    #         self.optimizer.step()
    #         ls_epoch.append(torch_utils.to_np(loss))
    #
    #
    #     self.training_loss.append(np.array(ls_epoch).mean())
    #     self.update_stats(0, None)
    #     if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
    #         self.targets.rep_net.load_state_dict(self.rep_net.state_dict())
    #         self.targets.val_net.load_state_dict(self.val_net.state_dict())
    #
    #     return self.test_fn()
    
    def cql_loss(self, q_s, q_s_a, q_tar):
        cql1_loss = torch.logsumexp(q_s, dim=1).mean() - q_s_a.mean()
        bellmann_error = self.vf_loss(q_s_a, q_tar)
        loss = self.alpha * cql1_loss + 0.5 * bellmann_error
        return loss

    # def test_fn(self):
    #     test_s, test_a, test_r, test_sp, test_term, _, _, _, _ = self.testset  # test_ap will be replaced if following the current estimation
    #     test_s = torch_utils.tensor(self.cfg.state_normalizer(test_s), self.cfg.device)
    #     test_r = torch_utils.tensor(test_r, self.cfg.device)
    #     test_sp = torch_utils.tensor(self.cfg.state_normalizer(test_sp), self.cfg.device)
    #     test_term = torch_utils.tensor(test_term, self.cfg.device)
    #     with torch.no_grad():
    #         q_s = self.val_net(self.rep_net(test_s))
    #         q_s_a = q_s[np.arange(len(test_s)), test_a]
    #         q_pred = q_s_a
    #         q_tar = test_r + (self.cfg.discount * (1 - test_term) * self.targets.val_net(self.targets.rep_net(test_sp)).max(1)[0])
    #         tloss = self.cql_loss(q_s, q_s_a, q_pred, q_tar)
    #
    #     if tloss - self.tloss_rec > 0:
    #         self.tloss_increase += 1
    #     else:
    #         self.tloss_increase = 0
    #     self.tloss_rec = tloss
    #     self.test_loss.append(tloss)
    #     if self.tloss_increase > self.cfg.early_cut_threshold:
    #         return "EarlyCutOff"
    #     return
    
    # def log_file(self, elapsed_time=-1):
    #     self.log_offline_training(elapsed_time)
