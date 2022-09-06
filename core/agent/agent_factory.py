import os

from core.agent.iql import IQLOnline, IQLOffline, IQLOfflineNoV
from core.agent.cql import *


class AgentFactory:
    @classmethod
    def create_agent_fn(cls, cfg):
        if cfg.agent_name == 'DQNAgent':
            return lambda: DQNAgent(cfg)
        elif cfg.agent_name == 'AWACOnline':
            return lambda: AWACOnline(cfg)
        elif cfg.agent_name == 'AWACOffline':
            return lambda: AWACOffline(cfg)
        elif cfg.agent_name == 'SAC':
            return lambda: SAC(cfg)
        elif cfg.agent_name == 'SACOffline':
            return lambda: SACOffline(cfg)
        elif cfg.agent_name == 'IQLOnline':
            return lambda: IQLOnline(cfg)
        elif cfg.agent_name == 'IQLOffline':
            return lambda: IQLOffline(cfg)
        elif cfg.agent_name == 'IQLOffline-RemoveV':
            return lambda: IQLOfflineNoV(cfg)
        elif cfg.agent_name == 'QRCOnline':
            return lambda: QRCOnline(cfg)
        elif cfg.agent_name == 'QRCOffline':
            return lambda: QRCOffline(cfg)
        # elif cfg.agent_name == 'SlowPolicyDQN':
        #     return lambda: SlowPolicyDQN(cfg)
        elif cfg.agent_name == 'FQIAgent':
            return lambda: FQIAgent(cfg)
        elif cfg.agent_name == 'SarsaAgent':
            return lambda: SarsaAgent(cfg)
        # elif cfg.agent_name == 'SarsaOffline':
        #     return lambda: SarsaOffline(cfg)
        elif cfg.agent_name == 'SarsaOfflineBatch':
            return lambda: SarsaOfflineBatch(cfg)
        elif cfg.agent_name == 'MonteCarloAgent':
            return lambda: MonteCarloAgent(cfg)
        elif cfg.agent_name == 'MonteCarloOffline':
            return lambda: MonteCarloOffline(cfg)
        elif cfg.agent_name == 'CQLAgentOffline':
            return lambda: CQLAgentOffline(cfg)
        elif cfg.agent_name == 'InSample':
            return lambda: InSample(cfg)
        elif cfg.agent_name == 'InSampleAC':
            return lambda: InSampleAC(cfg)
        # elif cfg.agent_name == 'QmaxCloneOffline':
        #     return lambda: QmaxCloneOffline(cfg)
        # elif cfg.agent_name == 'QmaxConstrOffline':
        #     return lambda: QmaxConstrOffline(cfg)
        elif cfg.agent_name == 'VI2D':
            return lambda: VI2D(cfg)
        else:
            print(cfg.agent_name)
            raise NotImplementedError