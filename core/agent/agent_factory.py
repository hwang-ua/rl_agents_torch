import os

from core.agent.iql import IQLOnline, IQLOffline


class AgentFactory:
    @classmethod
    def create_agent_fn(cls, cfg):
        if cfg.agent_name == 'IQLOnline':
            return lambda: IQLOnline(cfg)
        elif cfg.agent_name == 'IQLOffline':
            return lambda: IQLOffline(cfg)
        else:
            print(cfg.agent_name)
            raise NotImplementedError
