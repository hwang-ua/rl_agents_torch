import os

import core.environment.gridworld as gw
from core.environment.halfcheetah import HalfCheetah

class EnvFactory:
    @classmethod
    def create_env_fn(cls, cfg):
        if cfg.env_name == 'FourRoomNT':
            return lambda: gw.GridWorld(random_start=False)
        elif cfg.env_name == 'HalfCheetah':
            return lambda: HalfCheetah(cfg.seed)
        else:
            print(cfg.env_name)
            raise NotImplementedError
