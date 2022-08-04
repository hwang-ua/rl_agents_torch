import os

from core.environment.gridworlds import *

class EnvFactory:
    @classmethod
    def create_env_fn(cls, cfg):
        if cfg.env_name == 'FourRoom':
            return lambda: FourRoom(cfg.seed)
        else:
            print(cfg.env_name)
            raise NotImplementedError
