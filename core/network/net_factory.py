import numpy as np

from core.network import network_architectures, representation


class NetFactory:
    
    @classmethod
    def get_rep_fn(cls, cfg):
        # Creates a function for constructing the value value_network
        if cfg.rep_fn_config['rep_type'] == 'identity':
            return lambda: representation.IdentityRepresentation(cfg)
        elif cfg.rep_fn_config['rep_type'] == 'one_hot':
            return lambda: representation.OneHotRepresentation(cfg)
        else:
            raise NotImplementedError

    @classmethod
    def get_val_fn(cls, cfg):
        # Creates a function for constructing the value value_network
        if cfg.val_fn_config['val_fn_type'] == 'fc':
            return lambda: network_architectures.FCNetwork(cfg.device, np.prod(cfg.rep_fn().output_dim),
                                                           cfg.val_fn_config['hidden_units'],
                                                           cfg.val_fn_config.get('out_dim', cfg.action_dim), init_type=cfg.val_fn_config['init_type'])
        elif cfg.val_fn_config['val_fn_type'] == 'conv':
            return lambda: network_architectures.ConvNetwork(cfg.device, cfg.state_dim,
                                                             cfg.val_fn_config.get('out_dim', cfg.action_dim), cfg.val_fn_config['conv_architecture'], init_type=cfg.rep_fn_config['init_type'])
        elif cfg.val_fn_config['val_fn_type'] == 'linear':
            return lambda: network_architectures.LinearNetwork(cfg.device, np.prod(cfg.rep_fn().output_dim),
                                                               cfg.val_fn_config.get('out_dim', cfg.action_dim), init_type=cfg.val_fn_config['init_type'])
        elif cfg.val_fn_config['val_fn_type'] == 'constant':
            return lambda: network_architectures.Constant(cfg.device, cfg.val_fn_config.get('out_dim', cfg.action_dim), cfg.val_fn_config.get('value', 0))
        else:
            raise NotImplementedError

    @classmethod
    def get_double_critic_fn(cls, cfg):
        # Creates a function for constructing the actor network
        if cfg.critic_fn_config['network_type'] == 'fc':
            return lambda: network_architectures.DoubleCriticDiscrete(cfg.device, np.prod(cfg.critic_fn_config['in_dim']),
                                                                      cfg.critic_fn_config['hidden_units'],
                                                                      cfg.critic_fn_config.get('out_dim', cfg.action_dim),
                                                                      rep=cfg.rep_fn)
        else:
            raise NotImplementedError
        
    @classmethod
    def get_state_val_fn(cls, cfg):
        if hasattr(cfg, 'val_fn_config'):
            if cfg.val_fn_config['network_type'] == 'fc':
                return lambda: network_architectures.FCNetwork(cfg.device, np.prod(cfg.val_fn_config['in_dim']),
                                                               cfg.val_fn_config['hidden_units'], 1,
                                                               rep=cfg.rep_fn)
            else:
                raise NotImplementedError
        else:
            return None