import argparse
import copy
import os
import pickle as pkl
import torch

from core.environment import gridworld
from core.tabular.dataset import *
from core.tabular.agents import *

import core.agent.agent_factory as agent
import core.network.net_factory as network
import core.network.policy_factory as policy
import core.network.optimizer as optimizer
import core.network.activations as activations
import core.component.constraint as constraint
import core.component.replay as replay
import core.utils.normalizer as normalizer
from core.utils import torch_utils, schedule, logger, run_funcs, format_path
import core.utils.testers as tester
from experiment.sweeper.sweeper import Sweeper

np.set_printoptions(precision=3)
np.set_printoptions(linewidth=np.inf)

def replace_weight_init(dictionary, key, value):
    if hasattr(cfg, dictionary):
        a = getattr(cfg, dictionary)
        a[key] = value
    return dictionary
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run_tarbular")
    parser.add_argument('--id', default=0, type=int)
    parser.add_argument('--config-file', default='experiment/config/test_v0/mountain_car/dqn/temp.json')
    parser.add_argument('--device', default=-1, type=int, )
    parser.add_argument('--policy', default='mixed', help='opt / mixed / random')
    parser.add_argument('--size', default=10000, type=int)
    parser.add_argument('--is_ac', default=1, type=int)
    parser.add_argument('--weight-init', default=10., type=float)
    args = parser.parse_args()
    torch_utils.set_one_thread()
  
    project_root = os.path.abspath(os.path.dirname(__file__))
    extra_folder = "data_{}/init_{}".format(args.policy, args.weight_init)
    cfg = Sweeper(project_root, args.config_file, extra_name=extra_folder).parse(args.id)
    cfg.device = torch_utils.select_device(args.device)
    torch_utils.random_seed(cfg.seed)
  
    replace_weight_init("rep_fn_config", "info", args.weight_init)
    replace_weight_init("val_fn_config", "info", args.weight_init)
    replace_weight_init("policy_fn_config", "info", args.weight_init)
    replace_weight_init("critic_fn_config", "info", args.weight_init)
    
    cfg.env_fn = lambda: gridworld.GridWorld(random_start=False)
    gw = cfg.env_fn()
  
    """
    Value iteration
    """
    opt_q = value_iteration(gw.P, gw.r, cfg.discount, 10000)
  
    """
    Data collection
    """
    rs_gw = gridworld.GridWorld(random_start=True)
    evalset = random_data_collection(rs_gw, opt_q, 1000, 10)
    if args.policy == 'opt':
        # print(args.)
        dataset = optimal_data_collection(gw, opt_q, args.size, cfg.timeout)
    elif args.policy == 'mixed':
        dataset = mixed_data_collection(gw, opt_q, args.size, cfg.timeout, p_opt=0.01)
    elif args.policy == 'random':
        dataset = random_data_collection(rs_gw, opt_q, args.size, cfg.timeout)
    elif args.policy == 'missing_a':
        dataset = mixed_data_collection(gw, opt_q, args.size, cfg.timeout)
        dataset = remove_action(dataset, [[0, 6], [0, 6]], 2)
    elif args.policy == 'missing_s':
        dataset = mixed_data_collection(gw, opt_q, args.size, cfg.timeout)
        dataset = remove_state(dataset, np.array(np.where(np.eye(6)==1)).T)
    else:
        raise NotImplementedError
  
    data_dir = cfg.get_data_dir()
    dataset_pth = data_dir+"/dataset.pkl"
    evalset_pth = data_dir+"/evalset.pkl"
    q_pth = data_dir+"/q_table.pkl"
    with open(dataset_pth, 'wb') as f:
        pkl.dump(dataset, f)
    with open(evalset_pth, 'wb') as f:
        pkl.dump(evalset, f)
    torch.save(opt_q.reshape((gw.num_cols, gw.num_rows, gw.num_actions)), q_pth)

    cfg.offline_data_path = {"optimal": dataset_pth}
    cfg.evalset_path = {"dataEps100": evalset_pth}
    cfg.qmax_table = q_pth
  
    """
    Visualize dataset
    """
    visualize_dataset(dataset, gw, save=cfg.get_data_dir())

    """
    Run the experiment
    """
    cfg.rep_fn = network.NetFactory.get_rep_fn(cfg)
    if args.is_ac:
        cfg.policy_fn = policy.PolicyFactory.get_policy_fn(cfg)
        cfg.critic_fn = network.NetFactory.get_double_critic_fn(cfg)
        cfg.state_value_fn = network.NetFactory.get_state_val_fn(cfg)
        cfg.policy_optimizer_fn = optimizer.OptFactory.get_optimizer_fn(cfg)
        cfg.critic_optimizer_fn = optimizer.OptFactory.get_optimizer_fn(cfg)
        cfg.alpha_optimizer_fn = optimizer.OptFactory.get_optimizer_fn(cfg)
        cfg.vs_optimizer_fn = optimizer.OptFactory.get_optimizer_fn(cfg)
    else:
        cfg.rep_activation_fn = activations.ActvFactory.get_activation_fn(cfg)
        cfg.val_fn = network.NetFactory.get_val_fn(cfg)
        cfg.constr_fn = constraint.ConstraintFactory.get_constr_fn(cfg)
        cfg.optimizer_fn = optimizer.OptFactory.get_optimizer_fn(cfg)
        cfg.vf_loss_fn = optimizer.OptFactory.get_vf_loss_fn(cfg)
        cfg.eps_schedule = schedule.ScheduleFactory.get_eps_schedule(cfg)

    cfg.replay_fn = replay.ReplayFactory.get_replay_fn(cfg)
    # cfg.eps_schedule = schedule.ScheduleFactory.get_eps_schedule(cfg)
    cfg.state_normalizer, cfg.reward_normalizer = normalizer.NormalizerFactory.get_normalizer(cfg)
    cfg.offline_data = run_funcs.load_testset(cfg.offline_data_path, cfg.run)
    cfg.eval_data = run_funcs.load_testset(cfg.evalset_path, cfg.run)
    cfg.tester_fn = tester.TesterFactory.get_tester_fn(cfg)

    # Setting up the logger
    cfg.logger = logger.Logger(cfg)
    cfg.log_config()

    agent_obj = agent.AgentFactory.create_agent_fn(cfg)()
    run_funcs.run_steps(agent_obj)
    
    os.remove(dataset_pth)
    os.remove(evalset_pth)
    os.remove(q_pth)