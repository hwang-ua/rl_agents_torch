import os.path
import pickle
import time
import copy
import numpy as np

EARLYCUTOFF = "EarlyCutOff"

def load_testset(paths, run=0):
    if paths is not None:
        testsets = {}
        for name in paths:
            if name == "buffer":
                testsets[name] = {
                    'states': None,
                    'actions': None,
                    'rewards': None,
                    'next_states': None,
                    'terminations': None,
                }
            elif name == "diff_pi":
                pth = paths[name]
                with open(pth.format(run), 'rb') as f:
                    pairs = pickle.load(f)
                testsets[name] = {
                    'states': pairs["state0"],
                    'actions': None,
                    'rewards': None,
                    'next_states': pairs["state1"],
                    'terminations': None,
                }
            else:
                pth = paths[name]
                with open(pth.format(run), 'rb') as f:
                    testsets[name] = pickle.load(f)
        return testsets
    else:
        return {}

def run_steps(agent):
    # valuesets = load_true_values(agent.cfg)
    t0 = time.time()
    transitions = []
    agent.populate_returns(initialize=True)
    agent.random_fill_buffer(agent.cfg.warm_up_step)
    while True:
        if agent.cfg.log_interval and not agent.total_steps % agent.cfg.log_interval:
            if agent.cfg.tensorboard_logs: agent.log_tensorboard()
            mean, median, min, max = agent.log_file(elapsed_time=agent.cfg.log_interval / (time.time() - t0))
            t0 = time.time()

        if agent.cfg.eval_interval and not agent.total_steps % agent.cfg.eval_interval:
            if agent.cfg.evaluate_action_value and agent.total_steps > 1:
                agent.draw_action_value()
            agent.reset_population_flag() # Done evaluation, regenerate data next time
            # t0 = time.time()
        if agent.cfg.max_steps and agent.total_steps >= agent.cfg.max_steps:
            break

        seq = agent.step()
        
        if agent.cfg.early_cut_off and seq == EARLYCUTOFF:
            break

        if agent.cfg.log_observations:
            transitions.append(copy.deepcopy(seq))

    if agent.cfg.save_params:
        agent.save()

    if agent.cfg.log_observations:
        data_dir = agent.cfg.get_data_dir()
        with open(os.path.join(data_dir, 'transition.pkl'), 'wb') as f:
            pickle.dump(transitions, f)

