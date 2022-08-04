import os
import numpy as np
import torch
from collections import namedtuple
import copy

import matplotlib.pyplot as plt
import matplotlib as mpl

from core.utils import torch_utils


class Agent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.gamma = cfg.discount
        self.env = cfg.env_fn()
        self.replay = cfg.replay_fn()
        self.eval_env = copy.deepcopy(cfg.env_fn)()

        self.episode_reward = 0
        self.episode_rewards = []
        self.total_steps = 0
        self.reset = True
        self.ep_steps = 0
        self.num_episodes = 0
        self.timeout = cfg.timeout
        self.batch_indices = torch.arange(self.cfg.batch_size).long().to(cfg.device)
        self.ep_returns_queue_train = np.zeros(cfg.stats_queue_size)
        self.ep_returns_queue_test = np.zeros(cfg.stats_queue_size)
        self.train_stats_counter = 0
        self.test_stats_counter = 0
        self.agent_rng = np.random.RandomState(self.cfg.seed)
        self.test_rng = np.random.RandomState(self.cfg.seed)
        self.true_q_predictor = self.cfg.tester_fn.get('true_value_estimator', lambda x:None)
        self.eval_set = self.property_evaluation_dataset(cfg.eval_data)

        self.populate_latest = False
        self.populate_states, self.populate_actions, self.populate_true_qs = None, None, None
        
        self.state = None
        self.action = None
        self.next_state = None

    def offline_param_init(self):
        self.trainset, self.testset = self.training_set_construction(self.cfg.offline_data)
        self.training_size = len(self.trainset[0])
        self.training_indexs = np.arange(self.training_size)

        self.training_loss = []
        self.test_loss = []
        self.tloss_increase = 0
        self.tloss_rec = np.inf

    def feed_data(self):
        if self.reset is True:
            self.state = self.env.reset()
            self.reset = False
        action = self.policy(self.state, self.cfg.eps_schedule())
        next_state, reward, done, _ = self.env.step([action])
        self.replay.feed([self.state, action, reward, next_state, int(done)])
        prev_state = self.state
        self.state = next_state
        self.update_stats(reward, done)
        return prev_state, action, reward, next_state, int(done)

    def get_data(self):
        states, actions, rewards, next_states, terminals = self.replay.sample()
        in_ = torch_utils.tensor(self.cfg.state_normalizer(states), self.cfg.device)
        r = torch_utils.tensor(rewards, self.cfg.device)
        ns = torch_utils.tensor(self.cfg.state_normalizer(next_states), self.cfg.device)
        t = torch_utils.tensor(terminals, self.cfg.device)
        data = {
            'obs': in_,
            'act': actions,
            'reward': r,
            'obs2': ns,
            'done': t
        }
        return data

    def get_offline_data(self):
        train_s, train_a, train_r, train_ns, train_t, train_na, _, _, _ = self.trainset
        idxs = self.agent_rng.randint(0, len(train_s), size=self.cfg.batch_size)
        in_ = torch_utils.tensor(self.cfg.state_normalizer(train_s[idxs]), self.cfg.device)
        act = train_a[idxs]
        r = torch_utils.tensor(train_r[idxs], self.cfg.device)
        ns = torch_utils.tensor(self.cfg.state_normalizer(train_ns[idxs]), self.cfg.device)
        t = torch_utils.tensor(train_t[idxs], self.cfg.device)
        na = train_na[idxs]

        data = {
            'obs': in_,
            'act': act,
            'reward': r,
            'obs2': ns,
            'done': t,
            'act2': na
        }
        return data

    def fill_offline_data_to_buffer(self):
        self.trainset, self.testset = self.training_set_construction(self.cfg.offline_data)
        train_s, train_a, train_r, train_ns, train_t, _, _, _, _ = self.trainset
        for idx in range(len(train_s)):
            self.replay.feed([train_s[idx], train_a[idx], train_r[idx], train_ns[idx], train_t[idx]])

    def step(self):
        self.feed_data()
        data = self.get_data()
        if self.check_update():
            self.update(data)
    
    def check_update(self):
        return NotImplementedError
    
    def update(self, data):
        raise NotImplementedError
        
    def reset_population_flag(self):
        # Done evaluation, regenerate data at next checkpoint
        self.populate_latest = False
        self.populate_states, self.populate_actions, self.populate_true_qs = None, None, None

    def update_stats(self, reward, done):
        self.episode_reward += reward
        self.total_steps += 1
        self.ep_steps += 1
        if done or self.ep_steps == self.timeout:
            self.episode_rewards.append(self.episode_reward)
            self.num_episodes += 1
            if self.cfg.evaluation_criteria == "return":
                self.add_train_log(self.episode_reward)
            elif self.cfg.evaluation_criteria == "steps":
                self.add_train_log(self.ep_steps)
            else:
                raise NotImplementedError
            self.episode_reward = 0
            self.ep_steps = 0
            self.reset = True

    def add_train_log(self, ep_return):
        self.ep_returns_queue_train[self.train_stats_counter] = ep_return
        self.train_stats_counter += 1
        self.train_stats_counter = self.train_stats_counter % self.cfg.stats_queue_size

    def add_test_log(self, ep_return):
        self.ep_returns_queue_test[self.test_stats_counter] = ep_return
        self.test_stats_counter += 1
        self.test_stats_counter = self.test_stats_counter % self.cfg.stats_queue_size

    def populate_returns(self, log_traj=False, total_ep=None, initialize=False):
        total_ep = self.cfg.stats_queue_size if total_ep is None else total_ep
        total_steps = 0
        total_states = []
        total_actions = []
        total_returns = []
        for ep in range(total_ep):
            ep_return, steps, traj = self.eval_episode(log_traj=log_traj)
            total_steps += steps
            total_states += traj[0]
            total_actions += traj[1]
            total_returns += traj[2]
            if self.cfg.evaluation_criteria == "return":
                self.add_test_log(ep_return)
                if initialize:
                    self.add_train_log(ep_return)
            elif self.cfg.evaluation_criteria == "steps":
                self.add_test_log(steps)
                if initialize:
                    self.add_train_log(steps)
            else:
                raise NotImplementedError
        return [total_states, total_actions, total_returns]
    
    # hacking environment and policy
    def populate_returns_random_start(self, start_pos=None, start_policy=None, total_ep=None):
        total_ep = self.cfg.stats_queue_size if total_ep is None else total_ep
        total_states = []
        total_actions = []
        total_returns = []
        for ep in range(total_ep):
            s, a, ret = self.eval_episode_random_start(start_pos=start_pos, random_insert_policy=start_policy)
            total_states.append(s)
            total_actions.append(a)
            total_returns.append(ret)
        return [total_states, total_actions, total_returns]

    def random_fill_buffer(self, total_steps):
        state = self.eval_env.reset()
        for _ in range(total_steps):
            action = self.agent_rng.randint(0, self.cfg.action_dim)
            last_state = state
            state, reward, done, _ = self.eval_env.step([action])
            self.replay.feed([last_state, action, reward, state, int(done)])
            if done:
                state = self.eval_env.reset()

    def eval_episode(self, log_traj=False):
        ep_traj = []
        state = self.eval_env.reset()
        total_rewards = 0
        ep_steps = 0
        done = False
        while True:
            action = self.eval_step(state)
            last_state = state
            state, reward, done, _ = self.eval_env.step([action])
            if log_traj:
                ep_traj.append([last_state, action, reward])
            total_rewards += reward
            ep_steps += 1
            if done or ep_steps == self.cfg.timeout:
                break

        states = []
        actions = []
        rets = []
        if log_traj:
            ret = 0
            for i in range(len(ep_traj)-1, -1, -1):
                s, a, r = ep_traj[i]
                ret = r + self.cfg.discount * ret
                rets.insert(0, ret)
                actions.insert(0, a)
                states.insert(0, s)
        return total_rewards, ep_steps, [states, actions, rets]

    # hacking environment and policy
    def eval_episode_random_start(self, start_pos, random_insert_policy):
        if start_pos is None:
            ep_traj = []
            state = self.eval_env.reset()
            total_rewards = 0
            ep_steps = 0
            while True:
                action = self.eval_step(state)
                last_state = state
                state, reward, done, _ = self.eval_env.step([action])
                ep_traj.append([last_state, action, reward])
                total_rewards += reward
                ep_steps += 1
                if done or ep_steps == self.cfg.timeout:
                    break
            
            random_idx = self.test_rng.randint(len(ep_traj))
            random_start = ep_traj[random_idx][0]
        else:
            random_start = start_pos

        action = random_insert_policy(random_start)
        state, reward, done, _ = self.eval_env.hack_step(random_start, action)
        if done:
            return random_start, action, reward
        ep_traj = []
        total_rewards = 0
        ep_steps = 0
        while True:
            action = self.eval_step(state)
            last_state = state
            state, reward, done, _ = self.eval_env.hack_step(last_state, action)
            ep_traj.append([last_state, action, reward])
            total_rewards += reward
            ep_steps += 1
            if done or ep_steps == self.cfg.timeout:
                break

        ret = 0
        for i in range(len(ep_traj)-1, -1, -1):
            s, a, r = ep_traj[i]
            ret = r + self.cfg.discount * ret
        return s, a, ret

    def eval_episodes(self):
        return

    def log_return(self, log_ary, name, elapsed_time):
        rewards = log_ary
        total_episodes = len(self.episode_rewards)
        mean, median, min_, max_ = np.mean(rewards), np.median(rewards), np.min(rewards), np.max(rewards)

        log_str = '%s LOG: steps %d, episodes %3d, ' \
                  'returns %.2f/%.2f/%.2f/%.2f/%d (mean/median/min/max/num), %.2f steps/s'

        self.cfg.logger.info(log_str % (name, self.total_steps, total_episodes, mean, median,
                                        min_, max_, len(rewards),
                                        elapsed_time))
        return mean, median, min_, max_


    def log_file(self, elapsed_time=-1):
        mean, median, min_, max_ = self.log_return(self.ep_returns_queue_train, "TRAIN", elapsed_time)
        self.populate_states, self.populate_actions, self.populate_true_qs = self.populate_returns(log_traj=True)
        self.populate_latest = True
        mean, median, min_, max_ = self.log_return(self.ep_returns_queue_test, "TEST", elapsed_time)
        return mean, median, min_, max_

    def policy(self, state, eps):
        raise NotImplementedError

    def eval_step(self, state):
        raise NotImplementedError

    def save(self, filename):
        raise NotImplementedError

    def load(self, filename):
        raise NotImplementedError

    def one_hot_action(self, actions):
        one_hot = np.zeros((len(actions), self.cfg.action_dim))
        np.put_along_axis(one_hot, actions.reshape((-1, 1)), 1, axis=1)
        return one_hot
    
    def default_value_predictor(self):
        raise NotImplementedError
    
    def default_rep_predictor(self):
        raise NotImplementedError
    
    def training_set_construction(self, data_dict, value_predictor=None):
        if value_predictor is None:
            value_predictor = self.default_value_predictor()
            
        states = []
        actions = []
        rewards = []
        next_states = []
        terminations = []
        next_actions = []
        qmaxs = []
    
        for name in data_dict:
            states.append(data_dict[name]['states'])
            actions.append(data_dict[name]['actions'])
            rewards.append(data_dict[name]['rewards'])
            next_states.append(data_dict[name]['next_states'])
            terminations.append(data_dict[name]['terminations'])
            next_actions.append(np.concatenate([data_dict[name]['actions'][1:], [-1]]))  # Should not be used when using the current estimation in target construction
            if 'qmax' in data_dict[name].keys():
                qmaxs.append(data_dict[name]['qmax'])

        states = np.concatenate(states)
        actions = np.concatenate(actions)
        rewards = np.concatenate(rewards)
        next_states = np.concatenate(next_states)
        terminations = np.concatenate(terminations)
        next_actions = np.concatenate(next_actions)
        if len(qmaxs) > 0:
            qmaxs = np.concatenate(qmaxs)
        
        pred_returns = np.zeros(len(states))
        true_returns = np.zeros(len(states))
        for i in range(len(states) - 1, -1, -1):
            if i == len(states) - 1 or (not np.array_equal(next_states[i], states[i + 1])):
                pred_returns[i] = value_predictor(self.cfg.state_normalizer(states[i]))[actions[i]]
                true_pred = self.true_q_predictor(self.cfg.state_normalizer(states[i]))
                # print(self.cfg.state_normalizer(states[i]))
                true_returns[i] = 0 if true_pred is None else true_pred[actions[i]]
            else:
                end = 1.0 if terminations[i] else 0.0
                pred_returns[i] = rewards[i] + (1 - end) * self.cfg.discount * pred_returns[i + 1]
                true_returns[i] = rewards[i] + (1 - end) * self.cfg.discount * true_returns[i + 1]

        thrshd = int(len(states) * 0.8)
    
        training_s = states[: thrshd]
        training_a = actions[: thrshd]
        training_r = rewards[: thrshd]
        training_ns = next_states[: thrshd]
        training_t = terminations[: thrshd]
        training_na = next_actions[: thrshd]
        training_pred_ret = pred_returns[: thrshd]
        training_true_ret = true_returns[: thrshd]
        training_qmax = qmaxs[: thrshd]

        testing_s = states[thrshd:]
        testing_a = actions[thrshd:]
        testing_r = rewards[thrshd:]
        testing_ns = next_states[thrshd:]
        testing_t = terminations[thrshd:]
        testing_na = next_actions[thrshd:]
        testing_pred_ret = pred_returns[thrshd:]
        testing_true_ret = true_returns[thrshd:]
        testing_qmax = qmaxs[thrshd:]

        return [np.array(training_s), training_a, np.array(training_r), np.array(training_ns), np.array(training_t), training_na, training_pred_ret, training_true_ret, training_qmax], \
               [np.array(testing_s), testing_a, np.array(testing_r), np.array(testing_ns), np.array(testing_t), testing_na, testing_pred_ret, testing_true_ret, testing_qmax]

    def property_evaluation_dataset(self, data_dict):
        states = []
        actions = []
        returns = []
        for name in data_dict:
            states.append(data_dict[name]['states'])
            actions.append(data_dict[name]['actions'])
            if 'returns' in data_dict[name]:
                returns.append(data_dict[name]['returns'])
            else:
                true_returns = np.zeros(len(data_dict[name]['states']))
                rewards = data_dict[name]['rewards']
                next_states = data_dict[name]['next_states']
                terminations = data_dict[name]['terminations']
                for i in range(len(states) - 1, -1, -1):
                    if i == len(states) - 1 or (not np.array_equal(next_states[i], states[i + 1])):
                        true_pred = self.true_q_predictor(self.cfg.state_normalizer(states[i]))
                        true_returns[i] = 0 if true_pred is None else true_pred[actions[i]]
                    else:
                        end = 1.0 if terminations[i] else 0.0
                        true_returns[i] = rewards[i] + (1 - end) * self.cfg.discount * true_returns[i + 1]
                returns.append(true_returns)

        if len(states) > 0:
            states = np.concatenate(states)
            actions = np.concatenate(actions)
            returns = np.concatenate(returns)
        return [states, actions, returns]

    def draw_action_value(self):
        # test_s, _, _ = self.eval_set
        states, obstacles = self.eval_env.get_state_space()
        goal = self.eval_env.get_goal_coord()
        with torch.no_grad():
            qs = self.default_value_predictor()(torch_utils.tensor(self.cfg.state_normalizer(states), self.cfg.device))
            torch_utils.to_np(qs)

        fig, axs = plt.subplots(1, 1, figsize=(3,3))
        template = np.zeros((states[:, 0].max()+1, states[:, 1].max()+1))
        policy = np.zeros((states[:, 0].max()+1, states[:, 1].max()+1), dtype=str)
        action_list = [">", "<", "V", "A"]
        for idx, s in enumerate(states):
            template[s[0], s[1]] = qs[idx].max()
            policy[s[0], s[1]] = action_list[self.policy(s, 0)]
        img = axs.imshow(template, cmap="Blues", vmin=qs.min(), vmax=qs.max())
        axs.set_title("Action{}".format(self.eval_env.action_explaination))
        for obs in obstacles:
            axs.text(obs[1]-0.25, obs[0]+0.25, "X", color="orange")
        for idx, s in enumerate(states):
            axs.text(s[1]-0.25, s[0]+0.25, policy[s[0], s[1]], color="black")
        axs.text(goal[1]-0.25, goal[0]+0.25, "G", color="orange")
        fig.colorbar(img, ax=axs)
        
        pth = self.cfg.get_visualization_dir()
        plt.savefig("{}/action_values.png".format(pth), dpi=300, bbox_inches='tight')
        # plt.savefig("{}/action_values_{}.png".format(pth, self.total_steps), dpi=300, bbox_inches='tight')
        plt.close()
        plt.clf()

class ValueBased(Agent):
    def __init__(self, cfg):
        super(ValueBased, self).__init__(cfg)
        self.polyak = cfg.polyak
        self.rep_net = cfg.rep_fn()
        self.val_net = cfg.val_fn()

        # Creating Target Networks
        rep_net_target = cfg.rep_fn()
        rep_net_target.load_state_dict(self.rep_net.state_dict())
        val_net_target = cfg.val_fn()
        val_net_target.load_state_dict(self.val_net.state_dict())
        TargetNets = namedtuple('TargetNets', ['rep_net', 'val_net'])
        self.targets = TargetNets(rep_net=rep_net_target, val_net=val_net_target)
        if 'load_params' in self.cfg.rep_fn_config and self.cfg.rep_fn_config['load_params']:
            self.load_rep_fn(cfg.rep_fn_config['path'])
        if 'load_params' in self.cfg.val_fn_config and self.cfg.val_fn_config['load_params']:
            self.load_val_fn(cfg.val_fn_config['path'])

        params = list(self.rep_net.parameters()) + list(self.val_net.parameters())
        self.optimizer = cfg.optimizer_fn(params)

        self.vf_loss = cfg.vf_loss_fn()
        self.constr_fn = cfg.constr_fn()

    def default_value_predictor(self):
        return lambda x: self.val_net(self.rep_net(x))

    def default_rep_predictor(self):
        return lambda x: self.rep_net(x)

    def no_grad_value(self, state):
        with torch.no_grad():
            phi = self.rep_net(torch_utils.tensor(self.cfg.state_normalizer(state), self.cfg.device))
            # phi = self.rep_net(self.cfg.state_normalizer(state))
            q_values = self.val_net(phi)
        q_values = torch_utils.to_np(q_values).flatten()
        return q_values

    def policy(self, state, eps):
        if self.agent_rng.rand() < eps:
            action = self.agent_rng.randint(0, self.cfg.action_dim)
        else:
            q_values = self.no_grad_value(state)
            action = self.agent_rng.choice(np.flatnonzero(q_values == q_values.max()))
        return action

    def load_rep_fn(self, parameters_dir):
        path = os.path.join(self.cfg.data_root, parameters_dir)
        self.rep_net.load_state_dict(torch.load(path, map_location=self.cfg.device))
        self.targets.rep_net.load_state_dict(self.rep_net.state_dict())
        self.cfg.logger.info("Load rep function from {}".format(path))

    def load_val_fn(self, parameters_dir):
        path = os.path.join(self.cfg.data_root, parameters_dir)
        self.val_net.load_state_dict(torch.load(path, map_location=self.cfg.device))
        self.targets.val_net.load_state_dict(self.val_net.state_dict())
        self.cfg.logger.info("Load value function from {}".format(path))

    def save(self, early=False):
        parameters_dir = self.cfg.get_parameters_dir()
        if early:
            path = os.path.join(parameters_dir, "rep_net_earlystop")
        elif self.cfg.checkpoints:
            path = os.path.join(parameters_dir, "rep_net_{}".format(self.total_steps))
        else:
            path = os.path.join(parameters_dir, "rep_net")
        torch.save(self.rep_net.state_dict(), path)

        if early:
            path = os.path.join(parameters_dir, "val_net_earlystop")
        else:
            path = os.path.join(parameters_dir, "val_net")
        torch.save(self.val_net.state_dict(), path)

    def eval_step(self, state):
        with torch.no_grad():
            q_values = self.val_net(self.rep_net(torch_utils.tensor(self.cfg.state_normalizer(state), self.cfg.device)))
            q_values = torch_utils.to_np(q_values).flatten()
        return self.agent_rng.choice(np.flatnonzero(q_values == q_values.max()))

    def check_update(self):
        return self.cfg.rep_fn_config['train_params'] or self.cfg.val_fn_config['train_params']

    def sync_target(self):
        # self.targets.rep_net.load_state_dict(self.rep_net.state_dict())
        # self.targets.val_net.load_state_dict(self.val_net.state_dict())
        with torch.no_grad():
            for p, p_targ in zip(self.rep_net.parameters(), self.targets.rep_net.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
            for p, p_targ in zip(self.val_net.parameters(), self.targets.val_net.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

class ActorCritic(Agent):
    def __init__(self, cfg):
        super(ActorCritic, self).__init__(cfg)
        q1q2 = cfg.critic_fn()
        pi = cfg.policy_fn()
        AC = namedtuple('AC', ['q1q2', 'pi'])
        self.ac = AC(q1q2=q1q2, pi=pi)

        q1q2_target = cfg.critic_fn()
        pi_target = cfg.policy_fn()
        q1q2_target.load_state_dict(q1q2.state_dict())
        pi_target.load_state_dict(pi.state_dict())
        ACTarg = namedtuple('ACTarg', ['q1q2', 'pi'])
        self.ac_targ = ACTarg(q1q2=q1q2_target, pi=pi_target)
        self.ac_targ.q1q2.load_state_dict(self.ac.q1q2.state_dict())
        self.ac_targ.pi.load_state_dict(self.ac.pi.state_dict())

        # self.q_params = self.ac.q1q2.parameters()

        self.pi_optimizer = cfg.policy_optimizer_fn(list(self.ac.pi.parameters()))
        self.q_optimizer = cfg.critic_optimizer_fn(list(self.ac.q1q2.parameters()))
        self.polyak = cfg.polyak #0 is hard sync

        if 'load_params' in self.cfg.policy_fn_config and self.cfg.policy_fn_config['load_params']:
            self.load_actor_fn(cfg.policy_fn_config['path'])
        if 'load_params' in self.cfg.critic_fn_config and self.cfg.critic_fn_config['load_params']:
            self.load_critic_fn(cfg.critic_fn_config['path'])

    def default_value_predictor(self):
        def vp(x):
            with torch.no_grad():
                q1, q2 = self.ac.q1q2(x)
            return torch.minimum(q1, q2)
        return vp

    def default_rep_predictor(self):
        def rp(x):
            with torch.no_grad():
                rep = self.ac.pi.body(x)
            return rep
        return rp
    
    def feed_data(self):
        if self.reset is True:
            self.state = self.env.reset()
            self.reset = False
        action = self.policy(self.state, None)
        next_state, reward, done, _ = self.env.step([action])
        self.replay.feed([self.state, action, reward, next_state, int(done)])
        prev_state = self.state
        self.state = next_state
        self.update_stats(reward, done)
        return prev_state, action, reward, next_state, int(done)

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

    def load_actor_fn(self, parameters_dir):
        path = os.path.join(self.cfg.data_root, parameters_dir)
        self.ac.pi.load_state_dict(torch.load(path, map_location=self.cfg.device))
        self.ac_targ.pi.load_state_dict(self.ac.pi.state_dict())
        self.cfg.logger.info("Load actor function from {}".format(path))

    def load_critic_fn(self, parameters_dir):
        path = os.path.join(self.cfg.data_root, parameters_dir)
        self.ac.q1q2.load_state_dict(torch.load(path, map_location=self.cfg.device))
        self.ac_targ.q1q2.load_state_dict(self.ac.q1q2.state_dict())
        self.cfg.logger.info("Load critic function from {}".format(path))

    def policy(self, o, placeholder):
        o = torch_utils.tensor(self.cfg.state_normalizer(o), self.cfg.device)
        with torch.no_grad():
            a, _ = self.ac.pi(o)
        a = torch_utils.to_np(a)
        return a

    def eval_step(self, state):
        a = self.policy(state, None)
        return a

    def check_update(self):
        return self.cfg.policy_fn_config["train_params"] or self.cfg.critic_fn_config["train_params"]
    
    def compute_loss_q(self, data):
        o, a, r, op, d = data['obs'], data['act'], data['reward'], data['obs2'], data['done']

        q1, q2 = self.ac.q1q2(o)
        q1, q2 = q1[np.arange(len(a)), a], q2[np.arange(len(a)), a]

        with torch.no_grad():
            a2, logp_a2 = self.ac.pi(op)

            q1_pi_targ, q2_pi_targ = self.ac_targ.q1q2(op)
            q1_pi_targ, q2_pi_targ = q1_pi_targ[np.arange(len(a2)), a2], q2_pi_targ[np.arange(len(a2)), a2]
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ)# - self.alpha * logp_a2)

        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    def update(self, data):
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        self.pi_optimizer.zero_grad()
        loss_pi, log_prob = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
            self.sync_target()
            
        return loss_q, q_info, loss_pi, log_prob
            
    def sync_target(self):
        with torch.no_grad():
            for p, p_targ in zip(self.ac.q1q2.parameters(), self.ac_targ.q1q2.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
            for p, p_targ in zip(self.ac.pi.parameters(), self.ac_targ.pi.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def compute_loss_pi(self, data):
        raise NotImplementedError
