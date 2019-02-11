import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

HYPERPARAMS = {
    'tetris': {
        'env_name': "tetris",
        'stop_reward': 40.0,
        'run_name': 'tetris',
        'replay_size': 500000,
        'replay_initial': 250000,
        'target_update_rate': 0.00001,
        'epsilon_frames': 250000,
        'epsilon_start': 1.0,
        'epsilon_final': 0.02,
        'learning_rate': 0.0001,
        'gamma': 0.99,
        'batch_size': 32,
        'load_prev_games': True,
        'recorded_games_filename': 'game_multipleGames.txt.gz',
        'load_multiplier': 25,
        'log_net_every': 10000,
        'r_prop_alpha': 0.95,
        'r_prop_eps': 0.01,
        'step_size': 4,
        'env_count': 4,
        'save_model_reward': 35.0
    },
    'tetris_simple_working_in_42_minutes_wo_alpha_target_update': {
        'env_name': "tetris",
        'stop_reward': 22.0,
        'run_name': 'tetris',
        'replay_size': 50000,
        'replay_initial': 32,
        'target_net_sync': 10000,
        'epsilon_frames': 50000,
        'epsilon_start': 1.0,
        'epsilon_final': 0.02,
        'learning_rate': 0.0001,
        'gamma': 0.99,
        'batch_size': 32
    },
    'tetris_simple': {
        'env_name': "tetris",
        'stop_reward': 500.0,
        'run_name': 'tetris',
        'replay_size': 200000,  # maybe 20000?
        'replay_initial': 50000,
        'epsilon_frames': 100000,
        'epsilon_start': 1.0,
        'epsilon_final': 0.02,
        'learning_rate': 0.0001,
        'target_update_rate': 10000,
        'gamma': 0.99,
        'batch_size': 32,
        'load_prev_games': True,
        'recorded_games_filename': 'game_multipleGames.txt.gz',
        'load_multiplier': 25,
        'log_net_every': 10000,
        'r_prop_alpha': 0.95,
        'r_prop_eps': 0.01,
        'step_size': 4,
        'env_count': 4,
        'save_model_reward': 6.0
    },
}


def unpack_batch(batch):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(state)  # the result will be masked anyway
        else:
            last_states.append(np.array(exp.last_state, copy=False))
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), np.array(last_states, copy=False)


def collate_tuple(batch, cuda=False, cuda_async=False): # In most cases you should use the normal collate bellow!
    b_len = len(batch)
    biter = iter(batch)
    exp = next(biter)
    biter = iter(batch)
    fields = torch.ByteTensor(b_len, *exp.state[0].shape)
    stones = torch.ByteTensor(b_len, *exp.state[1].shape)
    actions = torch.LongTensor(b_len)
    rewards = torch.FloatTensor(b_len)
    dones = torch.FloatTensor(b_len)
    last_fields = torch.ByteTensor(b_len, *exp.state[0].shape)
    last_stones = torch.ByteTensor(b_len, *exp.state[1].shape)
    fields = fields.pin_memory()
    stones =stones.pin_memory()
    actions = actions.pin_memory()
    rewards = rewards.pin_memory()
    dones = dones.pin_memory()
    last_fields = last_fields.pin_memory()
    last_stones = last_stones.pin_memory()
    for i in range(b_len):
        fields[i] = exp.state[0]
        stones[i] = exp.state[1]
        actions[i] = int(exp.action)
        rewards[i] = exp.reward
        if exp.last_state is None:
            dones[i] = 0
            last_fields[i] = exp.state[0]
            last_stones[i] = exp.state[1]
        else:
            dones[i] = 1
            last_fields[i] = exp.last_state[0]
            last_stones[i] = exp.last_state[1]
        exp = next(biter)
    fields = Variable(fields)
    stones = Variable(stones)
    last_fields = Variable(last_fields, requires_grad=False, volatile=True)
    last_stones = Variable(last_stones, requires_grad=False, volatile=True)
    actions = Variable(actions, requires_grad=False)
    rewards = Variable(rewards, requires_grad=False, volatile=True)
    dones = Variable(dones, requires_grad=False, volatile=True)
    if cuda:
        fields = fields.cuda(async=cuda_async)
        stones = stones.cuda(async=cuda_async)
        last_fields = last_fields.cuda(async=cuda_async)
        last_stones = last_stones.cuda(async=cuda_async)
        actions = actions.cuda(async=cuda_async)
        rewards = rewards.cuda(async=cuda_async)
        dones = dones.cuda(async=cuda_async)
    return (fields, stones), actions, rewards, dones, (last_fields, last_stones)


def collate(batch, cuda=False, cuda_async=False):
    b_len = len(batch)
    biter = iter(batch)
    exp = next(biter)
    biter = iter(batch)
    states = torch.FloatTensor(b_len, *exp.state.shape)
    actions = torch.LongTensor(b_len)
    rewards = torch.FloatTensor(b_len)
    dones = torch.FloatTensor(b_len)
    last_states = torch.FloatTensor(b_len, *exp.state.shape)
    states = states.pin_memory()
    actions = actions.pin_memory()
    rewards = rewards.pin_memory()
    dones = dones.pin_memory()
    last_states = last_states.pin_memory()
    for i in range(b_len):
        states[i] = exp.state
        actions[i] = int(exp.action)
        rewards[i] = exp.reward
        if exp.last_state is None:
            dones[i] = 0
            last_states[i] = exp.state
        else:
            dones[i] = 1
            last_states[i] = exp.last_state
        exp = next(biter)
    states = Variable(states)
    last_states = Variable(last_states, requires_grad=False)
    actions = Variable(actions, requires_grad=False)
    rewards = Variable(rewards, requires_grad=False)
    dones = Variable(dones, requires_grad=False)
    if cuda:
        states = states.cuda(async=cuda_async)
        last_states = last_states.cuda(async=cuda_async)
        actions = actions.cuda(async=cuda_async)
        rewards = rewards.cuda(async=cuda_async)
        dones = dones.cuda(async=cuda_async)
    return states, actions, rewards, dones, last_states


def calc_loss_dqn(batch, net, tgt_net, gamma, cuda=False, cuda_async=False):
    states_v, actions_v, rewards_v, not_done_mask, next_states_v = collate(batch, cuda, cuda_async)


    net.train()
    tgt_net.eval()
    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    # todo: use this? this snippit is from the rainbow implementation
    # distr_v, qvals_v = net.both(torch.cat((states_v, next_states_v)))

    # 1.1 get predicted actions for next state from main net
    next_state_selected_actions = net(next_states_v).max(dim=1, keepdim=True)[1]
    # 1.2 get q values for this action using the target net
    next_state_Q_values = tgt_net(next_states_v)
    # 1.3 calculate double_q values by combining both
    double_q = next_state_Q_values.gather(1, next_state_selected_actions)
    # 1.4 just apply the bellman equation for the target values
    target_q = rewards_v + gamma * double_q.squeeze(-1) * not_done_mask

    # next_state_values = tgt_net(next_states_v).max(1)[0]
    # next_state_values[done_mask] = 0.0
    # next_state_values.volatile = False

    # expected_state_action_values = next_state_values * gamma + rewards_v
    # todo clip loss to -1 and 1
    return nn.functional.mse_loss(state_action_values, target_q)  # expected_state_action_values)


class RewardTracker:
    def __init__(self, writer, stop_reward, start_saving_reward=0.0):
        self.writer = writer
        self.stop_reward = stop_reward
        self.start_saving_reward = start_saving_reward
        self.current_mean_reward = -1.0
        self.last_saved_mean_reward = -1.0

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, frame, epsilon=None):
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        self.current_mean_reward = np.mean(self.total_rewards[-100:])
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: done %d games, mean reward %.3f, speed %.2f f/s%s" % (
            frame, len(self.total_rewards), self.current_mean_reward, speed, epsilon_str
        ))
        sys.stdout.flush()
        self.writer.add_scalar("speed", speed, frame)
        self.writer.add_scalar("reward_100", self.current_mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)
        if self.current_mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True
        return False

    def save_model(self, net, frame, net_name="tetris_net"):
        if self.current_mean_reward > self.start_saving_reward \
                and self.current_mean_reward > self.last_saved_mean_reward:
            torch.save(net, f'./models/{net_name}_{frame}_{self.current_mean_reward:.2f}.pkl')
            self.last_saved_mean_reward = self.current_mean_reward
            print('###################################')
            print('#saved model with {} mean reward#'.format(self.last_saved_mean_reward))
            print('###################################')


class EpsilonTracker:
    def __init__(self, epsilon_greedy_selector, params):
        self.epsilon_greedy_selector = epsilon_greedy_selector
        self.epsilon_start = params['epsilon_start']
        self.epsilon_final = params['epsilon_final']
        self.initial_frames = params['replay_initial']
        self.epsilon_frames = params['epsilon_frames']
        self.frame(0)

    def frame(self, frame):
        self.epsilon_greedy_selector.epsilon = min(1, max(self.epsilon_final, (
                (self.epsilon_final - self.epsilon_start) / self.epsilon_frames) * (frame - self.initial_frames) + 1))
        # if frame < self.initial_frames:
        #    self.epsilon_greedy_selector.epsilon = self.epsilon_start
        # else:
        #    self.epsilon_greedy_selector.epsilon = \
        #        max(self.epsilon_final, self.epsilon_start - frame / self.epsilon_frames)


def distr_projection(next_distr, rewards, dones, Vmin, Vmax, n_atoms, gamma):
    """
    Perform distribution projection aka Catergorical Algorithm from the
    "A Distributional Perspective on RL" paper
    """
    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, n_atoms), dtype=np.float32)
    delta_z = (Vmax - Vmin) / (n_atoms - 1)
    for atom in range(n_atoms):
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards + (Vmin + atom * delta_z) * gamma))
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]
    if dones.any():
        proj_distr[dones] = 0.0
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards[dones]))
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        eq_dones = dones.copy()
        eq_dones[dones] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l] = 1.0
        ne_mask = u != l
        ne_dones = dones.copy()
        ne_dones[dones] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l] = (u - b_j)[ne_mask]
            proj_distr[ne_dones, u] = (b_j - l)[ne_mask]
    return proj_distr
