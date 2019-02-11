#!/usr/bin/env python3
import json

import dqn.lib.utils
import dqn
import ptan.ptan as ptan
import dqn.lib.dqn_model as dqn_model
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.utils as vutils
import numpy as np

import torch.optim as optim

from tensorboardX import SummaryWriter

# n-step
from dqn.lib import common
from game_envs.record_game import load_game
from game_envs.tetris_env_v2 import TetrisEnv

from ptan.ptan.common.wrappers import FrameStack

REWARD_STEPS = 1

# priority replay
PRIO_REPLAY_ALPHA = 0.6
BETA_START = 0.4
BETA_FRAMES = 100000

# C51
Vmax = 10
Vmin = -10
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)


class DQNConvLayer(nn.Module):
    def __init__(self, input_shape):
        super(DQNConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class ModifiedDQNConvLayer(nn.Module):
    def __init__(self, input_shape):
        super(ModifiedDQNConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
            # nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1),
            # nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class RainbowDQN(nn.Module):
    def __init__(self, input_shape, n_actions, convLayer=DQNConvLayer):
        super(RainbowDQN, self).__init__()

        self.conv = convLayer(input_shape)

        conv_out_size = self._get_conv_out(input_shape)
        self.fc_val = nn.Sequential(
            dqn_model.NoisyLinear(conv_out_size, 512),
            nn.ReLU(),
            dqn_model.NoisyLinear(512, N_ATOMS)
        )

        self.fc_adv = nn.Sequential(
            dqn_model.NoisyLinear(conv_out_size, 512),
            nn.ReLU(),
            dqn_model.NoisyLinear(512, n_actions * N_ATOMS)
        )

        self.register_buffer("supports", torch.arange(Vmin, Vmax, DELTA_Z))
        self.softmax = nn.Softmax()

    def _get_conv_out(self, shape):
        o = self.conv(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, x):
        batch_size = x.size()[0]
        fx = x.float() / 8
        conv_out = self.conv(fx).view(batch_size, -1)
        val_out = self.fc_val(conv_out).view(batch_size, 1, N_ATOMS)
        adv_out = self.fc_adv(conv_out).view(batch_size, -1, N_ATOMS)
        adv_mean = adv_out.mean(dim=1, keepdim=True)
        return val_out + adv_out - adv_mean

    def both(self, x):
        cat_out = self(x)
        probs = self.apply_softmax(cat_out)
        weights = probs * Variable(self.supports, volatile=True)
        res = weights.sum(dim=2)
        return cat_out, res

    def qvals(self, x):
        return self.both(x)[1]

    def apply_softmax(self, t):
        return self.softmax(t.view(-1, N_ATOMS)).view(t.size())


def calc_loss(batch, batch_weights, net, tgt_net, gamma, cuda=False):
    states, actions, rewards, dones, next_states = dqn.lib.common.unpack_batch(batch)

    batch_size = len(batch)

    states_v = Variable(torch.from_numpy(states))
    actions_v = Variable(torch.from_numpy(actions))
    next_states_v = Variable(torch.from_numpy(next_states))
    batch_weights_v = Variable(torch.from_numpy(batch_weights))
    if cuda:
        states_v = states_v.cuda()
        actions_v = actions_v.cuda()
        next_states_v = next_states_v.cuda()
        batch_weights_v = batch_weights_v.cuda()

    # next state distribution
    # dueling arch -- actions from main net, distr from tgt_net

    # calc at once both next and cur states
    distr_v, qvals_v = net.both(torch.cat((states_v, next_states_v)))
    next_qvals_v = qvals_v[batch_size:]
    distr_v = distr_v[:batch_size]

    next_actions_v = next_qvals_v.max(1)[1]
    next_distr_v = tgt_net(next_states_v)
    next_best_distr_v = next_distr_v[range(batch_size), next_actions_v.data]
    next_best_distr_v = tgt_net.apply_softmax(next_best_distr_v)
    next_best_distr = next_best_distr_v.data.cpu().numpy()

    dones = dones.astype(np.bool)

    # project our distribution using Bellman update
    proj_distr = dqn.lib.common.distr_projection(next_best_distr, rewards, dones, Vmin, Vmax, N_ATOMS, gamma)

    # calculate net output
    state_action_values = distr_v[range(batch_size), actions_v.data]
    state_log_sm_v = F.log_softmax(state_action_values)
    proj_distr_v = Variable(torch.from_numpy(proj_distr))
    if cuda:
        proj_distr_v = proj_distr_v.cuda()

    loss_v = -state_log_sm_v * proj_distr_v
    loss_v = batch_weights_v * loss_v.sum(dim=1)
    return loss_v.mean(), loss_v + 1e-5


if __name__ == "__main__":
    params = common.HYPERPARAMS['benetris']
    parser = argparse.ArgumentParser()

    parser.add_argument("--cuda", default=torch.cuda.is_available(), action="store_true", help="Enable cuda")
    args = parser.parse_args()

    writer = SummaryWriter(comment="-" + params['run_name'] + "-rainbow")
    writer.add_text("params1", str(params))
    env = TetrisEnv(writer)
    env = FrameStack(env, 2)

    net = RainbowDQN(env.observation_space.shape, env.action_space.n, ModifiedDQNConvLayer)  # dqn_model.PoolNet)
    if args.cuda:
        print("cuda is available")
        net.cuda()

    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    epsilon_tracker = dqn.lib.common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(lambda x: net.qvals(x), selector, cuda=args.cuda, cuda_device_id=0)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=REWARD_STEPS)
    buffer = ptan.experience.PrioritizedReplayBuffer(exp_source, params['replay_size'], PRIO_REPLAY_ALPHA)
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

    frame_idx = 0
    beta = BETA_START

    if params.get('load_prev_games', False):
        loaded_games = load_game(params['recorded_games_filename'])
        for i in range(params['load_multiplier']):
            for exp in loaded_games:
                buffer._add(exp)

    with dqn.lib.common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            frame_idx += 1
            buffer.populate(1)
            epsilon_tracker.frame(frame_idx)
            beta = min(1.0, BETA_START + frame_idx * (1.0 - BETA_START) / BETA_FRAMES)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    break

            if len(buffer) < params['replay_initial']:
                continue

            if frame_idx % 10000 == 0:
                q = vutils.make_grid(torch.cat(torch.split(next(net.parameters()).data.cpu(), 1, 1), 0), normalize=True,
                                     scale_each=True)
                writer.add_image('conv_layers/Q', q, frame_idx)
                q_target = vutils.make_grid(
                    torch.cat(torch.split(next(tgt_net.target_model.parameters()).data.cpu(), 1, 1), 0), normalize=True,
                    scale_each=True)
                writer.add_image('conv_layers/Q_target', q_target, frame_idx)
                for name, param in net.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), frame_idx)

            optimizer.zero_grad()
            batch, batch_indices, batch_weights = buffer.sample(params['batch_size'], beta)
            loss_v, sample_prios_v = calc_loss(batch, batch_weights, net, tgt_net.target_model,
                                               params['gamma'] ** REWARD_STEPS, cuda=args.cuda)
            loss_v.backward()
            optimizer.step()
            buffer.update_priorities(batch_indices, sample_prios_v.data.cpu().numpy())

            if params['target_update_rate'] >= 1:
                if frame_idx % params['target_update_rate'] == 0:
                    tgt_net.sync()
            else:
                tgt_net.alpha_sync(params['target_update_rate'])
