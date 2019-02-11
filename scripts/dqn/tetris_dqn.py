import argparse

from tensorboardX import SummaryWriter
from torch import optim, nn
import numpy as np
from torch.autograd import Variable
import torch
from dqn.lib import common
import dqn.lib.dqn_model as models
from dqn.lib.utils import log_images_histograms
from game_envs.tetris_env_v2 import TetrisMultiEnv
from ptan import ptan
from ptan.ptan.common.wrappers import FrameStack
import torch.multiprocessing as mp


class StackedDQN(nn.Module):
    def __init__(self, input_shape, convnet):
        super(StackedDQN, self).__init__()

        self.conv = convnet(input_shape)

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 2
        b, c, h, w = x.shape
        f = x.view((b * c, 1, h, w))
        conv_out = self.conv(f).view(b, c, -1)
        q_values = self.fc(conv_out)
        '''
        q_values = Variable(torch.FloatTensor(b, c)).cuda()
        for i in range(c):
            f = fx[:, i]
            conv_out = self.conv(f.unsqueeze(1)).view(b, -1)
            q = self.fc(conv_out).squeeze(1)
            q_values[:, i] = q
        '''
        return q_values.squeeze(2)





def old_main():
    params = common.HYPERPARAMS['tetris_simple']
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True, action="store_true", help="Enable cuda")
    args = parser.parse_args()

    # setup step size
    step_size = params['step_size']
    params['batch_size'] *= step_size

    writer = SummaryWriter(comment="-" + params['run_name'] + "tetris_dqn")
    writer.add_text("dqn/params", str(params))
    env = TetrisMultiEnv(writer)
    env = FrameStack(env, 1)
    net = StackedDQN((1, env.observation_space.shape[1], env.observation_space.shape[2]), models.LineNet)

    if args.cuda:
        print('cuda is available!')
        net.cuda()
    else:
        print('cuda is not available!')

    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, cuda=args.cuda)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params['replay_size'])
    # optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])
    optimizer = optim.RMSprop(net.parameters(), lr=params['learning_rate'], alpha=params['r_prop_alpha'],
                              eps=params['r_prop_eps'])

    frame_idx = 0

    with common.RewardTracker(writer, params['stop_reward'], params['save_model_reward']) as reward_tracker:
        while True:
            frame_idx += step_size
            buffer.populate(step_size)
            epsilon_tracker.frame(frame_idx)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    break

            if len(buffer) < params['replay_initial']:
                continue

            optimizer.zero_grad()
            batch = buffer.sample(params['batch_size'])
            loss_v = common.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=params['gamma'], cuda=args.cuda)
            loss_v.backward()
            optimizer.step()

            if frame_idx % params['log_net_every'] == 0:
                log_images_histograms(net, tgt_net, writer, frame_idx)
                reward_tracker.save_model(net, frame_idx)

            if params['target_update_rate'] >= 1:
                if frame_idx % params['target_update_rate'] < step_size:
                    tgt_net.sync()
            else:
                tgt_net.alpha_sync(params['target_update_rate'])


def new_main():
    mp.set_start_method('spawn')
    params = common.HYPERPARAMS['tetris_simple']
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True, action="store_true", help="Enable cuda")
    args = parser.parse_args()

    # setup step size
    step_size = params['step_size']
    params['batch_size'] *= step_size

    # be aware that changing the second parameter in framestack will break this line
    # todo
    net = StackedDQN(
        (1, TetrisMultiEnv.get_observation_space().shape[1], TetrisMultiEnv.get_observation_space().shape[2]),
        models.LineNet)

    if args.cuda:
        print('cuda is available!')
        net.cuda()
    else:
        print('cuda is not available!')

    tgt_net = ptan.agent.TargetNet(net)

    buffer = ptan.experience.ExperienceReplayBuffer(experience_source=None, buffer_size=params['replay_size'])
    optimizer = optim.RMSprop(net.parameters(), lr=params['learning_rate'], alpha=params['r_prop_alpha'],
                              eps=params['r_prop_eps'])

    step_size = params['step_size']

    exp_queue = mp.Queue(maxsize=step_size * 2)
    play_proc = mp.Process(target=play_func, args=(params, net, tgt_net, args.cuda, exp_queue))
    play_proc.start()

    frame_idx = 0

    while play_proc.is_alive():
        frame_idx += step_size
        for _ in range(step_size):
            exp = exp_queue.get()
            if exp is None:
                play_proc.join()
                break
            buffer._add(exp)

        if len(buffer) < params['replay_initial']:
            continue

        optimizer.zero_grad()
        batch = buffer.sample(params['batch_size'])
        loss_v = common.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=params['gamma'], cuda=args.cuda)
        loss_v.backward()
        optimizer.step()

        if params['target_update_rate'] >= 1:
            if frame_idx % params['target_update_rate'] < step_size:
                tgt_net.sync()
        else:
            tgt_net.alpha_sync(params['target_update_rate'])


if __name__ == '__main__':
    # old_main()
    new_main()
