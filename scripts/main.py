import argparse

import time
from tensorboardX import SummaryWriter
from torch import optim, nn
import numpy as np
from torch.autograd import Variable
import torch
from dqn.lib import common
import dqn.lib.dqn_model as models
from dqn.lib.utils import log_images_histograms
import game_envs.tetris_env_v2 as Envs
from ptan import ptan
from ptan.ptan.agent import BaseAgent
from ptan.ptan.common.wrappers import FrameStack
from dqn.tetris_dqn import StackedDQN
import torch.cuda
import torch.backends.cudnn


class TorchDQNAgent(BaseAgent):
    """
    the TorchDQNAgent enables better cuda performance
    """

    def __init__(self, dqn_model, action_selector, cuda=False, cuda_device_id=None):
        self.dqn_model = dqn_model
        self.action_selector = action_selector
        self.cuda = cuda
        self.cuda_device_id = cuda_device_id

    def __call__(self, states, agent_states=None):
        sample = np.random.sample()  # This might lead to some instabilities!
        if sample < self.action_selector.epsilon:
            rand_actions = np.random.choice(states[0].shape[0], len(states))
            return rand_actions, agent_states
        if len(states) == 1:
            states = states[0].unsqueeze(0)
        else:
            states = torch.stack(states, dim=0)
        states.pin_memory()
        self.dqn_model.eval()
        v = Variable(states, volatile=True)
        if self.cuda:
            v = v.cuda(async=True)
        actions = self.dqn_model(v).max(dim=1)[1].data.cpu().numpy()
        return actions, agent_states


class TupleDQNAgent(BaseAgent):
    """
    the TupleDQNAgent works with tuples as input for the current stone
    """

    def __init__(self, dqn_model, action_selector, cuda=False, cuda_device_id=None):
        self.dqn_model = dqn_model
        self.action_selector = action_selector
        self.cuda = cuda
        self.cuda_device_id = cuda_device_id

    def __call__(self, states, agent_states=None):
        field = torch.stack([state[0] for state in states], dim=0)
        stone = torch.stack([state[1] for state in states], dim=0)
        stone.pin_memory()
        field.pin_memory()
        self.dqn_model.eval()
        f = Variable(field, volatile=True)
        s = Variable(stone, volatile=True)
        if self.cuda:
            f = f.cuda(async=True)
            s = s.cuda(async=True)
        q_v = self.dqn_model((f, s)).data.cpu().numpy()
        actions = self.action_selector(q_v)
        return actions, agent_states


if __name__ == '__main__':
    params = common.HYPERPARAMS['tetris_simple']
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True, action="store_true", help="Enable cuda")
    parser.add_argument("--viewer", default=True, action="store_true", help="Enable Viewer")
    parser.add_argument("--twitch", default=False, action="store_true", help="Enable Twitch")
    args = parser.parse_args()
    # fix for memory leak
    # the deterministic optimizations in the cudnn core apparently explode with our architecture
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # setup step size
    # the idea is to update the net only every n-th step but with a n-times bigger batchsize -> better performance
    step_size = params['step_size']
    params['batch_size'] *= step_size

    writer = SummaryWriter(comment="-" + params['run_name'] + "tetris_dqn")
    writer.add_text("dqn/params", str(params))

    # this environment stacks the possible actions to a 3D tensor
    env = Envs.TetrisMultiEnv(writer)

    # add viewer to the environment and render the game state with pygame
    # todo also make the twitchstream work like that
    if args.viewer:
        from game_envs import Viewer

        viewer = Viewer.Viewer(env.tetris)

    # if we are using more than one environment the data we store in the experience buffer is less correlated
    envs = [env] + [type(env)() for _ in range(params['env_count'] - 1)]  # More Envs more fun

    # standard StackedDQN (input: possible action_state pairs | output: q-value vector)
    net = StackedDQN((1, env.observation_space.shape[1], env.observation_space.shape[2]), models.LineNet)
    #net.load_state_dict(torch.load('models/model_parameters.pkl')) # load existing model
    # the target net copies the values of the main net every n-th step (if the target_update_rate is less then 1
    #  an alpha update is performed)
    tgt_net = ptan.agent.TargetNet(net)

    if args.cuda:
        print('cuda is available!')
        net.cuda()
        tgt_net.target_model.cuda()
    else:
        print('cuda is not available!')

    # setup the action-selector, the tracker that changes the epsilon and the agent
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = TorchDQNAgent(net, selector, cuda=args.cuda)

    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=params['gamma'], steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params['replay_size'])
    optimizer = optim.RMSprop(net.parameters(), lr=params['learning_rate'], alpha=params['r_prop_alpha'],
                              eps=params['r_prop_eps'])

    frame_idx = 0

    # the reward tracker is used for logging and also functions as break condition
    with common.RewardTracker(writer, params['stop_reward'], params['save_model_reward']) as reward_tracker:
        while True:
            # in every step do step_size steps in the environments
            frame_idx += step_size
            buffer.populate(step_size)
            epsilon_tracker.frame(frame_idx)

            # check break condition -> solved
            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    break

            # fill the replay buffer initially with data to avoid correlation
            if len(buffer) < params['replay_initial']:
                continue

            # draw batch, calculate loss and backwards pass
            optimizer.zero_grad()
            batch = buffer.sample(params['batch_size'])
            loss_v = common.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=params['gamma'], cuda=args.cuda,
                                          cuda_async=True)
            loss_v.backward()
            optimizer.step()

            # log more information
            if frame_idx % params['log_net_every'] == 0:
                log_images_histograms(net, tgt_net, writer, frame_idx)
                reward_tracker.save_model(net, frame_idx)

            # update the target net
            if params['target_update_rate'] >= 1:
                if frame_idx % params['target_update_rate'] < step_size:
                    tgt_net.sync()
            else:
                tgt_net.alpha_sync(params['target_update_rate'])

    # End Viewer
    if args.viewer:
        viewer.running = False
        viewer.join()
