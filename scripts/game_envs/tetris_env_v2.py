import gym
import time

import torch

import game_envs.InstaTetris as Insta
from gym import spaces
from tensorboardX import SummaryWriter
import numpy as np
from typing import Union
from dqn.lib.utils import make_gif
from game_envs import params


class TetrisEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, writer: Union[SummaryWriter, None] = None, evaluation=False):
        self.params = params.PARAMETERS['tetris_env']
        self.observation_space = spaces.Box(0, self.params['number_of_shapes'],
                                            (1, params.PARAMETERS['tetris']['rows'],
                                             params.PARAMETERS['tetris']['cols']))
        self.action_space = spaces.Discrete(self.params['number_of_actions'])
        self.max_play_length = self.params['max_play_length']
        self.tetris = self.params['simulator']()
        self.steps = 0
        self.rewards = []
        self.reward_sum = 0
        self.current_game_length = 0
        self.action_history = []
        self.add_writer(writer)
        self.evaluation = evaluation
        self.game_count = 0

    def _step(self, action):
        self.steps += 1
        self.current_game_length += 1
        if type(action) is not int:
            NotImplementedError("wrong action type: " + str(type(action)))
        self.action_history.append(action)
        game_state = self.tetris.step(action)
        reward = self.params['reward_function'](*game_state)
        self.reward_sum += reward
        return game_state[0], reward, game_state[2], {'action': game_state[
            -1]}  # this assumes that the last value in games_state is action; is needed because the different tetris environments provide different informations

    def _reset(self):
        if len(self.action_history) is not 0:
            self.rewards.append(self.reward_sum)
            self._log_game_data()

        self.reward_sum = 0
        self.current_game_length = 0
        self.action_history = []
        self.tetris.game_over = True
        self.last_score = 0
        self.game_count += 1
        return self.tetris.start_game()[0]

    def add_writer(self, writer: Union[SummaryWriter, None] = None):
        if writer is not None:
            writer.add_text("environment/params", str(self.params))
            self.tensorboard_writer = writer
        else:
            self.tensorboard_writer = None

    def _log_game_data(self):
        if self.tensorboard_writer is not None:
            x_axis = self.game_count if self.evaluation else self.steps
            self.tensorboard_writer.add_scalar('tetris/reward_sum', self.reward_sum, x_axis)
            self.tensorboard_writer.add_scalar('tetris/game_length', self.current_game_length, x_axis)
            # todo some thing is off here...i guess the bins variable needs to be adjusted
            self.tensorboard_writer.add_histogram('tetris/actions', np.array(self.action_history), x_axis,
                                                  bins=5)


class TetrisMultiEnv(TetrisEnv):
    def __init__(self, writer: Union[SummaryWriter, None] = None, evaluation=False):
        super(TetrisMultiEnv, self).__init__(writer, evaluation)
        self.observation_space = self.get_observation_space()
        self.action_space = self.get_action_space()

    def _step(self, action):
        data = super(TetrisMultiEnv, self)._step(action)
        a, h, w = data[0].shape
        temp = torch.zeros(self.observation_space.shape[0] - a, h, w)
        temp = torch.cat((data[0], temp))
        # temp[0:data[0].shape[0]] = data[0]
        done = data[2]
        # set done to true if max game length reached
        if self.current_game_length >= self.max_play_length:
            done = True
        return temp, data[1], done, data[3]

    def _reset(self):
        data = super(TetrisMultiEnv, self)._reset()
        temp = torch.zeros((self.observation_space.shape))
        for idx, dat in enumerate(data):
            temp[idx] = dat
        return temp

    @staticmethod
    def _calc_max_actions():
        p = params.PARAMETERS
        w = p['tetris']['cols']
        orientations = Insta.generate_orientations(p['tetris']['tetris_shapes'])
        max_actions = max([
            sum([w + 1 - len(y) for y in x])
            for x in orientations])
        return max_actions

    @staticmethod
    def get_observation_space():
        p = params.PARAMETERS
        return spaces.Box(0, p['tetris_env']['number_of_shapes'],
                          (TetrisMultiEnv._calc_max_actions(), p['tetris']['rows'],
                           p['tetris']['cols']))

    @staticmethod
    def get_action_space():
        return spaces.Discrete(TetrisMultiEnv._calc_max_actions())


class LinearEnv(TetrisEnv):
    def __init__(self, writer: Union[SummaryWriter, None] = None):
        super(LinearEnv, self).__init__(writer)
        self.observation_space = self.get_observation_space()
        self.action_space = self.get_action_space()
        self.tetris = Insta.VectorTetris()

    def _step(self, action):
        data = super(LinearEnv, self)._step(action)
        done = data[2]
        # set done to true if max game length reached
        if self.current_game_length >= self.max_play_length:
            done = True
        return data[0], data[1], done, data[3]

    def _reset(self):
        data = super(LinearEnv, self)._reset()
        return data

    @staticmethod
    def _calc_max_actions():
        return params.PARAMETERS['tetris']['cols'] * 4

    @staticmethod
    def get_observation_space():
        shapes = len(params.PARAMETERS['tetris']['tetris_shapes'])
        return spaces.Discrete(params.PARAMETERS['tetris']['cols'] + shapes + 1)

    @staticmethod
    def get_action_space():
        return spaces.Discrete(LinearEnv._calc_max_actions())


class ChooseEnv(LinearEnv):
    def __init__(self, writer: Union[SummaryWriter, None] = None):
        super(ChooseEnv, self).__init__(writer)
        self.tetris = Insta.ChooseTetris()

    @staticmethod
    def get_observation_space():
        p = params.PARAMETERS
        shapes = len(params.PARAMETERS['tetris']['tetris_shapes'])
        box = spaces.Box(0, p['tetris_env']['number_of_shapes'],
                         (TetrisMultiEnv._calc_max_actions(), p['tetris']['rows'],
                          p['tetris']['cols']))
        stones = spaces.Discrete(shapes)
        return spaces.Tuple((box, stones))


def fps_test(t):
    params.PARAMETERS['tetris_env']['simulator'] = Insta.Tetris
    env = TetrisEnv(SummaryWriter('trash'))
    env.reset()

    start = time.time()
    fps = 0
    games = 0
    seconds = t
    while time.time() - start < seconds:
        while not env.step(3)[2]:
            fps += 1
        env.reset()
        games += 1
    print(games, fps // seconds)
    env.close()


def human_test():
    params.PARAMETERS['tetris_env']['simulator'] = tet.TetrisPlayer
    env = TetrisEnv(SummaryWriter('trash'))
    env.reset()

    images = []

    while True:
        img, score, dead, action = env.step(0)
        images.append(img.squeeze().numpy())
        print(score)
        if dead:
            break

    make_gif(images, "./fuu.gif", 10)

    env.close()


if __name__ == '__main__':
    # human_test()
    fps_test(10)
