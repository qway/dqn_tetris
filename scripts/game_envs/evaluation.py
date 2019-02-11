import threading
import time
import torch
from tensorboardX import SummaryWriter
from torch.autograd import Variable

from dqn.tetris_dqn import StackedDQN
from game_envs import params
import game_envs.Viewer as Viewer
from game_envs.reward_functions import TwitchStream, Evaluator
from game_envs.tetris_env_v2 import TetrisMultiEnv
from ptan import ptan
from ptan.ptan.actions import ActionSelector
import dqn.lib.dqn_model as models


class Game_Evaluator:
    def __init__(self, selector: ActionSelector, model=None, modelfilename=None, modelless=False):
        params.PARAMETERS['tetris_env']['max_play_length'] = 2000
        self.env = TetrisMultiEnv(SummaryWriter(comment='-evaluation'), evaluation=True)
        self.modelless = modelless
        if model is None:
            if modelfilename is None:
                raise ValueError('provide a model or at least a path to a model')
            if modelless:
                self.model = StackedDQN((1, self.env.observation_space.shape[1], self.env.observation_space.shape[2]),
                                        models.LineNet)
                self.model.load_state_dict(torch.load(modelfilename))
            else:
                self.model = torch.load(modelfilename)
        else:
            self.model = model
        self.selector = selector

    def __call__(self, number_of_games):
        total_reward = 0.0
        for i in range(number_of_games):
            dead = False
            states = self.env.reset()
            states = states.unsqueeze(0)
            while not dead:
                v = Variable(states)
                if torch.cuda.is_available() and not self.modelless:
                    v = v.cuda()
                q_v = self.model(v)
                q = q_v.data.cpu().numpy()
                action = self.selector(q)
                states = self.env.step(action[0])
                total_reward += states[1]
                dead = states[2]
                states = states[0]
                states = states.unsqueeze(0)
        print(f'mean reward: {total_reward/number_of_games:.4f}')

    def stream(self):
        params.PARAMETERS['tetris_env']['reward_function'] = Evaluator([])
        twitch = Viewer.TwitchViewer(self.env.tetris)
        while True:
            dead = False
            states = self.env.reset()
            states = states.unsqueeze(0)
            while not dead:
                start = time.time()
                v = Variable(states)
                if torch.cuda.is_available() and not self.modelless:
                    v = v.cuda()
                q_v = self.model(v)
                q = q_v.data.cpu().numpy()
                action = self.selector(q)
                states = self.env.step(action[0])
                twitch()
                dead = states[2]
                states = states[0]
                states = states.unsqueeze(0)
                time.sleep(max(1. / 7 - (time.time() - start), 0))

    def start_stop_viewer(self, env):
        show_viewer = False
        viewer = None
        while True:
            cmd = input("toggle viewer with v\n")
            if cmd == "v":
                if show_viewer:
                    viewer.running = False
                    viewer.t.join()
                    show_viewer = False
                else:
                    viewer = Viewer.Viewer(env)
                    show_viewer = True

    def save_state_dict(self):
        torch.save(self.model.cpu().state_dict(), '../models/model_parameters.pkl')


if __name__ == '__main__':
    e = Game_Evaluator(ptan.actions.EpsilonGreedyActionSelector(epsilon=0),
                       modelfilename="../models/model_parameters.pkl", modelless=True)
    # e(10)
    # t = threading.Thread(target=e.start_stop_viewer, args=(e.env.tetris,))
    # t.start()
    e.stream()
    #e.save_state_dict()
