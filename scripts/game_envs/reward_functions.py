import threading
import time

import torch
import numpy as np
from scipy.misc import imresize
from torch import ByteTensor
from twitchstream.outputvideo import TwitchBufferedOutputStream
from skimage import util

import dqn.lib.utils


class Evaluator:
    def __init__(self, metrics):
        self.metrics = metrics
        self.total_reward = 0

    def __call__(self, img, score, dead, action=None):
        current_state_estimation = 0
        #img = img.clone()
        for weight, function in self.metrics:
            current_state_estimation += weight * function(img, score, dead, action)
        # current_state_estimation = np.clip(current_state_estimation, -1, 1)

        if dead:
            score = -1.
        elif score > 0:
            score = score
            #print("yeah a line was sent!", time.strftime("%H:%M:%S", time.gmtime()))
        else:
            score = 0.01
        self.total_reward += score
        return score


class RecorderWrapper:
    '''
    Old hack to record gifs of played games
    '''
    def __init__(self, evaluator, every_x):
        self.evaluator = evaluator
        self.playcount = 0
        self.count = 0
        self.images = []
        self.every_x_games = every_x

    def __call__(self, img, score, dead, action=None):
        self.count += 1
        board_img = img[0].clone().numpy()
        board_img[board_img > 1] = 0
        if self.playcount % self.every_x_games == 0:
            self.images.append(board_img)
        if self.playcount % self.every_x_games == 1:
            dqn.lib.utils.make_gif(self.images, 'gifs/game%03d.gif' % self.count)
            self.images = []
            self.playcount += 1
        if dead:
            self.playcount += 1
        return self.evaluator(img, score, dead, action)


class TwitchStream:
    '''
    Old hack for a twitch stream, new version is in Viewer.py
    '''
    def __init__(self, record_wrapper):
        self.record_wrapper = record_wrapper

        self.frame_shape = (160, 284, 3)  # (480, 640, 3)
        # this value will only be sent in the beginning
        self.frame = None  # np.zeros(self.frame_shape)
        self.graph = None
        self.games = []
        self.mean_reward = []
        t = threading.Thread(target=self._stream)
        t.start()

    def _stream(self):
        with TwitchBufferedOutputStream(twitch_stream_key='live_94032640_C0emaIZIK2YSd96BGIAA2yt6679Z0u',
                                        width=self.frame_shape[1],
                                        height=self.frame_shape[0],
                                        fps=30.,
                                        verbose=False,
                                        enable_audio=True) as videostream:
            while self.frame is None:
                time.sleep(0.05)

            # set the parameters for postprocessing
            a, b = self.frame.shape
            s = np.array([a, b], dtype=np.float32)
            self.ratio = min(self.frame_shape[0] / s[0], self.frame_shape[1] / s[1])
            resized_shape = s * self.ratio
            missing_height = self.frame_shape[0] - int(resized_shape[0])
            missing_width = self.frame_shape[1] - int(resized_shape[1])
            self.padding_top = int(missing_height)
            self.padding_right = int(missing_width)
            self.last_time = time.time()

            while True:
                if videostream.get_video_frame_buffer_state() < 30 and self.frame is not None:
                    self._send_image(self.frame.clone(), videostream)
                    # t = time.time()
                    # if t - self.last_time > 10:
                    #    self.games.append(self.record_wrapper.playcount)
                    #    self.mean_reward.append(
                    #        self.record_wrapper.evaluator.total_reward / self.record_wrapper.playcount)
                    # t = threading.Thread(target=self.construct_graph)
                    # t.start()
                    #    self.construct_graph()
                    #    self.last_time = t

                else:
                    time.sleep(.001)

    def _send_image(self, img, videostream):
        colored_image = img.squeeze().numpy()
        colored_image[colored_image > 1] = 0
        colored_image = dqn.lib.utils.matrix2image(colored_image)
        colored_image[colored_image == 0] = 128
        colored_image = util.invert(colored_image)
        resized_image = imresize(colored_image, self.ratio, interp='nearest')  # does eliminate the white stones
        padded_image = np.pad(resized_image, ((self.padding_top, 0), (0, self.padding_right), (0, 0)), mode='constant',
                              constant_values=50)

        videostream.send_video_frame(padded_image)

    def __call__(self, img, score, dead, action=None):
        self.frame = img[0]
        return self.record_wrapper(img, score, dead, action)

    def construct_graph(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        my_dpi = 100
        width = height = self.padding_right
        fig = plt.figure(figsize=(width / my_dpi, height / my_dpi), dpi=my_dpi)
        fig.add_subplot(111)
        plt.plot(self.games, self.mean_reward)
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        self.graph = data

"""here are different reward-functions listed"""


def process_environment_output(img, score, dead, action=None):
    return process_score(score, dead)


def simple_evaluator(img, score, dead, action=None):
    minimum_blocks_weight = 0.1
    relative_block_sum = minimum_blocks(img)

    processed_score = process_score(score, dead)


def height(img, score, dead, action=None):
    img[img > 1] = 0
    _, h, w = img.shape
    _, heights = torch.max(img, dim=1)
    heights = heights.squeeze().numpy()  # .astype(dtype=np.float32)
    heights = h - heights
    heights[heights == h] = 0
    m = np.max(heights)
    m = np.sum(range(m + 1)).__float__()
    m /= h
    return -m


"""here are different metrics listed"""


def minimum_blocks(img: ByteTensor, score, dead, action=None):
    img[img > 1] = 0  # Delete working tetronimo
    block_sum = torch.sum(img)
    relative_block_sum = 0.5 - block_sum / (img.shape[1] * img.shape[2])
    return relative_block_sum


def process_score(img, score, dead, action=None):
    if dead:
        score = -10  # will be clipped anyway but ensures that clearing lines allways good
    elif score > 0:
        score = 10  # will be clipped anyway but ensures that clearing lines allways good
        print("yeah a line was sent!", time.strftime("%H:%M:%S", time.gmtime()))
    else:
        score = 0
    return score


def bumpiness(img, score, dead, action=None):
    img[img > 1] = 0  # Delete working tetronimo
    _, h, w = img.shape
    heights = [0 for _ in range(w)]
    for i in range(w):
        for j in range(h):
            if img[0, j, i] != 0:
                heights[i] = h - j
                break
    # fix priorization of stone close to the edge
    heights = [0] + heights + [0]
    score = 0
    for i in range(w + 1):
        score += abs(heights[i] - heights[i + 1])
    score /= -(w - 1) * h
    return score


def holes(img, score, dead, action=None):
    img[img > 1] = 0  # Delete working tetronimo
    _, h, w = img.shape
    holes = 0
    for i in range(w):
        holey = False
        for j in range(h):
            if holey:
                if img[0, j, i] == 0:
                    holes += 1
            else:
                if img[0, j, i] != 0:
                    holey = True
    score = -min(2 * holes / (h * w), 1)
    return score
