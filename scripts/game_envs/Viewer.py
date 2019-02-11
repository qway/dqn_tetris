import time
import pygame
import threading
import torch
import numpy as np
from scipy.misc import imresize
from torch import ByteTensor
from twitchstream.chat import TwitchChatStream
from twitchstream.outputvideo import TwitchBufferedOutputStream
from skimage import util
import dqn.lib.utils
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw


class Viewer(threading.Thread):

    def __init__(self, sim):
        super(Viewer, self).__init__()
        import game_envs.params as params
        self.sim = sim
        w = sim.board_width
        h = sim.board_height
        self.cell_size = params.PARAMETERS['tetris_viewer']['cell_size']
        self.colors = params.PARAMETERS['tetris_viewer']['colors']
        self.width = self.cell_size * (w + 6)
        self.height = self.cell_size * h
        self.rlim = self.cell_size * w
        pygame.init()
        pygame.key.set_repeat(250, 25)
        self.bground_grid = [[9 if x % 2 == y % 2 else 0 for x in range(w)] for y in
                             range(h)]
        self.default_font = pygame.font.Font(
            pygame.font.get_default_font(), 12)

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.event.set_blocked(pygame.MOUSEMOTION)  # We do not need
        rect = pygame.Rect(0, 0, self.width, self.height)
        self.sub_screen = self.screen.subsurface(rect)
        self.running = True
        self.t = threading.Thread(target=self.run)
        self.t.start()

    def disp_msg(self, msg, topleft):
        x, y = topleft
        for line in msg.splitlines():
            self.screen.blit(
                self.default_font.render(
                    line,
                    False,
                    (255, 255, 255),
                    (0, 0, 0)),
                (x, y))
            y += 14

    def center_msg(self, msg):
        for i, line in enumerate(msg.splitlines()):
            msg_image = self.default_font.render(line, False,
                                                 (255, 255, 255), (0, 0, 0))

            msgim_center_x, msgim_center_y = msg_image.get_size()
            msgim_center_x //= 2
            msgim_center_y //= 2

            self.screen.blit(msg_image, (
                self.width // 2 - msgim_center_x,
                self.height // 2 - msgim_center_y + i * 22))

    def draw_matrix(self, matrix, offset):
        off_x, off_y = offset
        for y, row in enumerate(matrix):
            for x, val in enumerate(row):
                if val:
                    pygame.draw.rect(
                        self.screen,
                        self.colors[val],
                        pygame.Rect(
                            (off_x + x) *
                            self.cell_size,
                            (off_y + y) *
                            self.cell_size,
                            self.cell_size,
                            self.cell_size), 0)

    def run(self):
        while self.running:
            self.draw()
            time.sleep(0.1)

    def draw(self):
        self.screen.fill((0, 0, 0))
        self.draw_matrix(self.bground_grid, (0, 0))
        self.draw_matrix(self.sim.board, (0, 0))
        self.draw_matrix(self.sim.tetris_shapes[self.sim.next_stone], (self.sim.board_width + 2, 2))
        if self.sim.game_over:
            self.center_msg("""Game Over!\nYour score: %d
            Press space to continue""" % self.sim.score)

        pygame.draw.line(self.screen,
                         (255, 255, 255),
                         (self.rlim + 1, 0),
                         (self.rlim + 1, self.height - 1))
        self.disp_msg("Next:", (
            self.rlim + self.cell_size,
            2))
        self.disp_msg("Score: %.2f\n\nLevel: %d\
                \nLines: %d" % (self.sim.score, self.sim.level, self.sim.lines),
                      (self.rlim + self.cell_size, self.cell_size * 5))
        pygame.display.update()


class NoneViewer:

    def __init__(self, v):
        pass


class TwitchViewer:
    def __init__(self, tetris):
        self.tetris = tetris
        self.frame_shape = (160, 284, 3)  # (480, 640, 3)
        # this value will only be sent in the beginning
        self.frame = None  # np.zeros(self.frame_shape)
        self.graph = None
        self.games = []
        self.mean_reward = []
        self.videostream = TwitchBufferedOutputStream(twitch_stream_key='INSERT_KEY_HERE',
                                        width=self.frame_shape[1],
                                        height=self.frame_shape[0],
                                        fps=30.,
                                        verbose=False,
                                        enable_audio=True)
        a, b = tetris.board_height, self.tetris.board_width
        s = np.array([a, b], dtype=np.float32)
        self.ratio = min(self.frame_shape[0] / s[0], self.frame_shape[1] / s[1])
        resized_shape = s * self.ratio
        missing_height = self.frame_shape[0] - int(resized_shape[0])
        missing_width = self.frame_shape[1] - int(resized_shape[1])
        self.padding_top = int(missing_height)
        self.padding_right = int(missing_width)
        self.last_time = time.time()
        self.oldframe = np.zeros((a,b), dtype='int64')
        self.psy = False

    def __call__(self):
        frame = np.array(self.tetris.board)
        oldframe = frame.copy()
        frame *= 2
        frame -= self.oldframe
        while True:
            if self.videostream.get_video_frame_buffer_state() < 30:
                colored_image = dqn.lib.utils.matrix2image(frame)
                colored_image[colored_image == 0] = 128
                colored_image = util.invert(colored_image)
                resized_image = imresize(colored_image, self.ratio, interp='nearest')  # does eliminate the white stones
                padded_image = np.pad(resized_image, ((self.padding_top, 0), (0, self.padding_right), (0, 0)),
                                      mode='constant',
                                      constant_values=50)
                width = self.padding_right-resized_image.shape[1]//10
                img = Image.new('RGB', (width, 160))
                draw = ImageDraw.Draw(img)
                font = ImageFont.load_default()
                draw.text((5, 5), "Score: %.2f" % self.tetris.score, (50, 50, 50), font=font)
                draw.text((5, 15), f"Moves: {self.tetris.total_moves}", (50, 50, 50), font=font)
                draw.text((5, 25), f"Lines: {self.tetris.lines}", (50, 50, 50), font=font)
                draw.text((5, 35), f"Level: {self.tetris.level}", (50, 50, 50), font=font)
                padded_image[:, -width:] = np.array(img)
                if self.psy:
                    padded_image += util.invert(np.flipud(np.fliplr(padded_image)))
                    padded_image += util.invert((np.flipud(padded_image)))
                self.videostream.send_video_frame(padded_image)
                break
            else:
                time.sleep(.001)
        self.oldframe = oldframe

        # process all the messages
        if self.tetris.level % 20 > 17:
            self.psy = True
        else:
            self.psy = False
