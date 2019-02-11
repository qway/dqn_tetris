from random import randrange as rand

import time
import torch
import sys
from game_envs import params
import game_envs.Viewer as V
import numpy as np


def rotate_clockwise(shape):
    return [[shape[y][x]
             for y in range(len(shape))]
            for x in range(len(shape[0]) - 1, -1, -1)]


def generate_orientations(x):
    orientations = []
    for i in x:
        shapes = []
        shapes.append(i)
        for _ in range(4):
            i = rotate_clockwise(i)
            if i not in shapes:
                shapes.append(i)
        orientations.append(shapes)
    return orientations


class Tetris(object):
    def __init__(self):
        super(Tetris, self).__init__()
        p = params.PARAMETERS['tetris']
        self.tetris_shapes = p['tetris_shapes']
        self.tetris_orientations = generate_orientations(self.tetris_shapes)
        self.board_width = p['cols']
        self.board_height = p['rows']
        self.board = [[]]
        self.init_game()
        self.max_lines = max([max(len(x), len(x[0])) for x in self.tetris_shapes])
        self.total_moves = 0

    def init_game(self):
        self.board = [[0 for x in range(self.board_width)]
                      for y in range(self.board_height)]
        self.possible_states = []
        self.score = 0
        self.lines = 0
        self.level = 0
        self.next_stone = self.tetris_shapes[rand(len(self.tetris_shapes))]
        self.new_stone()
        self.field = torch.zeros((self.board_width, self.board_height)).byte()
        self.game_over = False
        self.total_moves = 0


    def start_game(self):
        if self.game_over:
            self.init_game()
            return self.step(-1)

    def new_stone(self):
        self.next_stone = rand(len(self.tetris_shapes))

    def check_collision(self, shape, offset):
        off_x, off_y = offset
        for cy, row in enumerate(shape):
            for cx, cell in enumerate(row):
                try:
                    if cell and self.board[cy + off_y][cx + off_x]:
                        return True
                except IndexError:
                    return True
        return False

    def add_cl_lines(self, n):
        self.lines += n
        max_height = self.get_height() + 1  # avoid dividing by zero
        self.score += n / max_height
        # self.score += 0 if n == 0 else ((n - 1) / self.max_lines + 1.0) / 2.0  # move it in the interval [0.5,1.0]
        if self.lines >= self.level * 6:
            self.level += 1

    def get_height(self):

        heights = []
        for i in range(self.board_width):
            heights += [0]
            for j in range(self.board_height):
                if self.board[j][i] != 0:
                    heights[i] = self.board_height - j
                    break
        return np.max(heights)

    def remove_row(self, row):
        del self.board[row]
        self.board = [[0 for i in range(self.board_width)]] + self.board

    def clear_rows(self):
        cleared_rows = 0
        while True:
            for i, row in enumerate(self.board):
                if 0 not in row:
                    self.remove_row(i)
                    cleared_rows += 1
                    break
            else:
                break
        self.add_cl_lines(cleared_rows)

    def join_matrixes(self, mat2, mat2_off):
        off_x, off_y = mat2_off
        for cy, row in enumerate(mat2):
            for cx, val in enumerate(row):
                self.board[cy + off_y - 1][cx + off_x] += 1 if val > 0 else 0

    def generate_states(self):
        states = []
        max_col = [self.board_height for _ in range(self.board_width)]
        for i, row in enumerate(self.board):
            for j, cols in enumerate(row):
                if cols != 0 and max_col[j] == self.board_height:
                    max_col[j] = i

        for stone in self.tetris_orientations[self.next_stone]:
            width = len(stone[0])
            height = len(stone)
            for stone_x in range(self.board_width - width + 1):
                stone_y = max(min(max_col[stone_x:stone_x + width]) - height, 0)
                if self.check_collision(stone, (stone_x, stone_y)):
                    continue
                while True:
                    stone_y += 1
                    if self.check_collision(stone, (stone_x, stone_y)):
                        break
                states.append((stone, (stone_x, stone_y)))
        return states

    def step(self, action):
        self.total_moves += 1
        self.last_score = self.score
        if action != -1:
            if action >= len(self.possible_states):
                action = rand(0, len(self.possible_states))
            self.join_matrixes(*self.possible_states[action])
            self.clear_rows()
        self.new_stone()
        self.possible_states = self.generate_states()
        if len(self.possible_states) == 0:
            self.game_over = True
        field = self.create_arr()
        return field, self.score - self.last_score, self.game_over

    def create_arr(self):
        field = torch.FloatTensor(self.board).unsqueeze(0)
        if len(self.possible_states) == 0:
            return field
        field = field.repeat(len(self.possible_states), 1, 1)
        for i, tup in enumerate(self.possible_states):
            stone, (x, y) = tup
            for cy, row in enumerate(stone):
                for cx, val in enumerate(row):
                    if y + cy > 0 and val > 0:
                        field[i][cy + y - 1][cx + x] = 2
        return field

    def quit(self):
        sys.exit()


class VectorTetris(Tetris):
    def __init__(self):
        super(VectorTetris, self).__init__()

    def create_arr(self):
        max_col = [self.board_height for _ in range(self.board_width)]
        for i, row in enumerate(self.board):
            for j, cols in enumerate(row):
                if cols != 0 and max_col[j] == self.board_height:
                    max_col[j] = i
        field = torch.ByteTensor(max_col)
        field /= self.board_height
        stone = torch.zeros(len(self.tetris_shapes)).byte()
        stone[self.next_stone] = 1
        states = torch.ByteTensor([len(self.possible_states)])
        return torch.cat((field, stone, states)).unsqueeze(0)


class ChooseTetris(Tetris):
    def __init__(self):
        super(ChooseTetris, self).__init__()

    def create_arr(self):
        field = torch.ByteTensor(self.board)
        stone = torch.zeros(len(self.tetris_shapes)).byte()
        stone[self.next_stone] = 1
        return field, stone


if __name__ == '__main__':
    t = Tetris()
    viewer = V.Viewer(t)
    viewer.start()
    t.init_game()

    state, score, dead = t.step(-1)
    start = time.time()
    frames = 0
    games = 0
    seconds = 10
    while time.time() - start < seconds:
        state, score, dead = t.step(0)
        if dead:
            t.start_game()
            games += 1
        frames += 1

    print('Games:', games)
    print('FPS:', frames // seconds)
    print('Frames', frames)
    viewer.running = False
    viewer.join()
