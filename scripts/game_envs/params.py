# import game_envs.tetris_v2 as tet
import game_envs.InstaTetris as insta
from game_envs import Viewer
from game_envs.reward_functions import *

PARAMETERS = {
    'tetris_viewer': {
        'cell_size': 18,
        'colors': [
            (0, 0, 0),
            (255, 255, 255),
            (100, 200, 115),
            (120, 108, 245),
            (255, 140, 50),
            (50, 120, 52),
            (146, 202, 73),
            (150, 161, 218),
            (255, 85, 85),
            (30, 30, 30),
            (0, 0, 0),# Helper color for background grid
        ]
    },
    'tetris': {
        'cols': 10, #10,
        'rows': 22,#22,
        'tetris_shapes-f': [
            [[2, 2],
             [2, 0]],
            [[2, 2]],
            [[2, 2, 2]],
            [[2, 2],
             [2, 2]]
        ],
        'tetris_shapes_false2': [
            [[2, 2],
             [2, 0]],
            [[2, 2]],
        ],
        'tetris_shapes': [
            [[2, 2, 2],
             [0, 2, 0]],

            [[0, 2, 2],
             [2, 2, 0]],

            [[2, 2, 0],
             [0, 2, 2]],

            [[2, 0, 0],
             [2, 2, 2]],

            [[0, 0, 2],
             [2, 2, 2]],

            [[2, 2, 2, 2]],

            [[2, 2],
             [2, 2]]
        ],
        'LEFT': 0,
        'RIGHT': 1,
        'DOWN': 2,
        'UP': 3,
        'RETURN': 4,
        'NOTHING': 5
    },
    'tetris_env': {
        'number_of_shapes': None,
        'number_of_actions': 4,
        'simulator': insta.Tetris,
        'reward_function': Evaluator([]),
        'max_play_length': 200
    }
}
