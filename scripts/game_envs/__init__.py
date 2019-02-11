import numpy as np
import game_envs.params

# set the number of shapes parameter to the correct values
params.PARAMETERS['tetris_env']['number_of_shapes'] = max(max(y) for y in
                                                          [max(x) for x in
                                                           params.PARAMETERS['tetris']['tetris_shapes']])
