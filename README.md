# Learning Tetris with Deep Q-Learning #

__Why reinforcement learning with Tetris?__

Deepmind has shown in several papers that it is possible to
use the same Deep Q-Network (DQN) to solve different atari
2600 games on a superhuman level 1 .
It would be interesting to see if it is possible to transfer these
techniques to a game that requires deeper understanding of
the game mechanics. Because of the great variance of possible
stones and the resulting task of planning Tetris is an NP-Hard
problem 2.
We set our task to use Reinforcement Learning (RL) with Con-
volutional Neural Networks (CNN) to learn optimal control
policies for Tetris.

![Environment](https://github.com/qway/dqn_tetris/blob/master/documentation/images/agent_env_loop.png)

__Architecture__

We employ a custom architecture using 1 dimensional convolutions to quickly reduce spatial size. After two additional
linear layers we are left with the q-value of the proposed action. All layers except the last one use ReLU as the
activation function. To address all possible tetromino placements, a batch of data consists of 34 possible placements 5
where the score is computed in parallel. In the end, the placement with the highest score is choosen and executed
with regard to an epsilon-greedy policy.

![Architecture](https://github.com/qway/dqn_tetris/blob/master/documentation/images/architecture.png)

__Conclusion__

1. To benchmark our agent performance we compared it to a random
playing agent, another implementation of two students from Stanford
and a (near) Perfect Bot. Since the bot can play forever we compare the average number of lines deleted after 1000 placed tetrominos.

| Name: | Our Agent | Random | Stanford | Perfect Bot |
| --- | --- | --- | --- | --- |
| Lines per 1000 tetrominos | 92 | 1 | 21 | 200 |


2. The network architecture from DQN was not really transferable. On the
one hand we did not need to apply downconvolution to the input, since
it was already small enough (22px × 10px) and on the other hand it
seems like more computing time would not solve the problem completely.
DeepMind trained for 50 mio steps (ca. 30 days).
3. Further work: As a next interesting step ”blind Tetris” could be to approached. Therefore we would recommend using a Recurrent Neural Network (RNN). Also transfer learning would be an interesting topic, which
could possibly lead to much faster learning success. The idea would be
that you start with one tetromino and once the agent plays optimal you
would add another tetromino and so on.

__Other work and gimmicks__

* Implementation of a simple Human Tetris environment for a good comparison and in preparation for supervised learning.
* Approaches of supervised learning, where we prefilled the experience buffer with recorded games played in the simple Human environment.
* Using Tensorboard for statistcs and evaluation.
* Twitch Live Stream to follow the learning progress.

# Issues: #
__Fix TetrisEnv() issue__

Fixed scripts.tetris_env local package import.
"scripts" Has to be marked as "roots folder" via right klick on "scripts" -> "Mark Directory as" -> "Sources Root".
Then TetrisEnv() has manually be added. (wait till red light bulb and then first option).
