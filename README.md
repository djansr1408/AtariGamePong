## Atari Game Pong in Tensor Flow

Reinforcement learning project - Training agent to play Atari game Pong in Tensor Flow

# Introduction

In this project, reinforcement learning will be used for training an agent to play Pong game. For this purpose, specifically *policy gradients* method is analysed from its theoretical aspects to practical implementation. After I read Andrej Karpathy's blog, I was fascinated how simple but still effective this method could be. In his blog, Andrej expalained very briefly this approach and implemented it in Python from scratch. I wanted to do basically the same implementation, but in Tensor Flow instead. Just for the clarity, I will try to sum up the most important points about *policy gradients* and to explain environment which I used for game simulation.

# Environment

Team of OpenAI researchers have developed *gym* environment which contains several Atari games. In this project, the focus is on the Pong game. The goal of this game is to pass the ball by the opponent by hitting it under some angle. This is presented on the Figure 1. The agent may move up or down at any moment. 

<p align="center">
<img style="float: center;margin:0 auto; " align="center" src="./images/hqdefault.jpg">   
<div align="center">
Figure 1: Pong game environment
</div>
</p>

# *Policy gradients* method

The idea behind this method is to learn approximatively optimal policy \pi. This is done using simple neural network presented on Figure 2. The network will calculate probabilities for going up/down based on the input. 

<p align="center">
<img style="float: center;margin:0 auto; " align="center" src="./images/nn.png">   
<div align="center">
Figure 2: Neural network simulating policy /pi
</div>
</p>

The raw input is frame of the shape 210x160x3 which will be cropped to size 80x80x3 and then converted from RGB to gray matrix of size 80x80. This final matrix will be reshaped to 6400x1 array. In order to detect movement in the game, two adjacent frames are subracted and their difference is then reshaped in a mentioned way. Hidded layer contains 200 neurons. Output layer has 2 neurons, so that first one gives probability of going up, and the second of going down. These two probabilities are complementary (Pup = 1 - Pdown). 






