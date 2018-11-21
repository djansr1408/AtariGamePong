## Atari Game Pong in Tensor Flow

Reinforcement learning project - Training agent to play Atari game Pong in Tensor Flow

# Introduction

In this project game agent is trained to play Atari game Pong using ***policy gradients*** method. The whole implementation is done in TensorFlow. After reading Andrej Karpathy's blog on reinforcement learning where he explained this method in reinforcement learning and published code for training agent from scratch in Python, I wanted to implement the same thing but in Tensor Flow instead. This helped me to better understand mentioned approach and gain some intuition on how agent learns to play the game without knowing anything about environment. Also, this was a great exercise while working with ***gym*** package which simulates the environment and implementing neural network in TF.

# Environment

Game environment is simulated using OpenAI package ***gym*** which contains support for many Atari games. It offers graphical represantation of the game, actions which agent can execute and rewards for that actions. More about the package could be found [here](https://gym.openai.com). 
In this project, the focus is on the Pong game. The goal of this game is to pass the ball by the opponent. This is presented on the Figure 1. The agent may move up or down at any moment. 

<p align="center">
<img style="float: center;margin:0 auto; " align="center" src="./images/hqdefault.jpg">   
<div align="center">
Figure 1: Pong game environment
</div>
</p>

# *Policy gradients* method

The idea behind this method is to learn approximatively optimal policy $\pi. This is done using simple neural network presented on Figure 2. The network will calculate probabilities for going up/down based on the input. 

<p align="center">
<img style="float: center;margin:0 auto; " align="center" src="./images/nn.png">   
<div align="center">
Figure 2: Neural network simulating policy $$\pi$$
</div>
</p>

The raw input is frame of the shape 210x160x3 which will be cropped to size 80x80x3 and then converted from RGB to gray matrix of size 80x80. This final matrix will be reshaped to 6400x1 array. In order to detect movement in the game, two adjacent frames are subracted and their difference is then reshaped in a mentioned way. Hidded layer contains 200 neurons. Output layer has 2 neurons, so that first one gives probability of going up, and the second of going down. These two probabilities are complementary (Pup = 1 - Pdown). 






