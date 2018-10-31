## Atari Game Pong in Tensor Flow

Reinforcement learning project - Training agent to play Atari game Pong in Tensor Flow

# Introduction

In this project, reinforcement learning will be used for training an agent to play Pong game. For this purpose, specifically *policy gradients* method is analysed from its theoretical aspects to practical implementation. After I read Andrej Karpathy's blog, I was fascinated how simple but still effective this method could be. In his blog, Andrej expalained very briefly this approach and implemented it in Python from scratch. I wanted to do basically the same implementation, but in Tensor Flow instead. Just for the clarity, I will try to sum up the most important points about *policy gradients* and to explain environment which I used for game simulation.

# Environment

Team of OpenAI researchers have developed *gym* envirnment which contains several Atari games. In this project, the focus is on the Pong game. The goal of this game is to pass the ball by the opponent by hitting it under some angle. This is presented on the Figure 1. 







