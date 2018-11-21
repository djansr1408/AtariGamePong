## Atari Game Pong in Tensor Flow

Reinforcement learning project - Training agent to play Atari game Pong in Tensor Flow

### About the project

In this project game agent is trained to play Atari game Pong using ***policy gradients*** method. The whole implementation is done in TensorFlow. After reading [Andrej Karpathy's blog](http://karpathy.github.io/2016/05/31/rl/) on reinforcement learning where he explained this method in reinforcement learning and published code for training agent from scratch in Python, I wanted to implement the same thing but in Tensor Flow instead. This helped me to better understand mentioned approach and gain some intuition on how agent learns to play the game without knowing anything about environment. Also, this was a great exercise in working with ***gym*** package which simulates the environment and implementing neural network in TF.

### Environment

Game environment is simulated using OpenAI package ***gym*** which contains support for many Atari games. It offers graphical represantation of the game, actions which agent can execute and rewards for that actions. More about the package could be found [here](https://gym.openai.com). 
In this project, the focus is on the Pong game. The goal of this game is to pass the ball by the opponent. This is presented on the Figure 1. The agent may move up or down at any moment. 

<p align="center">
<img style="float: center;margin:0 auto; " align="center" src="./images/hqdefault.jpg">   
<div align="center">
Figure 1: Pong game environment
</div>
</p>

### Simulation

At the begining agent behaves randomly and loses (game is over who ever first comes to 21) in almost 100% of games.
The agent is trained till the moment when the game becomes tied and there is 50% chance for win. On Figure 2 is represented one game with trained agent.
<p align="center">
<img style="float: center;margin:0 auto; " align="center" src="./pong-agent.gif">   
<div align="center">
Figure 2: Game simulation with trained agent(green)
</div>
</p>


